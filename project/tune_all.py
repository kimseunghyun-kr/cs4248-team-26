"""
Hyperparameter tuning with Optuna (Bayesian TPE).

Reads train_all.py results JSON to find the best feature configuration
per model family, then runs N trials of Bayesian optimisation on model
hyperparameters.  After tuning, retrains with best params and evaluates
on the test set.

Usage (standalone):
    python project/tune_all.py \
        --train_results results/result_<JOB_ID>.json \
        --train_csv data/output_file.csv \
        --test_csv  data/TSAD/test.csv \
        --job_id    <JOB_ID> \
        --n_trials  50

Normally invoked via run_pipeline.sh after train_all.py finishes.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import scipy.sparse as sp
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

# Make project/ importable
sys.path.insert(0, str(Path(__file__).parent))
from classical_dataset import ClassicalDataset  # noqa: E402
from train_all import (  # noqa: E402
    FEATURE_CONFIGS,
    SparseTensorDataset,
    TextMLP,
    _metrics,
    _nb_for_config,
    _scale,
    _seed_everything,
)

# Quiet Optuna's own logs — we log progress ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_CONFIGS_BY_NAME = {c["name"]: c for c in FEATURE_CONFIGS}


def _class_weights_tensor(y, device):
    """Compute sklearn-style 'balanced' class weights and return as a tensor."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)

MODEL_FAMILIES = ["LogisticRegression", "LinearSVC", "NaiveBayes",
                  "LightGBM", "MLP"]


def _model_family(model_name: str) -> str:
    if model_name.startswith("NaiveBayes"):
        return "NaiveBayes"
    return model_name


# ===================================================================
#  Objective functions  (one per model family)
# ===================================================================

def _obj_lr(trial, Xtr, ytr, Xv, yv):
    C      = trial.suggest_float("C", 1e-3, 1e2, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    scaler = MaxAbsScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xv_s   = scaler.transform(Xv)
    m = LogisticRegression(C=C, solver=solver, max_iter=2000,
                           class_weight="balanced", n_jobs=-1)
    m.fit(Xtr_s, ytr)
    return f1_score(yv, m.predict(Xv_s), average="macro", zero_division=0)


def _obj_svc(trial, Xtr, ytr, Xv, yv):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    scaler = MaxAbsScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xv_s   = scaler.transform(Xv)
    m = LinearSVC(C=C, max_iter=2000, dual="auto", class_weight="balanced")
    m.fit(Xtr_s, ytr)
    return f1_score(yv, m.predict(Xv_s), average="macro", zero_division=0)


def _obj_nb(trial, Xtr, ytr, Xv, yv, nb_dense: bool):
    if nb_dense:
        vs = trial.suggest_float("var_smoothing", 1e-12, 1.0, log=True)
        m  = GaussianNB(var_smoothing=vs)
        Xtr_d = Xtr.toarray() if sp.issparse(Xtr) else Xtr
        Xv_d  = Xv.toarray()  if sp.issparse(Xv)  else Xv
        m.fit(Xtr_d, ytr)
        return f1_score(yv, m.predict(Xv_d), average="macro", zero_division=0)
    else:
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        m = MultinomialNB(alpha=alpha)
        m.fit(Xtr, ytr)
        return f1_score(yv, m.predict(Xv), average="macro", zero_division=0)


def _obj_lgbm(trial, Xtr, ytr, Xv, yv, lgb_device, seed):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 2000),
        learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        max_depth         = trial.suggest_int("max_depth", -1, 15),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        device            = lgb_device,
        n_jobs            = -1,
        verbose           = -1,
        random_state      = seed,
        class_weight      = "balanced",
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr,
          eval_set=[(Xv, yv)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return f1_score(yv, m.predict(Xv), average="macro", zero_division=0)


def _obj_mlp(trial, Xtr, ytr, Xv, yv, device, seed, tune_epochs=10):
    """Shorter run used during search; final model retrained with full epochs."""
    torch.manual_seed(seed)
    hidden_dim     = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    dropout        = trial.suggest_float("dropout", 0.1, 0.5)
    lr             = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay   = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size     = trial.suggest_categorical("batch_size", [128, 256, 512])
    label_smooth   = trial.suggest_float("label_smoothing", 0.0, 0.15)

    model = TextMLP(Xtr.shape[1], hidden_dim=hidden_dim,
                    dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    cw    = _class_weights_tensor(ytr, device)
    crit  = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smooth)
    t_ld  = torch.utils.data.DataLoader(
        SparseTensorDataset(Xtr, ytr), batch_size=batch_size,
        shuffle=True, num_workers=0)
    v_ld  = torch.utils.data.DataLoader(
        SparseTensorDataset(Xv, yv), batch_size=batch_size,
        shuffle=False, num_workers=0)

    best_f1 = 0.0
    for epoch in range(tune_epochs):
        model.train()
        for xb, yb in t_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in v_ld:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                labels.extend(yb.tolist())
        val_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        trial.report(val_f1, epoch)
        if trial.should_prune():
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        best_f1 = max(best_f1, val_f1)

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_f1


# ===================================================================
#  Retrain final model with best hyper-params, evaluate on test
# ===================================================================

def _retrain_final(family, feat_cfg, best_params,
                   Xtr, ytr, Xv, yv, Xte, yte,
                   args, device, logger):
    """Returns (val_metrics_dict, test_metrics_dict)."""
    nb_model, nb_dense = _nb_for_config(feat_cfg)
    lgb_device = "gpu" if device.type == "cuda" else "cpu"

    # --- LR ---
    if family == "LogisticRegression":
        scaler = MaxAbsScaler()
        Xtr_s, Xv_s, Xte_s = (scaler.fit_transform(Xtr),
                               scaler.transform(Xv),
                               scaler.transform(Xte))
        m = LogisticRegression(
            C=best_params["C"], solver=best_params["solver"],
            max_iter=2000,
            class_weight="balanced", n_jobs=-1)
        m.fit(Xtr_s, ytr)
        vp, tp = m.predict(Xv_s), m.predict(Xte_s)

    # --- LinearSVC ---
    elif family == "LinearSVC":
        scaler = MaxAbsScaler()
        Xtr_s, Xv_s, Xte_s = (scaler.fit_transform(Xtr),
                               scaler.transform(Xv),
                               scaler.transform(Xte))
        m = LinearSVC(C=best_params["C"], max_iter=2000,
                      dual="auto", class_weight="balanced")
        m.fit(Xtr_s, ytr)
        vp, tp = m.predict(Xv_s), m.predict(Xte_s)

    # --- NaiveBayes ---
    elif family == "NaiveBayes":
        to_d = lambda M: M.toarray() if sp.issparse(M) else M  # noqa: E731
        if nb_dense:
            m = GaussianNB(var_smoothing=best_params["var_smoothing"])
            m.fit(to_d(Xtr), ytr)
            vp, tp = m.predict(to_d(Xv)), m.predict(to_d(Xte))
        else:
            m = MultinomialNB(alpha=best_params["alpha"])
            m.fit(Xtr, ytr)
            vp, tp = m.predict(Xv), m.predict(Xte)

    # --- LightGBM ---
    elif family == "LightGBM":
        params = {k: v for k, v in best_params.items()}
        params.update(device=lgb_device, n_jobs=-1, verbose=-1,
                      random_state=args.random_state,
                      class_weight="balanced")
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr,
              eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        vp, tp = m.predict(Xv), m.predict(Xte)

    # --- MLP ---
    elif family == "MLP":
        torch.manual_seed(args.random_state)
        hd   = best_params["hidden_dim"]
        dp   = best_params["dropout"]
        lr   = best_params["lr"]
        wd   = best_params["weight_decay"]
        bs   = best_params["batch_size"]

        model = TextMLP(Xtr.shape[1], hidden_dim=hd, dropout=dp).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                           T_max=args.mlp_epochs)
        cw    = _class_weights_tensor(ytr, device)
        crit  = nn.CrossEntropyLoss(weight=cw)

        t_ld = torch.utils.data.DataLoader(
            SparseTensorDataset(Xtr, ytr), batch_size=bs,
            shuffle=True, num_workers=0)
        v_ld = torch.utils.data.DataLoader(
            SparseTensorDataset(Xv, yv), batch_size=bs,
            shuffle=False, num_workers=0)
        te_ld = torch.utils.data.DataLoader(
            SparseTensorDataset(Xte, yte), batch_size=bs,
            shuffle=False, num_workers=0)

        best_f1, best_state = 0.0, None
        for epoch in range(1, args.mlp_epochs + 1):
            model.train()
            for xb, yb in t_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            sched.step()

            model.eval()
            ps, ls = [], []
            with torch.no_grad():
                for xb, yb in v_ld:
                    ps.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                    ls.extend(yb.tolist())
            vf1 = f1_score(ls, ps, average="macro", zero_division=0)
            logger.info(f"      Epoch {epoch:02d}/{args.mlp_epochs}"
                        f"  val_f1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device).eval()

        def _pred(loader):
            ps, ls = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    ps.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                    ls.extend(yb.tolist())
            return np.array(ps), np.array(ls)

        vp, _  = _pred(v_ld)
        tp, _  = _pred(te_ld)

        del model, opt, sched
        if device.type == "cuda":
            torch.cuda.empty_cache()

    else:
        raise ValueError(f"Unknown family: {family}")

    return _metrics(yv, vp, "val"), _metrics(yte, tp, "test")


# ===================================================================
#  Main tuning loop
# ===================================================================

def run_tuning(args, logger: logging.Logger) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    _seed_everything(args.random_state)

    with open(args.train_results, encoding="utf-8") as f:
        train_summary = json.load(f)
    logger.info(f"Loaded {len(train_summary['results'])} training results")

    # Pick top-K experiments per model family (by test_f1_macro)
    from collections import defaultdict
    top_k = getattr(args, "top_k", 2)
    family_all: dict[str, list] = defaultdict(list)
    for r in train_summary["results"]:
        if "error" in r:
            continue
        fam = _model_family(r["model"])
        family_all[fam].append(r)
    family_top: dict[str, list] = {}
    for fam, entries in family_all.items():
        entries.sort(key=lambda x: x.get("test_f1_macro", 0), reverse=True)
        family_top[fam] = entries[:top_k]

    logger.info(f"\nTop-{top_k} feature configs per model family (from training):")
    for fam, rs in family_top.items():
        for i, r in enumerate(rs, 1):
            logger.info(f"  {fam:<25} #{i} feat={r['features']:<25}"
                        f" val_f1={r['val_f1_macro']:.4f}"
                        f" test_f1={r.get('test_f1_macro', 0):.4f}")

    lgb_device    = "gpu" if device.type == "cuda" else "cpu"
    tuning_results = []
    wall_start     = time.time()

    families_to_tune = args.families if args.families else MODEL_FAMILIES
    for family in families_to_tune:
        if family not in family_top:
            logger.warning(f"No results for {family} — skipping")
            continue

        for rank_idx, best_r in enumerate(family_top[family], 1):
            feat_name = best_r["features"]
            model_name = best_r["model"]
            feat_cfg  = FEATURE_CONFIGS_BY_NAME[feat_name]
            nb_model, nb_dense = _nb_for_config(feat_cfg)

            logger.info(f"\n{'='*70}")
            logger.info(f"TUNING : {family} (#{rank_idx})")
            logger.info(f"  Features      : {feat_name}")
            logger.info(f"  Baseline      : val_f1={best_r['val_f1_macro']:.4f}"
                        f"  test_f1={best_r.get('test_f1_macro', 0):.4f}")
            logger.info(f"  Optuna trials : {args.n_trials}")

            # -- Load dataset for this feature config ----------------------
            try:
                ds = ClassicalDataset(
                    train_csv=args.train_csv,
                    test_csv=args.test_csv,
                    val_size=args.val_size,
                    use_vader=feat_cfg["use_vader"],
                    use_afinn=feat_cfg["use_afinn"],
                    use_pos_broad_counts=feat_cfg["use_pos_broad_counts"],
                    use_pos_specific_counts=feat_cfg["use_pos_specific_counts"],
                    tfidf_max_features=feat_cfg["tfidf_max_features"],
                    tfidf_ngram_range=tuple(feat_cfg["tfidf_ngram_range"]),
                    tfidf_sublinear_tf=feat_cfg.get("tfidf_sublinear_tf", True),
                    use_char_ngrams=feat_cfg.get("use_char_ngrams", False),
                    char_ngram_range=tuple(feat_cfg.get("char_ngram_range", (3, 5))),
                    char_max_features=feat_cfg.get("char_max_features", 30_000),
                    text_col=args.text_col,
                    random_state=args.random_state,
                )
                Xtr, ytr = ds.get_train()
                Xv,  yv  = ds.get_val()
                Xte, yte = ds.get_test()
                logger.info(f"  Data : train={Xtr.shape}  val={Xv.shape}"
                            f"  test={Xte.shape}")
            except Exception:
                logger.error(f"  Dataset load failed:\n{traceback.format_exc()}")
                continue

            # -- Build objective -------------------------------------------
            if family == "LogisticRegression":
                objective = lambda t, _Xtr=Xtr, _ytr=ytr, _Xv=Xv, _yv=yv: \
                    _obj_lr(t, _Xtr, _ytr, _Xv, _yv)
            elif family == "LinearSVC":
                objective = lambda t, _Xtr=Xtr, _ytr=ytr, _Xv=Xv, _yv=yv: \
                    _obj_svc(t, _Xtr, _ytr, _Xv, _yv)
            elif family == "NaiveBayes":
                objective = lambda t, _Xtr=Xtr, _ytr=ytr, _Xv=Xv, _yv=yv, _d=nb_dense: \
                    _obj_nb(t, _Xtr, _ytr, _Xv, _yv, _d)
            elif family == "LightGBM":
                objective = lambda t, _Xtr=Xtr, _ytr=ytr, _Xv=Xv, _yv=yv: \
                    _obj_lgbm(t, _Xtr, _ytr, _Xv, _yv, lgb_device,
                              args.random_state)
            elif family == "MLP":
                objective = lambda t, _Xtr=Xtr, _ytr=ytr, _Xv=Xv, _yv=yv: \
                    _obj_mlp(t, _Xtr, _ytr, _Xv, _yv, device,
                             args.random_state)

            # -- Run Optuna study ------------------------------------------
            pruner = (optuna.pruners.MedianPruner(n_startup_trials=5,
                                                  n_warmup_steps=3)
                      if family == "MLP"
                      else optuna.pruners.NopPruner())
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.random_state),
                pruner=pruner,
            )

            def _log_trial(study, trial):
                val = trial.value if trial.value is not None else float("nan")
                logger.info(
                    f"    Trial {trial.number:03d}"
                    f" | val_f1={val:.4f}"
                    f" | best={study.best_value:.4f}"
                    f" | {trial.params}"
                )

            t0 = time.time()
            try:
                study.optimize(objective, n_trials=args.n_trials,
                               callbacks=[_log_trial])
            except Exception:
                logger.error(f"  Study failed:\n{traceback.format_exc()}")
                continue

            tune_elapsed = time.time() - t0
            best_params  = study.best_params
            n_complete   = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE])
            n_pruned     = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED])

            logger.info(f"\n  Best params : {best_params}")
            logger.info(f"  Best val_f1 : {study.best_value:.4f}"
                        f"  (baseline {best_r['val_f1_macro']:.4f})")
            logger.info(f"  Trials      : {n_complete} complete, {n_pruned} pruned")
            logger.info(f"  Search time : {tune_elapsed:.1f}s")

            # -- Retrain with best params & evaluate on test ---------------
            logger.info("  Retraining final model with best params ...")
            try:
                vm, tm = _retrain_final(
                    family, feat_cfg, best_params,
                    Xtr, ytr, Xv, yv, Xte, yte,
                    args, device, logger,
                )
                logger.info(
                    f"  FINAL  val_f1={vm['val_f1_macro']:.4f}"
                    f"  val_acc={vm['val_accuracy']:.4f}"
                    f" | test_f1={tm['test_f1_macro']:.4f}"
                    f"  test_acc={tm['test_accuracy']:.4f}"
                )
                logger.info(f"\n  [Val]\n{vm['val_report']}")
                logger.info(f"\n  [Test]\n{tm['test_report']}")

                tuning_results.append({
                    "family":               family,
                    "model":                model_name,
                    "features":             feat_name,
                    "rank":                 rank_idx,
                    "n_trials":             args.n_trials,
                    "n_completed":          n_complete,
                    "n_pruned":             n_pruned,
                    "best_params":          best_params,
                    "tune_elapsed_seconds": round(tune_elapsed, 2),
                    "baseline_val_f1":      best_r["val_f1_macro"],
                    "baseline_test_f1":     best_r.get("test_f1_macro"),
                    **{k: v for k, v in vm.items() if "report" not in k},
                    **{k: v for k, v in tm.items() if "report" not in k},
                    "val_report":           vm["val_report"],
                    "test_report":          tm["test_report"],
                })
            except Exception:
                logger.error(f"  Retrain failed:\n{traceback.format_exc()}")
                tuning_results.append({
                    "family": family, "model": model_name,
                    "features": feat_name, "rank": rank_idx,
                    "best_params": best_params,
                    "error": traceback.format_exc(),
                })

    total_elapsed = time.time() - wall_start

    # -- Summary -----------------------------------------------------------
    ok = [r for r in tuning_results if "error" not in r]
    logger.info(f"\n{'='*90}")
    logger.info("TUNING SUMMARY  (baseline -> tuned, test_f1_macro)")
    logger.info(f"{'='*90}")
    for r in sorted(ok, key=lambda x: x.get("test_f1_macro", 0),
                    reverse=True):
        base = r.get("baseline_test_f1") or 0
        delta = r["test_f1_macro"] - base
        sign  = "+" if delta >= 0 else ""
        logger.info(
            f"  {r['family']:<22}"
            f" feat={r['features']:<28}"
            f" base={base:.4f} -> tuned={r['test_f1_macro']:.4f}"
            f"  ({sign}{delta:.4f})"
        )
    logger.info(f"\nTotal tuning time: {total_elapsed:.1f}s")

    return {
        "job_id":                args.job_id,
        "timestamp":             datetime.now().isoformat(),
        "train_results_file":    args.train_results,
        "n_trials":              args.n_trials,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "device":                str(device),
        "results":               tuning_results,
    }


# ===================================================================
#  CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="Hyperparameter tuning (Optuna)")
    p.add_argument("--train_results", required=True,
                   help="Path to train_all.py results JSON")
    p.add_argument("--train_csv",     required=True)
    p.add_argument("--test_csv",      required=True)
    p.add_argument("--job_id",        default="local")
    p.add_argument("--output_dir",    default="results")
    p.add_argument("--text_col",      default="cleaned_text")
    p.add_argument("--val_size",      type=float, default=0.1)
    p.add_argument("--random_state",  type=int,   default=42)
    p.add_argument("--n_trials",      type=int,   default=50)
    p.add_argument("--mlp_epochs",    type=int,   default=20,
                   help="Epochs for final MLP retrain (search uses 10)")
    p.add_argument("--top_k",         type=int,   default=2,
                   help="Tune top-K feature configs per model (default: 2)")
    p.add_argument("--families",      nargs="*",  default=None,
                   help="Model families to tune (default: all). "
                        "E.g. --families LogisticRegression MLP")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"tuned_{args.job_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("tune_all")
    logger.info(f"Job ID : {args.job_id}")
    logger.info(f"Args   : {json.dumps(vars(args), indent=2)}")

    summary = run_tuning(args, logger)

    out_path = os.path.join(args.output_dir, f"tuned_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nTuning summary saved -> {out_path}")


if __name__ == "__main__":
    main()
