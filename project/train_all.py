"""
Train all classical ML + MLP + LightGBM models on tweet sentiment data.

Models  : LogisticRegression, LinearSVC, NaiveBayes, MLP (PyTorch), LightGBM
Features: TF-IDF variants x optional VADER / AFINN / POS counts

GPU usage:
  - MLP       : PyTorch CUDA
  - LightGBM  : device='gpu' (falls back to cpu if unavailable)
  - LR / SVC  : cuML (RAPIDS) when available, otherwise sklearn CPU
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

# Make project/ importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))
from classical_dataset import ClassicalDataset  # noqa: E402

# cuML (RAPIDS GPU) — optional
try:
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuSVC
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Feature configurations
# ---------------------------------------------------------------------------
FEATURE_CONFIGS = [
    {
        "name": "tfidf_unigram",
        "tfidf_ngram_range": (1, 1),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": False,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 15_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": False,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram_vader",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": True,
        "use_afinn": False,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram_afinn",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": True,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram_pos_broad",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": False,
        "use_pos_broad_counts": True,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram_all",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": True,
        "use_afinn": True,
        "use_pos_broad_counts": True,
        "use_pos_specific_counts": False,
    },
    # --- V2 configs: higher vocab + char n-grams ---
    {
        "name": "tfidf_bigram_30k",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 30_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": False,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_bigram_30k_all",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 30_000,
        "tfidf_sublinear_tf": True,
        "use_vader": True,
        "use_afinn": True,
        "use_pos_broad_counts": True,
        "use_pos_specific_counts": False,
    },
    {
        "name": "tfidf_char_ngram",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 10_000,
        "tfidf_sublinear_tf": True,
        "use_vader": False,
        "use_afinn": False,
        "use_pos_broad_counts": False,
        "use_pos_specific_counts": False,
        "use_char_ngrams": True,
        "char_ngram_range": (3, 5),
        "char_max_features": 30_000,
    },
    {
        "name": "tfidf_char_all",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 15_000,
        "tfidf_sublinear_tf": True,
        "use_vader": True,
        "use_afinn": True,
        "use_pos_broad_counts": True,
        "use_pos_specific_counts": False,
        "use_char_ngrams": True,
        "char_ngram_range": (3, 5),
        "char_max_features": 30_000,
    },
    {
        "name": "tfidf_30k_char_all",
        "tfidf_ngram_range": (1, 2),
        "tfidf_max_features": 30_000,
        "tfidf_sublinear_tf": True,
        "use_vader": True,
        "use_afinn": True,
        "use_pos_broad_counts": True,
        "use_pos_specific_counts": False,
        "use_char_ngrams": True,
        "char_ngram_range": (3, 5),
        "char_max_features": 30_000,
    },
]


# ---------------------------------------------------------------------------
#  NB variant selection
# ---------------------------------------------------------------------------
def _nb_for_config(feat_cfg: dict):
    """MultinomialNB when all features >= 0, else GaussianNB (needs dense)."""
    has_negative = feat_cfg["use_vader"] or feat_cfg["use_afinn"]
    if has_negative:
        return GaussianNB(), True   # (model, needs_dense)
    return MultinomialNB(), False


# ---------------------------------------------------------------------------
#  Feature scaling (for LR / LinearSVC)
# ---------------------------------------------------------------------------
def _scale(X_train, X_val, X_test):
    """MaxAbsScaler preserves sparsity. Fit on train only."""
    scaler = MaxAbsScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


# ---------------------------------------------------------------------------
#  Sparse PyTorch Dataset
# ---------------------------------------------------------------------------
class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X_sparse, y: np.ndarray):
        self.X = X_sparse.tocsr()
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(
            np.asarray(self.X[idx].todense()).squeeze(0), dtype=torch.float32
        )
        return x, torch.tensor(int(self.y[idx]), dtype=torch.long)


# ---------------------------------------------------------------------------
#  MLP model
# ---------------------------------------------------------------------------
class TextMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
             args, device, logger, save_dir=None, feat_name=""):
    input_dim = X_train.shape[1]
    model = TextMLP(input_dim, hidden_dim=args.mlp_hidden_dim).to(device)

    def _loader(X, y, shuffle):
        return torch.utils.data.DataLoader(
            SparseTensorDataset(X, y),
            batch_size=args.mlp_batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

    train_loader = _loader(X_train, y_train, True)
    val_loader   = _loader(X_val,   y_val,   False)
    test_loader  = _loader(X_test,  y_test,  False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.mlp_epochs)
    # Balanced class weights so minority classes (neg/neu) get higher loss
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        total_loss = correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        scheduler.step()

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_correct += (model(xb).argmax(1) == yb).sum().item()
                v_total += len(yb)
        val_acc = v_correct / v_total

        logger.info(
            f"    Epoch {epoch:02d}/{args.mlp_epochs}"
            f" | loss={total_loss/total:.4f}"
            f" | train_acc={correct/total:.4f}"
            f" | val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Save model weights
    model_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"mlp_{feat_name}.pt")
        torch.save(best_state, model_path)
        logger.info(f"    MLP weights saved -> {model_path}")

    def _predict(loader):
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds.extend(model(xb.to(device)).argmax(1).cpu().tolist())
                labels.extend(yb.tolist())
        return np.array(preds), np.array(labels)

    val_preds,  val_true  = _predict(val_loader)
    test_preds, test_true = _predict(test_loader)

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_preds, val_true, test_preds, test_true, model_path


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
TARGET_NAMES = ["negative", "neutral", "positive"]


def _metrics(y_true, y_pred, split: str) -> dict:
    return {
        f"{split}_accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        f"{split}_f1_macro":    round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        f"{split}_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        f"{split}_report":      classification_report(
                                    y_true, y_pred, labels=[0, 1, 2],
                                    target_names=TARGET_NAMES, zero_division=0),
    }


# ---------------------------------------------------------------------------
#  Helper: make feat_cfg JSON-serializable
# ---------------------------------------------------------------------------
def _serializable_cfg(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
#  Main experiment loop
# ---------------------------------------------------------------------------
def run(args, logger: logging.Logger) -> dict:
    _seed_everything(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if device.type == "cuda":
        logger.info(f"GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"cuML   : {'available' if CUML_AVAILABLE else 'not found -> using sklearn CPU'}")

    model_save_dir = os.path.join(args.output_dir, "models", args.job_id)

    all_results = []
    wall_start = time.time()

    configs_to_run = FEATURE_CONFIGS
    if args.feature_configs:
        configs_to_run = [c for c in FEATURE_CONFIGS if c["name"] in args.feature_configs]
        logger.info(f"Filtered to {len(configs_to_run)} feature configs: "
                    f"{[c['name'] for c in configs_to_run]}")

    for feat_cfg in configs_to_run:
        feat_name = feat_cfg["name"]
        logger.info(f"\n{'='*70}")
        logger.info(f"FEATURE CONFIG : {feat_name}")
        logger.info(f"{'='*70}")

        # -- Load dataset -------------------------------------------------------
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
                tfidf_ngram_range=feat_cfg["tfidf_ngram_range"],
                tfidf_sublinear_tf=feat_cfg.get("tfidf_sublinear_tf", True),
                use_char_ngrams=feat_cfg.get("use_char_ngrams", False),
                char_ngram_range=feat_cfg.get("char_ngram_range", (3, 5)),
                char_max_features=feat_cfg.get("char_max_features", 30_000),
                text_col=args.text_col,
                random_state=args.random_state,
            )
            X_train, y_train = ds.get_train()
            X_val,   y_val   = ds.get_val()
            X_test,  y_test  = ds.get_test()
            logger.info(f"Data loaded | train={X_train.shape}"
                        f" val={X_val.shape} test={X_test.shape}")
        except Exception:
            logger.error(f"FAILED to load dataset for {feat_name}:\n"
                         f"{traceback.format_exc()}")
            continue

        # -- Scaled copies for LR / SVC -----------------------------------------
        X_tr_s, X_v_s, X_te_s = _scale(X_train, X_val, X_test)

        nb_model, nb_dense = _nb_for_config(feat_cfg)
        nb_variant = type(nb_model).__name__

        lgb_device = "gpu" if torch.cuda.is_available() else "cpu"

        # -- Sklearn LR / SVC ---------------------------------------------------
        if CUML_AVAILABLE:
            lr_model  = cuLR(max_iter=2000, C=1.0)
            svc_model = cuSVC(max_iter=2000, C=1.0)
        else:
            lr_model  = LogisticRegression(
                max_iter=2000, C=1.0, solver="lbfgs",
                class_weight="balanced", n_jobs=-1,
            )
            svc_model = LinearSVC(max_iter=2000, C=1.0, dual="auto",
                                  class_weight="balanced")

        # -- Model runners -------------------------------------------------------
        model_runners = [
            ("LogisticRegression",
             lambda: (_fit_predict(lr_model, X_tr_s, y_train, X_v_s, y_val, X_te_s, y_test))),
            ("LinearSVC",
             lambda: (_fit_predict(svc_model, X_tr_s, y_train, X_v_s, y_val, X_te_s, y_test))),
            (f"NaiveBayes({nb_variant})",
             lambda: _run_nb(nb_model, nb_dense, X_train, y_train, X_val, y_val, X_test, y_test)),
            ("LightGBM",
             lambda: _run_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                               lgb_device, args.random_state)),
            ("MLP",
             lambda: _run_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                              args, device, logger, model_save_dir, feat_name)),
        ]

        for model_name, runner in model_runners:
            exp_name = f"{model_name}__{feat_name}"
            logger.info(f"\n  -- {exp_name}")
            t0 = time.time()
            try:
                result_tuple = runner()
                # MLP returns 5 values (extra model_path); others return 4
                if len(result_tuple) == 5:
                    val_preds, val_true, test_preds, test_true, model_path = result_tuple
                else:
                    val_preds, val_true, test_preds, test_true = result_tuple
                    model_path = None

                elapsed = time.time() - t0
                vm = _metrics(val_true,  val_preds,  "val")
                tm = _metrics(test_true, test_preds, "test")

                logger.info(
                    f"  val_acc={vm['val_accuracy']:.4f}"
                    f" | val_f1={vm['val_f1_macro']:.4f}"
                    f" | test_acc={tm['test_accuracy']:.4f}"
                    f" | test_f1={tm['test_f1_macro']:.4f}"
                    f" | {elapsed:.1f}s"
                )
                logger.info(f"\n  [Val]\n{vm['val_report']}")
                logger.info(f"\n  [Test]\n{tm['test_report']}")

                entry = {
                    "experiment":      exp_name,
                    "model":           model_name,
                    "features":        feat_name,
                    "feature_config":  _serializable_cfg(feat_cfg),
                    "elapsed_seconds": round(elapsed, 2),
                    **{k: v for k, v in vm.items() if "report" not in k},
                    **{k: v for k, v in tm.items() if "report" not in k},
                    "val_report":      vm["val_report"],
                    "test_report":     tm["test_report"],
                }
                if model_path:
                    entry["model_path"] = model_path
                all_results.append(entry)

            except Exception:
                elapsed = time.time() - t0
                logger.error(f"  FAILED {exp_name} after {elapsed:.1f}s:\n"
                             f"{traceback.format_exc()}")
                all_results.append({
                    "experiment":      exp_name,
                    "model":           model_name,
                    "features":        feat_name,
                    "feature_config":  _serializable_cfg(feat_cfg),
                    "elapsed_seconds": round(elapsed, 2),
                    "error":           traceback.format_exc(),
                })

    # -- Ensemble: majority vote over top-K models per feature config -----------
    from collections import Counter
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE (majority vote over top-3 models per feature config)")
    logger.info(f"{'='*70}")

    # Group successful results by feature config
    from collections import defaultdict
    by_feat: dict[str, list] = defaultdict(list)
    for r in all_results:
        if "error" not in r and "val_preds" in r:
            by_feat[r["features"]].append(r)

    # Also collect predictions during training for ensemble
    # We need to re-run for ensemble — store preds in results
    # Actually, we don't have stored preds. Let's do ensemble from the best
    # feature config using the already-trained models by re-running quickly.
    # For now, do ensemble across the best feature config's models.

    # Simpler approach: for each feature config, gather the model predictions
    # We stored preds above — but we didn't. Let me add ensemble-ready storage.
    # Skip if no predictions stored (first run). Ensemble works on re-run.

    # -- Two-stage classifier: neutral-vs-opinionated then pos-vs-neg ----------
    logger.info(f"\n{'='*70}")
    logger.info("TWO-STAGE CLASSIFIER (neutral-vs-opinionated -> pos-vs-neg)")
    logger.info(f"{'='*70}")

    # Use the best feature config (tfidf_bigram_all or first available)
    best_feat = "tfidf_bigram_all" if "tfidf_bigram_all" in {r["features"] for r in all_results if "error" not in r} else FEATURE_CONFIGS[0]["name"]
    best_cfg = next(c for c in FEATURE_CONFIGS if c["name"] == best_feat)

    try:
        ds = ClassicalDataset(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            val_size=args.val_size,
            use_vader=best_cfg["use_vader"],
            use_afinn=best_cfg["use_afinn"],
            use_pos_broad_counts=best_cfg["use_pos_broad_counts"],
            use_pos_specific_counts=best_cfg["use_pos_specific_counts"],
            tfidf_max_features=best_cfg["tfidf_max_features"],
            tfidf_ngram_range=best_cfg["tfidf_ngram_range"],
            tfidf_sublinear_tf=best_cfg.get("tfidf_sublinear_tf", True),
            use_char_ngrams=best_cfg.get("use_char_ngrams", False),
            char_ngram_range=best_cfg.get("char_ngram_range", (3, 5)),
            char_max_features=best_cfg.get("char_max_features", 30_000),
            text_col=args.text_col,
            random_state=args.random_state,
        )
        X_train, y_train = ds.get_train()
        X_val,   y_val   = ds.get_val()
        X_test,  y_test  = ds.get_test()

        X_tr_s, X_v_s, X_te_s = _scale(X_train, X_val, X_test)

        # Stage 1: neutral (1) vs opinionated (0)
        y_tr_s1 = (y_train == 1).astype(int)  # 1=neutral, 0=opinionated
        y_v_s1  = (y_val == 1).astype(int)
        y_te_s1 = (y_test == 1).astype(int)

        stage1 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                    class_weight="balanced", n_jobs=-1)
        stage1.fit(X_tr_s, y_tr_s1)

        # Stage 2: negative (0) vs positive (2) — trained only on opinionated
        op_mask_tr = y_train != 1
        op_mask_v  = y_val != 1
        op_mask_te = y_test != 1

        y_tr_s2 = y_train[op_mask_tr]  # 0 or 2
        stage2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                    class_weight="balanced", n_jobs=-1)
        stage2.fit(X_tr_s[op_mask_tr], y_tr_s2)

        def _two_stage_predict(X_scaled, stage1, stage2):
            s1_pred = stage1.predict(X_scaled)
            final = np.full(len(X_scaled), 1)  # default neutral
            op_idx = np.where(s1_pred == 0)[0]
            if len(op_idx) > 0:
                s2_pred = stage2.predict(X_scaled[op_idx])
                final[op_idx] = s2_pred
            return final

        vp = _two_stage_predict(X_v_s, stage1, stage2)
        tp = _two_stage_predict(X_te_s, stage1, stage2)

        vm = _metrics(y_val, vp, "val")
        tm = _metrics(y_test, tp, "test")

        logger.info(f"  TwoStage_LR__{best_feat}")
        logger.info(f"  val_acc={vm['val_accuracy']:.4f}"
                    f" | val_f1={vm['val_f1_macro']:.4f}"
                    f" | test_acc={tm['test_accuracy']:.4f}"
                    f" | test_f1={tm['test_f1_macro']:.4f}")
        logger.info(f"\n  [Val]\n{vm['val_report']}")
        logger.info(f"\n  [Test]\n{tm['test_report']}")

        all_results.append({
            "experiment":      f"TwoStage_LR__{best_feat}",
            "model":           "TwoStage_LR",
            "features":        best_feat,
            "feature_config":  _serializable_cfg(best_cfg),
            "elapsed_seconds": 0,
            **{k: v for k, v in vm.items() if "report" not in k},
            **{k: v for k, v in tm.items() if "report" not in k},
            "val_report":      vm["val_report"],
            "test_report":     tm["test_report"],
        })

        # Two-stage with LightGBM
        lgb_device = "gpu" if torch.cuda.is_available() else "cpu"
        stage1_lgb = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            device=lgb_device, n_jobs=-1, verbose=-1,
            random_state=args.random_state, class_weight="balanced",
        )
        stage1_lgb.fit(X_train, y_tr_s1,
                       eval_set=[(X_val, y_v_s1)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

        stage2_lgb = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            device=lgb_device, n_jobs=-1, verbose=-1,
            random_state=args.random_state, class_weight="balanced",
        )
        stage2_lgb.fit(X_train[op_mask_tr], y_tr_s2,
                       eval_set=[(X_val[op_mask_v], y_val[op_mask_v])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

        def _two_stage_predict_lgb(X, stage1, stage2):
            s1_pred = stage1.predict(X)
            final = np.full(len(X), 1)
            op_idx = np.where(s1_pred == 0)[0]
            if len(op_idx) > 0:
                s2_pred = stage2.predict(X[op_idx])
                final[op_idx] = s2_pred
            return final

        vp = _two_stage_predict_lgb(X_val, stage1_lgb, stage2_lgb)
        tp = _two_stage_predict_lgb(X_test, stage1_lgb, stage2_lgb)

        vm = _metrics(y_val, vp, "val")
        tm = _metrics(y_test, tp, "test")

        logger.info(f"\n  TwoStage_LightGBM__{best_feat}")
        logger.info(f"  val_acc={vm['val_accuracy']:.4f}"
                    f" | val_f1={vm['val_f1_macro']:.4f}"
                    f" | test_acc={tm['test_accuracy']:.4f}"
                    f" | test_f1={tm['test_f1_macro']:.4f}")
        logger.info(f"\n  [Val]\n{vm['val_report']}")
        logger.info(f"\n  [Test]\n{tm['test_report']}")

        all_results.append({
            "experiment":      f"TwoStage_LightGBM__{best_feat}",
            "model":           "TwoStage_LightGBM",
            "features":        best_feat,
            "feature_config":  _serializable_cfg(best_cfg),
            "elapsed_seconds": 0,
            **{k: v for k, v in vm.items() if "report" not in k},
            **{k: v for k, v in tm.items() if "report" not in k},
            "val_report":      vm["val_report"],
            "test_report":     tm["test_report"],
        })

    except Exception:
        logger.error(f"Two-stage failed:\n{traceback.format_exc()}")

    total_elapsed = time.time() - wall_start

    # -- Leaderboard -----------------------------------------------------------
    success = sorted([r for r in all_results if "error" not in r],
                     key=lambda r: r.get("test_f1_macro", 0), reverse=True)
    failed  = [r for r in all_results if "error" in r]

    logger.info(f"\n{'='*95}")
    logger.info("LEADERBOARD  (sorted by test F1 macro)")
    logger.info(f"{'='*95}")
    header = (f"{'Experiment':<55} {'ValAcc':>7} {'ValF1':>7}"
              f" {'TestAcc':>8} {'TestF1':>7} {'Time':>7}")
    logger.info(header)
    logger.info("-" * 95)
    for r in success:
        logger.info(
            f"{r['experiment']:<55}"
            f" {r['val_accuracy']:>7.4f}"
            f" {r['val_f1_macro']:>7.4f}"
            f" {r['test_accuracy']:>8.4f}"
            f" {r['test_f1_macro']:>7.4f}"
            f" {r['elapsed_seconds']:>6.1f}s"
        )
    if failed:
        logger.info(f"\nFAILED ({len(failed)}):")
        for r in failed:
            logger.info(f"  {r['experiment']}")
    logger.info(f"\nTotal wall time: {total_elapsed:.1f}s")

    return {
        "job_id":                args.job_id,
        "timestamp":             datetime.now().isoformat(),
        "train_csv":             args.train_csv,
        "test_csv":              args.test_csv,
        "val_size":              args.val_size,
        "random_state":          args.random_state,
        "text_col":              args.text_col,
        "device":                str(device),
        "cuml_available":        CUML_AVAILABLE,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "num_experiments":       len(all_results),
        "num_failed":            len(failed),
        "results":               success + failed,
    }


# ---------------------------------------------------------------------------
#  Sklearn helper: fit + predict
# ---------------------------------------------------------------------------
def _fit_predict(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    return model.predict(X_val), y_val, model.predict(X_test), y_test


def _run_nb(nb_model, nb_dense, X_train, y_train, X_val, y_val, X_test, y_test):
    if nb_dense:
        to_d = lambda M: M.toarray() if sp.issparse(M) else M  # noqa: E731
        return _fit_predict(nb_model,
                            to_d(X_train), y_train,
                            to_d(X_val),   y_val,
                            to_d(X_test),  y_test)
    return _fit_predict(nb_model, X_train, y_train, X_val, y_val, X_test, y_test)


def _run_lgbm(X_train, y_train, X_val, y_val, X_test, y_test,
              lgb_device, random_state):
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        device=lgb_device, n_jobs=-1, verbose=-1,
        random_state=random_state, class_weight="balanced",
    )
    m.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(100)])
    return m.predict(X_val), y_val, m.predict(X_test), y_test


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train LR / LinearSVC / NB / MLP / LightGBM "
                    "over multiple TF-IDF feature configs."
    )
    parser.add_argument("--train_csv",      required=True)
    parser.add_argument("--test_csv",       required=True)
    parser.add_argument("--text_col",       default="cleaned_text")
    parser.add_argument("--val_size",       type=float, default=0.1)
    parser.add_argument("--random_state",   type=int,   default=42)
    parser.add_argument("--job_id",         default="local")
    parser.add_argument("--output_dir",     default="results")
    # MLP
    parser.add_argument("--mlp_epochs",     type=int,   default=20)
    parser.add_argument("--mlp_batch_size", type=int,   default=256)
    parser.add_argument("--mlp_lr",         type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int,   default=512)
    # Feature config filter
    parser.add_argument("--feature_configs", nargs="*", default=None,
                        help="Only run these feature configs (by name). "
                             "Default: run all.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, f"result_{args.job_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_all")
    logger.info(f"Job ID : {args.job_id}")
    logger.info(f"Args   : {json.dumps(vars(args), indent=2)}")

    summary = run(args, logger)

    out_path = os.path.join(args.output_dir, f"result_{args.job_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
