"""
Visualize cached condition embeddings with a shared pooled projection.

For PCA, the basis is fit once on normalized embeddings pooled across
conditions so before/after panels stay in the same coordinate system and
actual movement is visible.

For t-SNE / UMAP, the embedding is fit jointly on the pooled normalized
sample embeddings, so all conditions still live in one shared visualization
space, but the result should be interpreted as a neighborhood view rather
than a linear axis-aligned view.

Run from project/ directory:
  python pipeline/plot_pca.py --cache_dir /path/to/cache/roberta
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Failed to import PyTorch before plotting.\n"
        "This script loads cached '.pt' embedding files, so it needs a working "
        "PyTorch runtime even though the PCA itself is CPU-side.\n\n"
        "If you see an error like 'libtorch_cuda.so: failed to map segment from "
        "shared object', that usually means the current node cannot load the "
        "CUDA-enabled PyTorch build. On this cluster, the simplest fix is to run "
        "the plotting command on an allocated compute node or inside the same "
        "Slurm environment used for training.\n\n"
        f"Original import error: {exc}"
    ) from exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_COLORS = {
    0: "#d55e00",
    1: "#7f7f7f",
    2: "#009e73",
}
def _build_condition_specs() -> OrderedDict:
    specs = OrderedDict(
        [
            ("B1 (raw)", {"slug": "b1_raw"}),
            ("D1 (debias_vl)", {"slug": "d1_debias_vl"}),
            ("D2 (CBDC)", {"slug": "d2_cbdc"}),
        ]
    )
    if os.environ.get("INCLUDE_D25", "1") == "1":
        specs["D2.5 (CBDC no-label-select)"] = {"slug": "d25_cbdc_no_label_select"}
    specs["D3 (debias_vl->CBDC)"] = {"slug": "d3_debias_vl_cbdc"}
    if os.environ.get("INCLUDE_D4") == "1":
        specs["D4 (adv-discovery->CBDC)"] = {"slug": "d4_adv_discovery_cbdc"}
    return specs


CONDITION_SPECS = _build_condition_specs()
LABEL_ALIASES = {
    label.lower(): label
    for label in CONDITION_SPECS
}
LABEL_ALIASES.update(
    {
        spec["slug"]: label
        for label, spec in CONDITION_SPECS.items()
    }
)


def _default_cache_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("CACHE_DIR", os.path.join(root, "cache"))


def _condition_slug(condition_label: str) -> str:
    return CONDITION_SPECS[condition_label]["slug"]


def _raw_split_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"z_tweet_{split}.pt")


def _condition_dir(cache_dir: str, condition_label: str) -> str:
    return os.path.join(cache_dir, "conditions", _condition_slug(condition_label))


def _condition_artifact_path(cache_dir: str, condition_label: str, filename: str) -> str:
    return os.path.join(_condition_dir(cache_dir, condition_label), filename)


def _split_path(cache_dir: str, condition_label: str, split: str) -> str:
    if condition_label == "B1 (raw)":
        return _raw_split_path(cache_dir, split)
    return os.path.join(_condition_dir(cache_dir, condition_label), f"z_tweet_{split}.pt")


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _resolve_condition_labels(requested: list[str] | None, cache_dir: str, split: str) -> list[str]:
    if requested:
        resolved = []
        for item in requested:
            key = item.strip().lower()
            if key not in LABEL_ALIASES:
                raise KeyError(f"Unknown condition: {item}")
            resolved.append(LABEL_ALIASES[key])
        return resolved

    available = []
    for label in CONDITION_SPECS:
        if os.path.exists(_split_path(cache_dir, label, split)):
            available.append(label)
    if not available:
        raise FileNotFoundError(
            f"No cached split embeddings found for split='{split}' under {cache_dir}."
        )
    return available


def _load_condition_payload(cache_dir: str, condition_label: str, split: str) -> dict:
    path = _split_path(cache_dir, condition_label, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached split for {condition_label}: {path}")
    return torch.load(path, map_location="cpu")


def _load_condition_prototypes(cache_dir: str, condition_label: str) -> torch.Tensor | None:
    path = _condition_artifact_path(cache_dir, condition_label, "class_prompt_prototypes.pt")
    if not os.path.exists(path):
        return None
    return _normalize(torch.load(path, map_location="cpu"))


def _compute_centroids(coords: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    centroids = {}
    for label_id in range(len(LABEL_NAMES)):
        mask = labels == label_id
        if int(mask.sum()) == 0:
            continue
        centroids[label_id] = coords[mask].mean(dim=0)
    return centroids


def _build_output_path(cache_dir: str, split: str, n_components: int, output: str | None) -> str:
    if output:
        return output
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_name = os.path.basename(os.path.normpath(cache_dir))
    stem = "pca3d" if n_components == 3 else "pca"
    return os.path.join(root, "results", f"{stem}_{cache_name}_{split}.png")


def _build_output_path_for_method(
    cache_dir: str,
    split: str,
    method: str,
    n_components: int,
    output: str | None,
) -> str:
    if output:
        return output
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_name = os.path.basename(os.path.normpath(cache_dir))
    stem = method
    if n_components == 3:
        stem = f"{stem}3d"
    return os.path.join(root, "results", f"{stem}_{cache_name}_{split}.png")


def _compute_projection(
    method: str,
    n_components: int,
    fit_bank: torch.Tensor,
    embeddings_by_condition: dict[str, torch.Tensor],
    prototypes_by_condition_raw: dict[str, torch.Tensor | None],
    tsne_perplexity: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None], str | None]:
    fit_bank_np = fit_bank.numpy()

    if method == "pca":
        projector = PCA(n_components=n_components)
        projector.fit(fit_bank_np)
        projected_embeddings = {
            label: torch.from_numpy(projector.transform(embeddings.numpy())).float()
            for label, embeddings in embeddings_by_condition.items()
        }
        projected_prototypes = {}
        for condition_label, prototypes in prototypes_by_condition_raw.items():
            if prototypes is None:
                projected_prototypes[condition_label] = None
                continue
            if prototypes.shape[1] != fit_bank.shape[1]:
                raise ValueError(
                    f"Prototype dimensionality mismatch for {condition_label}: "
                    f"prototype dim={prototypes.shape[1]} but embedding dim={fit_bank.shape[1]}"
                )
            projected_prototypes[condition_label] = torch.from_numpy(
                projector.transform(prototypes.numpy())
            ).float()
        variance = projector.explained_variance_ratio_ * 100.0
        summary = ", ".join(f"PC{i + 1}={variance[i]:.4f}" for i in range(n_components))
        return projected_embeddings, projected_prototypes, summary

    if method == "tsne":
        fit_parts = [fit_bank_np]
        sample_counts = [len(embeddings_by_condition[label]) for label in embeddings_by_condition]
        proto_counts = {}
        for condition_label, prototypes in prototypes_by_condition_raw.items():
            if prototypes is None:
                proto_counts[condition_label] = 0
                continue
            if prototypes.shape[1] != fit_bank.shape[1]:
                raise ValueError(
                    f"Prototype dimensionality mismatch for {condition_label}: "
                    f"prototype dim={prototypes.shape[1]} but embedding dim={fit_bank.shape[1]}"
                )
            fit_parts.append(prototypes.numpy())
            proto_counts[condition_label] = len(prototypes)

        joint_bank = fit_parts[0] if len(fit_parts) == 1 else np.concatenate(fit_parts, axis=0)
        projector = TSNE(
            n_components=n_components,
            metric="cosine",
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            max_iter=1000,
        )
        embedded = torch.from_numpy(projector.fit_transform(joint_bank)).float()

        projected_embeddings = {}
        offset = 0
        condition_order = list(embeddings_by_condition.keys())
        for condition_label, count in zip(condition_order, sample_counts):
            projected_embeddings[condition_label] = embedded[offset:offset + count]
            offset += count

        projected_prototypes = {}
        for condition_label in condition_order:
            count = proto_counts.get(condition_label, 0)
            if count == 0:
                projected_prototypes[condition_label] = None
                continue
            projected_prototypes[condition_label] = embedded[offset:offset + count]
            offset += count

        summary = f"t-SNE metric=cosine perplexity={tsne_perplexity:.1f}"
        return projected_embeddings, projected_prototypes, summary

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "UMAP requested, but the 'umap-learn' package is not installed in this environment."
            ) from exc

        projector = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            random_state=seed,
        )
        projector.fit(fit_bank_np)
        projected_embeddings = {
            label: torch.from_numpy(projector.transform(embeddings.numpy())).float()
            for label, embeddings in embeddings_by_condition.items()
        }
        projected_prototypes = {}
        for condition_label, prototypes in prototypes_by_condition_raw.items():
            if prototypes is None:
                projected_prototypes[condition_label] = None
                continue
            if prototypes.shape[1] != fit_bank.shape[1]:
                raise ValueError(
                    f"Prototype dimensionality mismatch for {condition_label}: "
                    f"prototype dim={prototypes.shape[1]} but embedding dim={fit_bank.shape[1]}"
                )
            projected_prototypes[condition_label] = torch.from_numpy(
                projector.transform(prototypes.numpy())
            ).float()
        return projected_embeddings, projected_prototypes, "UMAP metric=cosine"

    raise ValueError(f"Unsupported method: {method}")


def _subsample_mask(labels: torch.Tensor, max_points: int | None, seed: int) -> torch.Tensor:
    if max_points is None or len(labels) <= max_points:
        return torch.ones(len(labels), dtype=torch.bool)

    gen = torch.Generator()
    gen.manual_seed(seed)
    chosen = []
    per_class = max(1, max_points // len(LABEL_NAMES))

    for label_id in range(len(LABEL_NAMES)):
        idx = torch.nonzero(labels == label_id, as_tuple=False).flatten()
        if len(idx) <= per_class:
            chosen.append(idx)
            continue
        perm = torch.randperm(len(idx), generator=gen)[:per_class]
        chosen.append(idx[perm])

    mask = torch.zeros(len(labels), dtype=torch.bool)
    merged = torch.cat(chosen).unique()
    if len(merged) < max_points:
        mask[merged] = True
        remainder = max_points - int(mask.sum())
        if remainder > 0:
            available = torch.nonzero(~mask, as_tuple=False).flatten()
            perm = torch.randperm(len(available), generator=gen)[:remainder]
            mask[available[perm]] = True
        return mask

    mask[merged[:max_points]] = True
    return mask


def _write_coordinates_csv(
    csv_path: str,
    conditions: list[str],
    projected_embeddings: dict[str, torch.Tensor],
    labels_by_condition: dict[str, torch.Tensor],
    centroids_by_condition: dict[str, dict[int, torch.Tensor]],
    prototypes_by_condition: dict[str, torch.Tensor | None],
    reference_condition: str | None,
    reference_centroids: dict[int, torch.Tensor] | None,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    dims = projected_embeddings[conditions[0]].shape[1]
    pc_headers = [f"pc{i + 1}" for i in range(dims)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "kind", "label", "sample_index", *pc_headers])

        for condition_label in conditions:
            labels = labels_by_condition[condition_label]
            coords = projected_embeddings[condition_label]
            for idx, (coord, label_id) in enumerate(zip(coords.tolist(), labels.tolist())):
                writer.writerow(
                    [condition_label, "sample", LABEL_NAMES[label_id], idx, *coord]
                )

            for label_id, centroid in centroids_by_condition[condition_label].items():
                writer.writerow(
                    [
                        condition_label,
                        "class_centroid",
                        LABEL_NAMES[label_id],
                        "",
                        *[float(v) for v in centroid.tolist()],
                    ]
                )

            prototypes = prototypes_by_condition.get(condition_label)
            if prototypes is not None:
                for label_id, proto in enumerate(prototypes.tolist()):
                    writer.writerow(
                        [
                            condition_label,
                            "prompt_prototype",
                            LABEL_NAMES[label_id],
                            "",
                            *proto,
                        ]
                    )

        if reference_condition and reference_centroids:
            for label_id, centroid in reference_centroids.items():
                writer.writerow(
                    [
                        reference_condition,
                        "reference_centroid",
                        LABEL_NAMES[label_id],
                        "",
                        *[float(v) for v in centroid.tolist()],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shared-basis projections for cached condition embeddings.")
    parser.add_argument("--cache_dir", default=_default_cache_dir(), help="Condition cache directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to visualize.")
    parser.add_argument(
        "--method",
        choices=["pca", "tsne", "umap"],
        default="pca",
        help="Projection method. PCA is linear; t-SNE and UMAP are nonlinear neighborhood views.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Condition labels or slugs to plot. Defaults to all available cached conditions.",
    )
    parser.add_argument("--output", help="Output PNG path.")
    parser.add_argument(
        "--max_points_per_condition",
        type=int,
        default=None,
        help="Optional per-condition subsample cap for cleaner scatter plots.",
    )
    parser.add_argument(
        "--no_prototypes",
        action="store_true",
        help="Hide class prompt prototype markers.",
    )
    parser.add_argument(
        "--no_class_centroids",
        action="store_true",
        help="Hide current-condition class centroid markers (X).",
    )
    parser.add_argument(
        "--no_reference_centroids",
        action="store_true",
        help="Hide reference-condition centroid markers (circles).",
    )
    parser.add_argument(
        "--reference_condition",
        default="b1_raw",
        help="Reference condition whose class centroids are overlaid to show movement.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of projected components to visualize.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE. Ignored for PCA/UMAP.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for subsampling.")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    condition_labels = _resolve_condition_labels(args.conditions, cache_dir, args.split)

    ref_key = args.reference_condition.strip().lower()
    reference_condition = LABEL_ALIASES.get(ref_key)
    if reference_condition not in condition_labels:
        reference_condition = condition_labels[0]

    payloads = {
        label: _load_condition_payload(cache_dir, label, args.split)
        for label in condition_labels
    }

    embeddings_by_condition = {}
    labels_by_condition = {}
    for idx, condition_label in enumerate(condition_labels):
        labels = payloads[condition_label]["labels"].long().cpu()
        embeddings = _normalize(payloads[condition_label]["embeddings"].cpu())
        mask = _subsample_mask(labels, args.max_points_per_condition, args.seed + idx)
        labels_by_condition[condition_label] = labels[mask]
        embeddings_by_condition[condition_label] = embeddings[mask]

    fit_bank = torch.cat([embeddings_by_condition[label] for label in condition_labels], dim=0)
    prototypes_raw = {}
    if not args.no_prototypes:
        for condition_label in condition_labels:
            prototypes_raw[condition_label] = _load_condition_prototypes(cache_dir, condition_label)
    else:
        prototypes_raw = {label: None for label in condition_labels}

    projected_embeddings, prototypes_by_condition, projection_summary = _compute_projection(
        method=args.method,
        n_components=args.n_components,
        fit_bank=fit_bank,
        embeddings_by_condition=embeddings_by_condition,
        prototypes_by_condition_raw=prototypes_raw,
        tsne_perplexity=args.tsne_perplexity,
        seed=args.seed,
    )

    centroids_by_condition = {
        label: _compute_centroids(projected_embeddings[label], labels_by_condition[label])
        for label in condition_labels
    }
    reference_centroids = None if args.no_reference_centroids else centroids_by_condition.get(reference_condition)

    n_conditions = len(condition_labels)
    ncols = min(3, n_conditions)
    nrows = math.ceil(n_conditions / ncols)
    subplot_kw = {"projection": "3d"} if args.n_components == 3 else {}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 5.4 * nrows),
        squeeze=False,
        subplot_kw=subplot_kw,
    )
    axes_flat = axes.flatten()

    for ax_idx, condition_label in enumerate(condition_labels):
        ax = axes_flat[ax_idx]
        coords = projected_embeddings[condition_label]
        labels = labels_by_condition[condition_label]

        for label_id, label_name in enumerate(LABEL_NAMES):
            mask = labels == label_id
            if int(mask.sum()) == 0:
                continue
            pts = coords[mask]
            if args.n_components == 3:
                ax.scatter(
                    pts[:, 0].numpy(),
                    pts[:, 1].numpy(),
                    pts[:, 2].numpy(),
                    s=10,
                    alpha=0.42,
                    c=LABEL_COLORS[label_id],
                    label=label_name,
                    linewidths=0,
                    depthshade=False,
                )
            else:
                ax.scatter(
                    pts[:, 0].numpy(),
                    pts[:, 1].numpy(),
                    s=12,
                    alpha=0.45,
                    c=LABEL_COLORS[label_id],
                    label=label_name,
                    linewidths=0,
                )

        current_centroids = centroids_by_condition[condition_label]
        if reference_centroids is not None:
            for label_id, ref_centroid in reference_centroids.items():
                if condition_label != reference_condition and label_id in current_centroids:
                    cur = current_centroids[label_id]
                    if args.n_components == 3:
                        ax.plot(
                            [float(ref_centroid[0]), float(cur[0])],
                            [float(ref_centroid[1]), float(cur[1])],
                            [float(ref_centroid[2]), float(cur[2])],
                            linestyle="--",
                            linewidth=1.1,
                            color=LABEL_COLORS[label_id],
                            alpha=0.8,
                        )
                    else:
                        ax.plot(
                            [float(ref_centroid[0]), float(cur[0])],
                            [float(ref_centroid[1]), float(cur[1])],
                            linestyle="--",
                            linewidth=1.2,
                            color=LABEL_COLORS[label_id],
                            alpha=0.8,
                        )
                if args.n_components == 3:
                    ax.scatter(
                        float(ref_centroid[0]),
                        float(ref_centroid[1]),
                        float(ref_centroid[2]),
                        s=70,
                        facecolors="white",
                        edgecolors=LABEL_COLORS[label_id],
                        linewidths=1.4,
                        marker="o",
                        depthshade=False,
                    )
                else:
                    ax.scatter(
                        float(ref_centroid[0]),
                        float(ref_centroid[1]),
                        s=90,
                        facecolors="white",
                        edgecolors=LABEL_COLORS[label_id],
                        linewidths=1.6,
                        marker="o",
                        zorder=5,
                    )

        if not args.no_class_centroids:
            for label_id, centroid in current_centroids.items():
                if args.n_components == 3:
                    ax.scatter(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                        s=90,
                        c=LABEL_COLORS[label_id],
                        marker="X",
                        edgecolors="black",
                        linewidths=0.5,
                        depthshade=False,
                    )
                else:
                    ax.scatter(
                        float(centroid[0]),
                        float(centroid[1]),
                        s=110,
                        c=LABEL_COLORS[label_id],
                        marker="X",
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=6,
                    )

        prototypes = prototypes_by_condition.get(condition_label)
        if prototypes is not None:
            for label_id, proto in enumerate(prototypes):
                if args.n_components == 3:
                    ax.scatter(
                        float(proto[0]),
                        float(proto[1]),
                        float(proto[2]),
                        s=125,
                        c=LABEL_COLORS[label_id],
                        marker="*",
                        edgecolors="black",
                        linewidths=0.6,
                        depthshade=False,
                    )
                else:
                    ax.scatter(
                        float(proto[0]),
                        float(proto[1]),
                        s=150,
                        c=LABEL_COLORS[label_id],
                        marker="*",
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=7,
                    )

        ax.set_title(condition_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if args.n_components == 3:
            ax.set_zlabel("PC3")
            ax.view_init(elev=24, azim=-58)
            ax.grid(True, alpha=0.12)
        else:
            ax.grid(alpha=0.15)

    for ax in axes_flat[n_conditions:]:
        ax.axis("off")

    class_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=LABEL_COLORS[idx], markeredgecolor="none", markersize=7, label=name)
        for idx, name in enumerate(LABEL_NAMES)
    ]
    marker_handles = []
    if reference_centroids is not None:
        marker_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=8, label=f"{reference_condition} centroid")
        )
    if not args.no_class_centroids:
        marker_handles.append(
            Line2D([0], [0], marker="X", linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Current class centroid")
        )
    if not args.no_prototypes:
        marker_handles.append(
            Line2D([0], [0], marker="*", linestyle="", markerfacecolor="black", markeredgecolor="black", markersize=10, label="Prompt prototype")
        )

    title_prefix = {
        "pca": "Shared-basis PCA",
        "tsne": "Shared-fit t-SNE",
        "umap": "Shared-fit UMAP",
    }[args.method]
    if args.method == "pca":
        variance = PCA(n_components=args.n_components).fit(fit_bank.numpy()).explained_variance_ratio_ * 100.0
        variance_text = " | ".join(
            f"PC{i + 1}={variance[i]:.1f}%"
            for i in range(args.n_components)
        )
        title_suffix = variance_text
    else:
        title_suffix = projection_summary or args.method.upper()
    fig.suptitle(
        (
            f"{title_prefix} of normalized {args.split} embeddings\n"
            f"{title_suffix} | cache={os.path.basename(cache_dir)}"
        ),
        fontsize=14,
    )
    fig.legend(
        handles=class_handles + marker_handles,
        loc="lower center",
        ncol=min(6, len(class_handles) + len(marker_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    output_path = _build_output_path_for_method(
        cache_dir,
        args.split,
        args.method,
        args.n_components,
        args.output,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    _write_coordinates_csv(
        csv_path=csv_path,
        conditions=condition_labels,
        projected_embeddings=projected_embeddings,
        labels_by_condition=labels_by_condition,
        centroids_by_condition=centroids_by_condition,
        prototypes_by_condition=prototypes_by_condition,
        reference_condition=reference_condition,
        reference_centroids=reference_centroids,
    )

    print(f"Saved projection plot -> {output_path}")
    print(f"Saved projection coordinates -> {csv_path}")
    if args.method == "pca":
        print("Explained variance: " + ", ".join(
            f"PC{i + 1}={variance[i]:.4f}"
            for i in range(args.n_components)
        ))
    elif projection_summary:
        print(projection_summary)


if __name__ == "__main__":
    main()
