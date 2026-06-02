import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn


CLASS_NAMES = ["Sell", "Hold", "Buy"]
GROUP_ORDER = ["Blue chip stocks", "PennyStocks", "Crypto"]
DEFAULT_SCHEMES = [
    "timeCV_2020",
    "progressiveHorizonCV_val2020_2022_test2023_2025",
    "outerInnerProgressive_train2012_2019_val2020_2022_test2023_2025",
    "leaveOneBlockOutCV_val2020_2022_test2023_2025",
    "loocv_undersampled_ratio2.0",
]
SUPPORTED_BEST_SCHEMES = set(DEFAULT_SCHEMES)
DISPLAY_LABEL_MAP = {
    "CrudeOil/Average/PHP/bbl": "Oil",
    "Gold_PHP/troy/oz": "Gold",
}
SCHEME_LABEL_MAP = {
    "timeCV_2020": "CHV-SKF",
    "progressiveHorizonCV_val2020_2022_test2023_2025": "EW-WFV",
    "outerInnerProgressive_train2012_2019_val2020_2022_test2023_2025": "NWFV",
    "leaveOneBlockOutCV_val2020_2022_test2023_2025": "BCV",
    "loocv_undersampled_ratio2.0": "BCV-PU",
}
SIGNED_TO_CLASS = {-1: 0, 0: 1, 1: 2}


class SmallCNN_BN(nn.Module):
    def __init__(self, h: int, w: int, n_classes: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            feat = self.fe(torch.zeros(1, 1, h, w)).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(feat, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fe(x))


@dataclass
class AssetInfo:
    group: str
    symbol: str
    asset_root: Path
    dataset_dir: Path
    model_root: Path


@dataclass
class SchemeSelection:
    scheme_name: str
    checkpoint_path: Path
    test_metrics_path: Path | None


@dataclass
class ExplainedSample:
    sample_index: int
    date: pd.Timestamp
    true_class_idx: int
    pred_class_idx: int
    pred_probs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for CNN-MIC checkpoints across asset groups."
    )
    parser.add_argument(
        "--stocks",
        nargs="*",
        default=None,
        help="Optional stock/asset symbols to include, e.g. AC JFC BTC.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional asset groups to include: 'Blue chip stocks', 'PennyStocks', 'Crypto'.",
    )
    parser.add_argument(
        "--scheme",
        default="best",
        help=(
            "Validation scheme to analyze. Use 'best' to pick the highest test-accuracy "
            "scheme among final checkpoints available for each asset."
        ),
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Number of training samples to use as SHAP background.",
    )
    parser.add_argument(
        "--samples-per-asset",
        type=int,
        default=20,
        help="Number of test samples to explain per asset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for background and explanation sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("shap_outputs"),
        help="Directory where plots and CSV summaries will be written.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device. Use 'cuda' if available and desired.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Restrict explained samples to correctly predicted test cases only.",
    )
    parser.add_argument(
        "--fixed-scheme-control",
        action="store_true",
        help=(
            "When used with --scheme best and multiple assets, choose one common best scheme "
            "across all selected assets instead of allowing a different best scheme per asset."
        ),
    )
    return parser.parse_args()


def normalize_group_name(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "bluechip": "Blue chip stocks",
        "bluechips": "Blue chip stocks",
        "blue chip": "Blue chip stocks",
        "blue chip stocks": "Blue chip stocks",
        "penny": "PennyStocks",
        "pennystocks": "PennyStocks",
        "penny stocks": "PennyStocks",
        "crypto": "Crypto",
        "cryptocurrency": "Crypto",
        "cryptocurrencies": "Crypto",
    }
    return aliases.get(lowered, value)


def discover_assets(root: Path) -> list[AssetInfo]:
    manifests = sorted(root.glob("Thesis/Stocks/*/*/dataset_out/tech_macro_v2/manifest.json"))
    assets: list[AssetInfo] = []
    for manifest_path in manifests:
        dataset_dir = manifest_path.parent
        asset_root = dataset_dir.parent.parent
        group = asset_root.parent.name
        symbol = asset_root.name
        model_root = asset_root / symbol / "tech_macro"
        if model_root.exists():
            assets.append(
                AssetInfo(
                    group=group,
                    symbol=symbol,
                    asset_root=asset_root,
                    dataset_dir=dataset_dir,
                    model_root=model_root,
                )
            )
    return assets


def filter_assets(
    assets: Iterable[AssetInfo],
    groups: list[str] | None,
    stocks: list[str] | None,
) -> list[AssetInfo]:
    group_filter = None
    if groups:
        group_filter = {normalize_group_name(group) for group in groups}
    stock_filter = None
    if stocks:
        stock_filter = {stock.upper() for stock in stocks}

    filtered = []
    for asset in assets:
        if group_filter and asset.group not in group_filter:
            continue
        if stock_filter and asset.symbol.upper() not in stock_filter:
            continue
        filtered.append(asset)
    return filtered


def choose_primary_metrics_file(scheme_dir: Path) -> Path | None:
    metric_files = [
        path for path in scheme_dir.glob("test_metrics*.json")
        if "evalscript" not in path.name and ".ipynb_checkpoints" not in str(path)
    ]
    if not metric_files:
        return None

    ranked = []
    for path in metric_files:
        score = (-len(path.stem), path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = int(payload.get("rows", 0))
            score = (-rows, path.name)
        except Exception:
            pass
        ranked.append((score, path))

    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def find_checkpoint_path(scheme_dir: Path) -> Path | None:
    direct = scheme_dir / "final_fulltrain.pt"
    nested = scheme_dir / "final_fit" / "final_fulltrain.pt"
    final_model = scheme_dir / "final_model.pt"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    if final_model.exists():
        return final_model
    return None


def available_schemes(asset: AssetInfo) -> list[SchemeSelection]:
    selections: list[SchemeSelection] = []
    for scheme_dir in sorted(asset.model_root.iterdir()):
        if not scheme_dir.is_dir():
            continue
        checkpoint = find_checkpoint_path(scheme_dir)
        if checkpoint is None:
            continue
        selections.append(
            SchemeSelection(
                scheme_name=scheme_dir.name,
                checkpoint_path=checkpoint,
                test_metrics_path=choose_primary_metrics_file(scheme_dir),
            )
        )
    return selections


def read_accuracy(metrics_path: Path | None) -> float:
    if metrics_path is None:
        return float("-inf")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload.get("acc", float("-inf")))
    except Exception:
        return float("-inf")


def mean_accuracy_for_scheme(assets: list[AssetInfo], scheme_name: str) -> float:
    scores = []
    for asset in assets:
        for selection in available_schemes(asset):
            if selection.scheme_name == scheme_name:
                score = read_accuracy(selection.test_metrics_path)
                if np.isfinite(score):
                    scores.append(score)
                break
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def choose_fixed_best_scheme(assets: list[AssetInfo]) -> str:
    ranked = sorted(
        (
            (scheme_name, mean_accuracy_for_scheme(assets, scheme_name))
            for scheme_name in SUPPORTED_BEST_SCHEMES
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked or not np.isfinite(ranked[0][1]):
        raise ValueError("Could not determine a shared best scheme across the selected assets.")
    return ranked[0][0]


def select_scheme(asset: AssetInfo, requested_scheme: str) -> SchemeSelection:
    schemes = available_schemes(asset)
    if not schemes:
        raise FileNotFoundError(f"No final CNN-MIC checkpoints found for {asset.symbol}.")

    if requested_scheme != "best":
        for selection in schemes:
            if selection.scheme_name == requested_scheme:
                return selection
        available = ", ".join(selection.scheme_name for selection in schemes)
        raise ValueError(f"{asset.symbol}: scheme '{requested_scheme}' not available. Found: {available}")

    candidates = [selection for selection in schemes if selection.scheme_name in SUPPORTED_BEST_SCHEMES]
    if not candidates:
        candidates = schemes
    candidates.sort(key=lambda item: read_accuracy(item.test_metrics_path), reverse=True)
    return candidates[0]


def parse_test_start_year(scheme_name: str, metrics_path: Path | None = None) -> int:
    if scheme_name.startswith("timeCV_"):
        return int(scheme_name.rsplit("_", 1)[1]) + 1
    match = re.search(r"_test(\d{4})_(\d{4})", scheme_name)
    if match:
        return int(match.group(1))
    if metrics_path is not None:
        metrics_match = re.search(r"test_metrics_(\d{4})(?:_\d{4}|_onward)?", metrics_path.stem)
        if metrics_match:
            return int(metrics_match.group(1))
    raise ValueError(f"Could not infer test start year from scheme name: {scheme_name}")


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.Series, dict]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    index_df = pd.read_csv(dataset_dir / "index.csv", parse_dates=["Date"])
    images = np.load(dataset_dir / "images.npy").astype(np.float32) / 255.0
    labels = np.load(dataset_dir / "labels.npy").astype(np.int64)
    return images, labels, index_df["Date"], manifest


def build_model(checkpoint_path: Path, height: int, width: int, device: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = SmallCNN_BN(
        h=int(checkpoint.get("h", height)) if isinstance(checkpoint, dict) else height,
        w=int(checkpoint.get("w", width)) if isinstance(checkpoint, dict) else width,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def map_signed_labels_to_class_indices(labels: np.ndarray) -> np.ndarray:
    unique_labels = set(np.unique(labels).tolist())
    expected_labels = set(SIGNED_TO_CLASS.keys())
    unexpected_labels = unique_labels - expected_labels
    if unexpected_labels:
        raise ValueError(
            f"Unexpected label values found: {sorted(unexpected_labels)}. "
            f"Expected only {sorted(expected_labels)}."
        )
    return np.vectorize(SIGNED_TO_CLASS.get)(labels).astype(np.int64)


def choose_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        raise ValueError("No samples available for the requested split.")
    if len(candidates) <= limit:
        return candidates
    return np.sort(rng.choice(candidates, size=limit, replace=False))


@torch.no_grad()
def predict_samples(model: nn.Module, images: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    tensor = torch.from_numpy(images).unsqueeze(1).to(device)
    probs = []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size]
        probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.vstack(probs)


def build_explainer(model: nn.Module, background: torch.Tensor):
    try:
        return shap.DeepExplainer(model, background), "DeepExplainer"
    except Exception:
        return shap.GradientExplainer(model, background), "GradientExplainer"


def compute_shap_values(
    model: nn.Module,
    images: np.ndarray,
    background_idx: np.ndarray,
    explain_idx: np.ndarray,
    device: str,
):
    background = torch.from_numpy(images[background_idx]).unsqueeze(1).to(device)
    explain = torch.from_numpy(images[explain_idx]).unsqueeze(1).to(device)
    explainer, explainer_name = build_explainer(model, background)
    shap_values = explainer.shap_values(explain)
    return shap_values, explain.cpu().numpy(), explainer_name


def get_class_shap(shap_values, class_idx: int) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[class_idx])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 5:
            values = values[..., class_idx]
        else:
            raise ValueError(f"Unexpected SHAP array shape: {values.shape}")

    if values.ndim == 3:
        values = values[:, np.newaxis, :, :]
    if values.ndim != 4:
        raise ValueError(f"Unexpected per-class SHAP shape: {values.shape}")
    return values


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def format_row_label(value: str) -> str:
    return DISPLAY_LABEL_MAP.get(value, value)


def format_scheme_label(value: str) -> str:
    return SCHEME_LABEL_MAP.get(value, value)


def save_heatmap(
    heatmap: np.ndarray,
    row_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Lookback Window Column")
    ax.set_ylabel("Indicator Row")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_row_bar(
    row_importance: np.ndarray,
    tech_count: int,
    row_labels: list[str],
    title: str,
    output_path: Path,
    sort_desc: bool = False,
) -> None:
    labels = list(row_labels)
    values = np.asarray(row_importance)
    colors = ["steelblue" if idx < tech_count else "indianred" for idx in range(len(row_labels))]

    if sort_desc:
        order = np.argsort(values)[::-1]
        values = values[order]
        labels = [labels[idx] for idx in order]
        colors = [colors[idx] for idx in order]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y, values, color=colors, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| per Row")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_tech_macro_bar(
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    x = np.arange(len(summary_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        summary_df["technical_total"],
        width,
        label="Technical",
        color="steelblue",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        summary_df["macro_total"],
        width,
        label="Macro",
        color="indianred",
        edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["class_name"])
    ax.set_ylabel("Total Mean |SHAP|")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def samples_to_frame(samples: list[ExplainedSample], asset_name: str, scheme_label: str, scheme_name: str) -> pd.DataFrame:
    rows = []
    for sample in samples:
        row = {
            "asset": asset_name,
            "scheme": scheme_label,
            "scheme_id": scheme_name,
            "sample_index": sample.sample_index,
            "date": sample.date,
            "true_class_idx": sample.true_class_idx,
            "true_class_name": CLASS_NAMES[sample.true_class_idx],
            "pred_class_idx": sample.pred_class_idx,
            "pred_class_name": CLASS_NAMES[sample.pred_class_idx],
            "correct": bool(sample.true_class_idx == sample.pred_class_idx),
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"p_{class_name.lower()}"] = float(sample.pred_probs[class_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_summary(
    shap_values,
    manifest: dict,
    asset_name: str,
    scheme_name: str,
    scheme_label: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech_rows = list(manifest["tech_rows"])
    macro_rows = list(manifest["macro_rows"])
    raw_row_labels = tech_rows + macro_rows
    row_labels = [format_row_label(label) for label in raw_row_labels]
    tech_count = len(tech_rows)

    class_rows = []
    contribution_rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_shap = get_class_shap(shap_values, class_idx)
        mean_abs_heatmap = np.abs(class_shap).mean(axis=(0, 1))
        row_importance = mean_abs_heatmap.mean(axis=1)
        technical_total = float(row_importance[:tech_count].sum())
        macro_total = float(row_importance[tech_count:].sum())

        for row_idx, row_name in enumerate(row_labels):
            class_rows.append(
                {
                    "asset": asset_name,
                    "scheme": scheme_label,
                    "scheme_id": scheme_name,
                    "class_name": class_name,
                    "row_index": row_idx,
                    "row_name": row_name,
                    "row_type": "technical" if row_idx < tech_count else "macro",
                    "mean_abs_shap": float(row_importance[row_idx]),
                }
            )

        contribution_rows.append(
                {
                    "asset": asset_name,
                    "scheme": scheme_label,
                    "scheme_id": scheme_name,
                    "class_name": class_name,
                    "technical_total": technical_total,
                    "macro_total": macro_total,
                "macro_share": (
                    macro_total / (technical_total + macro_total)
                    if (technical_total + macro_total) > 0
                    else np.nan
                ),
            }
        )

        prefix = sanitize_name(f"{asset_name}_{scheme_name}_{class_name}")
        save_heatmap(
            mean_abs_heatmap,
            row_labels,
            f"{asset_name} | {scheme_label} | {class_name} mean |SHAP| heatmap",
            output_dir / f"{prefix}_heatmap.png",
        )
        save_row_bar(
            row_importance,
            tech_count,
            row_labels,
            f"{asset_name} | {scheme_label} | {class_name} row importance",
            output_dir / f"{prefix}_row_importance.png",
            sort_desc=True,
        )

    class_df = pd.DataFrame(class_rows)
    contribution_df = pd.DataFrame(contribution_rows)
    save_tech_macro_bar(
        contribution_df,
        f"{asset_name} | {scheme_label} technical vs macro contribution",
        output_dir / f"{sanitize_name(f'{asset_name}_{scheme_name}')}_tech_vs_macro.png",
    )
    return class_df, contribution_df


def predicted_class_summary(
    shap_values,
    samples: list[ExplainedSample],
    manifest: dict,
    asset_name: str,
    scheme_name: str,
    scheme_label: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tech_rows = list(manifest["tech_rows"])
    macro_rows = list(manifest["macro_rows"])
    row_labels = [format_row_label(label) for label in (tech_rows + macro_rows)]
    tech_count = len(tech_rows)

    predicted_heatmaps = []
    sample_rows = []
    for sample_pos, sample in enumerate(samples):
        class_shap = get_class_shap(shap_values, sample.pred_class_idx)
        sample_heatmap = np.abs(class_shap[sample_pos]).mean(axis=0)
        predicted_heatmaps.append(sample_heatmap)
        row_importance = sample_heatmap.mean(axis=1)
        for row_idx, row_name in enumerate(row_labels):
            sample_rows.append(
                {
                    "asset": asset_name,
                    "scheme": scheme_label,
                    "scheme_id": scheme_name,
                    "sample_index": sample.sample_index,
                    "date": sample.date,
                    "pred_class_name": CLASS_NAMES[sample.pred_class_idx],
                    "true_class_name": CLASS_NAMES[sample.true_class_idx],
                    "correct": bool(sample.true_class_idx == sample.pred_class_idx),
                    "row_index": row_idx,
                    "row_name": row_name,
                    "row_type": "technical" if row_idx < tech_count else "macro",
                    "mean_abs_shap": float(row_importance[row_idx]),
                }
            )

    predicted_stack = np.stack(predicted_heatmaps, axis=0)
    mean_abs_heatmap = predicted_stack.mean(axis=0)
    row_importance = mean_abs_heatmap.mean(axis=1)
    technical_total = float(row_importance[:tech_count].sum())
    macro_total = float(row_importance[tech_count:].sum())

    prefix = sanitize_name(f"{asset_name}_{scheme_name}_predicted_class")
    save_heatmap(
        mean_abs_heatmap,
        row_labels,
        f"{asset_name} | {scheme_label} | Predicted-class mean |SHAP| heatmap",
        output_dir / f"{prefix}_heatmap.png",
    )
    save_row_bar(
        row_importance,
        tech_count,
        row_labels,
        f"{asset_name} | {scheme_label} | Predicted-class row importance",
        output_dir / f"{prefix}_row_importance.png",
        sort_desc=True,
    )

    contribution_df = pd.DataFrame(
        [
            {
                "asset": asset_name,
                "scheme": scheme_label,
                "scheme_id": scheme_name,
                "view": "predicted_class",
                "technical_total": technical_total,
                "macro_total": macro_total,
                "macro_share": (
                    macro_total / (technical_total + macro_total)
                    if (technical_total + macro_total) > 0
                    else np.nan
                ),
            }
        ]
    )
    save_tech_macro_bar(
        pd.DataFrame(
            [{
                "class_name": "Predicted",
                "technical_total": technical_total,
                "macro_total": macro_total,
            }]
        ),
        f"{asset_name} | {scheme_label} | Predicted-class technical vs macro",
        output_dir / f"{prefix}_tech_vs_macro.png",
    )
    sample_rows_df = pd.DataFrame(sample_rows).sort_values(
        by=["sample_index", "mean_abs_shap"],
        ascending=[True, False],
    )
    return sample_rows_df, contribution_df


def aggregate_group_outputs(
    group_name: str,
    scheme_name: str,
    class_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if class_df.empty:
        return

    grouped = (
        class_df.groupby(["class_name", "row_index", "row_name", "row_type"], as_index=False)["mean_abs_shap"]
        .mean()
    )
    contribution = (
        grouped.groupby(["class_name", "row_type"], as_index=False)["mean_abs_shap"]
        .sum()
        .pivot(index="class_name", columns="row_type", values="mean_abs_shap")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    contribution = contribution.rename(
        columns={
            "technical": "technical_total",
            "macro": "macro_total",
        }
    )
    for column in ["technical_total", "macro_total"]:
        if column not in contribution:
            contribution[column] = 0.0

    row_labels = grouped.sort_values("row_index")["row_name"].tolist()
    heatmaps = []
    for class_name in CLASS_NAMES:
        class_slice = grouped[grouped["class_name"] == class_name].sort_values("row_index")
        heatmaps.append(class_slice["mean_abs_shap"].to_numpy())
    heatmap_matrix = np.vstack(heatmaps)

    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(heatmap_matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title(f"{group_name} | {scheme_name} grouped row importance")
    fig.colorbar(image, ax=ax, label="Mean row |SHAP|")
    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}')}_group_heatmap.png", dpi=200)
    plt.close(fig)

    save_tech_macro_bar(
        contribution[["class_name", "technical_total", "macro_total"]],
        f"{group_name} | {scheme_name} technical vs macro contribution",
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}')}_tech_vs_macro.png",
    )

    grouped.to_csv(
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}')}_row_importance.csv",
        index=False,
    )
    contribution.to_csv(
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}')}_tech_macro_summary.csv",
        index=False,
    )


def aggregate_predicted_group_outputs(
    group_name: str,
    scheme_name: str,
    predicted_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if predicted_df.empty:
        return

    grouped = (
        predicted_df.groupby(["row_index", "row_name", "row_type"], as_index=False)["mean_abs_shap"]
        .mean()
    )
    contribution = (
        grouped.groupby(["row_type"], as_index=False)["mean_abs_shap"]
        .sum()
        .pivot_table(columns="row_type", values="mean_abs_shap", aggfunc="first")
        .reset_index(drop=True)
        .rename_axis(None, axis=1)
    )
    for source, target in (("technical", "technical_total"), ("macro", "macro_total")):
        contribution[target] = contribution[source] if source in contribution else 0.0
    contribution = contribution[["technical_total", "macro_total"]]
    contribution.insert(0, "view", "predicted_class")
    contribution["macro_share"] = contribution.apply(
        lambda row: (
            row["macro_total"] / (row["technical_total"] + row["macro_total"])
            if (row["technical_total"] + row["macro_total"]) > 0
            else np.nan
        ),
        axis=1,
    )

    row_labels = grouped.sort_values("row_index")["row_name"].tolist()
    row_values = grouped.sort_values("row_index")["mean_abs_shap"].to_numpy()[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(12, 2.8))
    image = ax.imshow(row_values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels(["Predicted"])
    ax.set_title(f"{group_name} | {scheme_name} predicted-class row importance")
    fig.colorbar(image, ax=ax, label="Mean row |SHAP|")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}_predicted_class')}_group_heatmap.png",
        dpi=200,
    )
    plt.close(fig)

    save_tech_macro_bar(
        pd.DataFrame(
            [{
                "class_name": "Predicted",
                "technical_total": float(contribution["technical_total"].iloc[0]),
                "macro_total": float(contribution["macro_total"].iloc[0]),
            }]
        ),
        f"{group_name} | {scheme_name} predicted-class technical vs macro contribution",
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}_predicted_class')}_tech_vs_macro.png",
    )

    grouped.sort_values(
        by="mean_abs_shap",
        ascending=False,
    ).to_csv(
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}_predicted_class')}_row_importance.csv",
        index=False,
    )
    contribution.to_csv(
        output_dir / f"{sanitize_name(f'{group_name}_{scheme_name}_predicted_class')}_tech_macro_summary.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    assets = filter_assets(
        discover_assets(root),
        groups=args.groups,
        stocks=args.stocks,
    )
    if not assets:
        raise SystemExit("No matching CNN-MIC assets were found.")

    requested_scheme = args.scheme
    if args.fixed_scheme_control and args.scheme == "best" and len(assets) > 1:
        requested_scheme = choose_fixed_best_scheme(assets)

    all_asset_rows = []
    all_asset_contrib = []
    all_predicted_rows = []
    all_predicted_contrib = []
    run_manifest_rows = []

    for asset in assets:
        scheme = select_scheme(asset, requested_scheme)
        scheme_label = format_scheme_label(scheme.scheme_name)
        images, labels, dates, manifest = load_dataset(asset.dataset_dir)
        test_start_year = parse_test_start_year(scheme.scheme_name, scheme.test_metrics_path)
        train_mask = dates.dt.year.to_numpy() < test_start_year
        test_mask = dates.dt.year.to_numpy() >= test_start_year

        background_idx = choose_indices(train_mask, args.background_size, rng)
        model = build_model(
            checkpoint_path=scheme.checkpoint_path,
            height=int(manifest["height_rows"]),
            width=int(manifest["width_cols"]),
            device=args.device,
        )
        test_candidates = np.flatnonzero(test_mask)
        test_probs = predict_samples(model, images[test_candidates], args.device)
        true_classes = map_signed_labels_to_class_indices(labels[test_candidates])
        pred_classes = test_probs.argmax(axis=1)
        if args.correct_only:
            candidate_mask = pred_classes == true_classes
        else:
            candidate_mask = np.ones(len(test_candidates), dtype=bool)
        eligible_test_count = int(candidate_mask.sum())
        explain_idx = choose_indices(
            np.isin(np.arange(len(images)), test_candidates[candidate_mask]),
            args.samples_per_asset,
            rng,
        )

        test_lookup = {global_idx: pos for pos, global_idx in enumerate(test_candidates)}
        explained_samples = [
            ExplainedSample(
                sample_index=int(global_idx),
                date=dates.iloc[global_idx],
                true_class_idx=int(true_classes[test_lookup[global_idx]]),
                pred_class_idx=int(pred_classes[test_lookup[global_idx]]),
                pred_probs=test_probs[test_lookup[global_idx]],
            )
            for global_idx in explain_idx
        ]

        shap_values, _, explainer_name = compute_shap_values(
            model=model,
            images=images,
            background_idx=background_idx,
            explain_idx=explain_idx,
            device=args.device,
        )

        asset_name = f"{asset.group}:{asset.symbol}"
        asset_dir = args.out_dir / sanitize_name(asset_name) / sanitize_name(scheme.scheme_name)
        asset_dir.mkdir(parents=True, exist_ok=True)

        class_df, contribution_df = per_class_summary(
            shap_values=shap_values,
            manifest=manifest,
            asset_name=asset_name,
            scheme_name=scheme.scheme_name,
            scheme_label=scheme_label,
            output_dir=asset_dir,
        )
        predicted_rows_df, predicted_contrib_df = predicted_class_summary(
            shap_values=shap_values,
            samples=explained_samples,
            manifest=manifest,
            asset_name=asset_name,
            scheme_name=scheme.scheme_name,
            scheme_label=scheme_label,
            output_dir=asset_dir,
        )
        sample_predictions_df = samples_to_frame(
            explained_samples,
            asset_name=asset_name,
            scheme_label=scheme_label,
            scheme_name=scheme.scheme_name,
        )
        class_df["group"] = asset.group
        class_df["symbol"] = asset.symbol
        contribution_df["group"] = asset.group
        contribution_df["symbol"] = asset.symbol
        predicted_rows_df["group"] = asset.group
        predicted_rows_df["symbol"] = asset.symbol
        predicted_contrib_df["group"] = asset.group
        predicted_contrib_df["symbol"] = asset.symbol
        sample_predictions_df["group"] = asset.group
        sample_predictions_df["symbol"] = asset.symbol
        class_df.to_csv(asset_dir / "row_importance.csv", index=False)
        contribution_df.to_csv(asset_dir / "tech_macro_summary.csv", index=False)
        predicted_rows_df.to_csv(asset_dir / "predicted_class_row_importance.csv", index=False)
        predicted_contrib_df.to_csv(asset_dir / "predicted_class_tech_macro_summary.csv", index=False)
        sample_predictions_df.to_csv(asset_dir / "explained_sample_predictions.csv", index=False)

        all_asset_rows.append(class_df)
        all_asset_contrib.append(contribution_df)
        all_predicted_rows.append(predicted_rows_df)
        all_predicted_contrib.append(predicted_contrib_df)
        run_manifest_rows.append(
            {
                "group": asset.group,
                "symbol": asset.symbol,
                "scheme": scheme_label,
                "scheme_id": scheme.scheme_name,
                "requested_scheme": requested_scheme,
                "checkpoint_path": str(scheme.checkpoint_path),
                "test_metrics_path": str(scheme.test_metrics_path) if scheme.test_metrics_path else "",
                "background_size": len(background_idx),
                "eligible_test_samples": eligible_test_count,
                "explained_samples": len(explain_idx),
                "test_start_year": test_start_year,
                "correct_only": args.correct_only,
                "explainer": explainer_name,
            }
        )
        print(
            f"Processed {asset.group} {asset.symbol} using {scheme.scheme_name} "
            f"(eligible test samples: {eligible_test_count}, explained: {len(explain_idx)})"
        )

    run_manifest_df = pd.DataFrame(run_manifest_rows)
    run_manifest_df.to_csv(args.out_dir / "run_manifest.csv", index=False)

    asset_rows_df = pd.concat(all_asset_rows, ignore_index=True)
    asset_contrib_df = pd.concat(all_asset_contrib, ignore_index=True)
    predicted_rows_df = pd.concat(all_predicted_rows, ignore_index=True)
    predicted_contrib_df = pd.concat(all_predicted_contrib, ignore_index=True)
    asset_rows_df.to_csv(args.out_dir / "all_assets_row_importance.csv", index=False)
    asset_contrib_df.to_csv(args.out_dir / "all_assets_tech_macro_summary.csv", index=False)
    predicted_rows_df.to_csv(args.out_dir / "all_assets_predicted_class_row_importance.csv", index=False)
    predicted_contrib_df.to_csv(args.out_dir / "all_assets_predicted_class_tech_macro_summary.csv", index=False)

    for group_name in GROUP_ORDER:
        group_slice = asset_rows_df[asset_rows_df["group"] == group_name]
        if group_slice.empty:
            continue
        scheme_name = (
            "mixed_schemes"
            if group_slice["scheme"].nunique() > 1
            else str(group_slice["scheme"].iloc[0])
        )
        aggregate_group_outputs(
            group_name=group_name,
            scheme_name=scheme_name,
            class_df=group_slice,
            output_dir=args.out_dir,
        )
        predicted_group_slice = predicted_rows_df[predicted_rows_df["group"] == group_name]
        if not predicted_group_slice.empty:
            predicted_scheme_name = (
                "mixed_schemes"
                if predicted_group_slice["scheme"].nunique() > 1
                else str(predicted_group_slice["scheme"].iloc[0])
            )
            aggregate_predicted_group_outputs(
                group_name=group_name,
                scheme_name=predicted_scheme_name,
                predicted_df=predicted_group_slice,
                output_dir=args.out_dir,
            )


if __name__ == "__main__":
    main()
