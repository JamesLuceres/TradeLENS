import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

METHOD_SPECS = [
    {
        "label": "Chronological Holdout\nwith Stratified K-Fold",
        "relative_path": Path("timeCV_2020") / "cv_summary.csv",
        "metric_column": "val_acc",
    },
    {
        "label": "Expanding-Window\nWalk-Forward",
        "relative_path": (
            Path("progressiveHorizonCV_val2020_2022_test2023_2025")
            / "cv_summary_all_horizons.csv"
        ),
        "metric_column": "val_acc",
    },
    {
        "label": "Nested Walk-Forward Validation",
        "relative_path": (
            Path("outerInnerProgressive_train2012_2019_val2020_2022_test2023_2025")
            / "cv_summary_outer_inner.csv"
        ),
        "metric_column": "val_acc",
    },
    {
        "label": "Blocked Cross-Validation",
        "relative_path": (
            Path("leaveOneBlockOutCV_val2020_2022_test2023_2025")
            / "cv_summary.csv"
        ),
        "metric_column": "val_acc",
    },
    {
        "label": "Blocked CV with\nPartial Undersampling",
        "relative_path": Path("loocv_undersampled_ratio2.0") / "cv_summary.csv",
        "metric_column": "val_acc",
    },
]


def load_metric_values(base_dir: Path, relative_path: Path, metric_column: str) -> np.ndarray:
    csv_path = base_dir / relative_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        raise KeyError(f"Column {metric_column!r} not found in {csv_path}")

    values = pd.to_numeric(df[metric_column], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        raise ValueError(f"No valid values found in {csv_path}")

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation accuracy boxplots for any stock using tech-only and tech-macro result folders."
    )
    parser.add_argument(
        "--stock",
        required=True,
        help="Stock label for the chart title, for example AC or JFC.",
    )
    parser.add_argument(
        "--tech-only",
        required=True,
        dest="tech_only",
        help="Path to the stock's tech_only folder.",
    )
    parser.add_argument(
        "--tech-macro",
        required=True,
        dest="tech_macro",
        help="Path to the stock's tech_macro folder.",
    )
    return parser.parse_args()


def build_plot_dataframe(model_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []

    for model_name, model_dir in model_dirs.items():
        for spec in METHOD_SPECS:
            values = load_metric_values(
                base_dir=model_dir,
                relative_path=spec["relative_path"],
                metric_column=spec["metric_column"],
            )
            for value in values:
                rows.append(
                    {
                        "model": model_name,
                        "method": spec["label"],
                        "accuracy": value,
                    }
                )

    return pd.DataFrame(rows)


def style_boxplot(boxplot_dict: dict, facecolor: str) -> None:
    for box in boxplot_dict["boxes"]:
        box.set(facecolor=facecolor, edgecolor="black", linewidth=1.25)

    for median in boxplot_dict["medians"]:
        median.set(color="black", linewidth=1.6)

    for whisker in boxplot_dict["whiskers"]:
        whisker.set(color="black", linewidth=1.1)

    for cap in boxplot_dict["caps"]:
        cap.set(color="black", linewidth=1.1)


def main() -> None:
    args = parse_args()
    model_dirs = {
        "Tech Only": Path(args.tech_only),
        "Tech + Macro": Path(args.tech_macro),
    }

    plot_df = build_plot_dataframe(model_dirs)
    methods = [spec["label"] for spec in METHOD_SPECS]

    fig, ax = plt.subplots(figsize=(14, 7))

    positions_left = [i - 0.18 for i in range(1, len(methods) + 1)]
    positions_right = [i + 0.18 for i in range(1, len(methods) + 1)]

    tech_only_data = [
        plot_df[
            (plot_df["model"] == "Tech Only") & (plot_df["method"] == method)
        ]["accuracy"].to_numpy()
        for method in methods
    ]
    tech_macro_data = [
        plot_df[
            (plot_df["model"] == "Tech + Macro") & (plot_df["method"] == method)
        ]["accuracy"].to_numpy()
        for method in methods
    ]

    bp_left = ax.boxplot(
        tech_only_data,
        positions=positions_left,
        widths=0.3,
        patch_artist=True,
        manage_ticks=False,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="navy",
            markeredgecolor="black",
            markersize=7,
        ),
    )
    bp_right = ax.boxplot(
        tech_macro_data,
        positions=positions_right,
        widths=0.3,
        patch_artist=True,
        manage_ticks=False,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="darkred",
            markeredgecolor="black",
            markersize=7,
        ),
    )

    style_boxplot(bp_left, facecolor="skyblue")
    style_boxplot(bp_right, facecolor="lightcoral")

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=0, ha="center")
    ax.set_ylim(0.50, 1.00)
    ax.set_yticks(np.arange(0.50, 1.01, 0.10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("Validation Method")
    ax.set_title(f"{args.stock} Validation Accuracy Performance")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        Patch(facecolor="skyblue", edgecolor="black", label="Tech Only"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Tech + Macro"),
    ]
    ax.legend(handles=legend_handles)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
