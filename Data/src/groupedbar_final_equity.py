import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_FILES = [
    Path("Bluechip Stocks Performance Summary.xlsx"),
    Path("Penny Stocks Performance Summary.xlsx"),
    Path("Crypto Performance Summary.xlsx"),
]

METHOD_LABEL_MAP = {
    "Chronological Holdout Validation with Stratified K-Fold":
        "Chronological Holdout\nwith Stratified K-Fold",
    "Expanding-Window Walk-Forward Validation":
        "Expanding-Window\nWalk-Forward",
    "Expanding-Window Walk-Forward Analysis (WFA)":
        "Expanding-Window\nWalk-Forward",
    "Nested Walk-Forward Validation": "Nested Walk-Forward",
    "Nested Walk-Forward\nValidation": "Nested Walk-Forward",
    "Blocked Cross-Validation (BCV)": "Blocked Cross-Validation",
    "Blocked Cross-\nValidation (BCV)": "Blocked Cross-Validation",
    "Balanced Blocked Cross-Validation":
        "Blocked CV with\nPartial Undersampling",
    "Blocked Cross-Validation with Partial Undersampling":
        "Blocked CV with\nPartial Undersampling",
}

INITIAL_EQUITY = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts for final equity from the summary Excel files."
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Path to a summary Excel file. Repeat the flag to process multiple files.",
    )
    return parser.parse_args()


def detect_financial_header_row(df_raw: pd.DataFrame) -> int:
    for idx, row in df_raw.iterrows():
        row_values = [str(value).strip() for value in row.tolist() if pd.notna(value)]
        if "Financial Performance" in row_values:
            return idx + 1
    raise ValueError("Could not find the financial performance section in the workbook.")


def normalize_method_label(label: str) -> str:
    clean_label = str(label).strip()
    return METHOD_LABEL_MAP.get(clean_label, clean_label)


def load_final_equity_data(file_path: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, header=None)
    header_row = detect_financial_header_row(df_raw)

    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = headers
    df = df.rename(columns={"Model & Metric": "metric_or_model", "Stock": "stock"})

    if "metric_or_model" not in df.columns or "stock" not in df.columns:
        raise KeyError(f"Expected 'Model & Metric' and 'Stock' columns in {file_path}")

    df["metric_or_model"] = df["metric_or_model"].ffill()

    models = []
    current_model = None
    for value in df["metric_or_model"]:
        text = str(value)
        if "CNN-TIC" in text:
            current_model = "CNN-TIC"
        elif "CNN-MIC" in text:
            current_model = "CNN-MIC"
        models.append(current_model)
    df["model"] = models

    final_equity_rows = df[df["metric_or_model"] == "Final Equity (PHP)"].copy()

    method_columns = [
        column for column in final_equity_rows.columns
        if column not in {"metric_or_model", "stock", "model"}
    ]

    plot_df = final_equity_rows.melt(
        id_vars=["model", "stock"],
        value_vars=method_columns,
        var_name="validation_method",
        value_name="final_equity",
    )
    plot_df["final_equity"] = pd.to_numeric(plot_df["final_equity"], errors="coerce")
    plot_df["stock"] = plot_df["stock"].astype(str).str.strip()
    plot_df["validation_method"] = plot_df["validation_method"].apply(normalize_method_label)
    plot_df = plot_df.dropna(subset=["final_equity"])
    plot_df = plot_df[plot_df["stock"] != ""]
    plot_df["source_file"] = file_path.name

    return plot_df


def plot_grouped_bars(plot_df: pd.DataFrame, source_name: str) -> None:
    methods = list(dict.fromkeys(plot_df["validation_method"]))
    stocks = plot_df["stock"].dropna().unique()

    for stock_name in stocks:
        stock_df = plot_df[plot_df["stock"] == stock_name]
        tic_values = []
        mic_values = []

        for method in methods:
            tic_match = stock_df[
                (stock_df["model"] == "CNN-TIC")
                & (stock_df["validation_method"] == method)
            ]["final_equity"]
            mic_match = stock_df[
                (stock_df["model"] == "CNN-MIC")
                & (stock_df["validation_method"] == method)
            ]["final_equity"]

            tic_values.append(tic_match.iloc[0] if not tic_match.empty else np.nan)
            mic_values.append(mic_match.iloc[0] if not mic_match.empty else np.nan)

        x = np.arange(len(methods))
        width = 0.34
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(
            x - width / 2,
            tic_values,
            width,
            color="skyblue",
            edgecolor="black",
            label="CNN-TIC",
        )
        ax.bar(
            x + width / 2,
            mic_values,
            width,
            color="lightcoral",
            edgecolor="black",
            label="CNN-MIC",
        )

        visible_values = [value for value in tic_values + mic_values if pd.notna(value)] + [INITIAL_EQUITY]
        min_value = min(visible_values)
        max_value = max(visible_values)
        span = max_value - min_value if max_value != min_value else max_value if max_value else 1
        y_min = min_value - span * 0.12
        y_max = max_value + span * 0.10

        ax.axhline(
            INITIAL_EQUITY,
            color="darkgreen",
            linestyle=(0, (6, 4)),
            linewidth=2.0,
            alpha=0.9,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0, ha="center")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Final Equity (PHP)")
        ax.set_xlabel("Validation Method")
        ax.set_title(f"Final Equity Comparison of CNN-TIC and CNN-MIC, {stock_name}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        legend_handles = [
            Patch(facecolor="skyblue", edgecolor="black", label="CNN-TIC"),
            Patch(facecolor="lightcoral", edgecolor="black", label="CNN-MIC"),
        ]
        ax.legend(
            handles=legend_handles,
            title=source_name.replace(".xlsx", ""),
            loc="upper right",
            bbox_to_anchor=(0.985, 0.99),
            frameon=True,
            borderaxespad=0.4,
        )
        ax.annotate(
            "Initial Equity = PHP 10,000",
            xy=(0.02, INITIAL_EQUITY),
            xycoords=("axes fraction", "data"),
            xytext=(0, 0),
            textcoords="offset points",
            color="darkgreen",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="darkgreen",
                linewidth=1.2,
                alpha=0.95,
            ),
        )

        plt.tight_layout()
        plt.show()


def main() -> None:
    args = parse_args()
    files = [Path(path) for path in args.files] if args.files else DEFAULT_FILES

    for file_path in files:
        plot_df = load_final_equity_data(file_path)
        plot_grouped_bars(plot_df, file_path.name)


if __name__ == "__main__":
    main()
