import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


DEFAULT_FILES = [
    Path("Bluechip Stocks Performance Summary.xlsx"),
    Path("Penny Stocks Performance Summary.xlsx"),
    Path("Crypto Performance Summary.xlsx"),
]

MODEL_LABELS = {
    "CNN-TIC": "CNN-TIC",
    "CNN-MIC": "CNN-MIC",
}

METHOD_LABEL_MAP = {
    "Chronological Holdout Validation with Stratified K-Fold":
        "Chronological Holdout\nwith Stratified K-Fold",
    "Expanding-Window Walk-Forward Validation":
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts for test mean accuracy from the summary Excel files."
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Path to a summary Excel file. Repeat the flag to process multiple files.",
    )
    return parser.parse_args()


def detect_test_section(df_raw: pd.DataFrame) -> tuple[int, int]:
    for idx, row in df_raw.iterrows():
        row_values = [str(value).strip() for value in row.tolist() if pd.notna(value)]
        if any(value in {"Test Accuracy", "Test Performance"} for value in row_values):
            header_row = idx + 1
            data_start_row = idx + 2
            return header_row, data_start_row
    raise ValueError("Could not find a test section marker in the workbook.")


def normalize_method_label(label: str) -> str:
    clean_label = str(label).strip()
    return METHOD_LABEL_MAP.get(clean_label, clean_label)


def load_test_accuracy_data(file_path: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, header=None)
    header_row, data_start_row = detect_test_section(df_raw)

    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[data_start_row:].copy()
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

    test_accuracy_rows = df[df["metric_or_model"] == "Test Accuracy"].copy()
    if test_accuracy_rows.empty:
        test_accuracy_rows = df[df["metric_or_model"] == "Mean Accuracy"].copy()

    method_columns = [
        column for column in test_accuracy_rows.columns
        if column not in {"metric_or_model", "stock", "model"}
    ]

    plot_df = test_accuracy_rows.melt(
        id_vars=["model", "stock"],
        value_vars=method_columns,
        var_name="validation_method",
        value_name="accuracy",
    )
    plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")
    plot_df["stock"] = plot_df["stock"].astype(str).str.strip()
    plot_df["validation_method"] = plot_df["validation_method"].apply(normalize_method_label)
    plot_df = plot_df.dropna(subset=["accuracy"])
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
            ]["accuracy"]
            mic_match = stock_df[
                (stock_df["model"] == "CNN-MIC")
                & (stock_df["validation_method"] == method)
            ]["accuracy"]

            tic_values.append(tic_match.iloc[0] if not tic_match.empty else np.nan)
            mic_values.append(mic_match.iloc[0] if not mic_match.empty else np.nan)

        x = np.arange(len(methods))
        width = 0.34

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - width / 2, tic_values, width, color="skyblue", edgecolor="black", label="CNN-TIC")
        ax.bar(x + width / 2, mic_values, width, color="lightcoral", edgecolor="black", label="CNN-MIC")

        visible_values = [value for value in tic_values + mic_values if pd.notna(value)]
        y_min = max(0.0, np.floor((min(visible_values) - 0.03) / 0.05) * 0.05)
        y_max = min(1.0, np.ceil((max(visible_values) + 0.03) / 0.05) * 0.05)

        if y_max - y_min < 0.20:
            y_max = min(1.0, y_min + 0.20)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0, ha="center")
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.05))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("Test Accuracy")
        ax.set_xlabel("Validation Method")
        ax.set_title(f"Test Mean Accuracy Comparison of CNN-TIC and CNN-MIC, {stock_name}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        legend_handles = [
            Patch(facecolor="skyblue", edgecolor="black", label="CNN-TIC"),
            Patch(facecolor="lightcoral", edgecolor="black", label="CNN-MIC"),
        ]
        ax.legend(handles=legend_handles, title=source_name.replace(".xlsx", ""))

        plt.tight_layout()
        plt.show()


def main() -> None:
    args = parse_args()
    files = [Path(path) for path in args.files] if args.files else DEFAULT_FILES

    for file_path in files:
        plot_df = load_test_accuracy_data(file_path)
        plot_grouped_bars(plot_df, file_path.name)


if __name__ == "__main__":
    main()
