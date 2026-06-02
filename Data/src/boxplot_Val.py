import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.ticker import PercentFormatter

# ===== 1) Load Excel file =====
file_path = "Bluechip Stocks Performance Summary.xlsx"
df = pd.read_excel(file_path)

# ===== 2) Use first row as header =====
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# ===== 3) Rename important columns =====
df = df.rename(columns={
    "Model & Metric": "metric_or_model",
    "Stock": "stock"
})

# Forward fill labels
df["metric_or_model"] = df["metric_or_model"].ffill()

# ===== 4) Keep only Validation Performance section =====
stop_idx = df[df["metric_or_model"] == "Test Accuracy"].index
if len(stop_idx) > 0:
    df_val = df.loc[:stop_idx[0] - 1].copy()
else:
    df_val = df.copy()

# ===== 5) Detect model for each row =====
current_model = None
models = []

for val in df_val["metric_or_model"]:
    text = str(val)

    if "CNN-TIC" in text:
        current_model = "CNN-TIC"
    elif "CNN-MIC" in text:
        current_model = "CNN-MIC"

    models.append(current_model)

df_val["model"] = models

# ===== 6) Keep only Mean Accuracy rows =====
mean_acc = df_val[df_val["metric_or_model"] == "Mean Accuracy"].copy()

# ===== 7) Exact validation column names =====
validation_cols = [
    "Chronological Holdout Validation with Stratified K-Fold",
    "Expanding-Window Walk-Forward Validation",
    "Nested Walk-Forward\nValidation",
    "Blocked Cross-\nValidation (BCV)",
    "Blocked Cross-Validation with Partial Undersampling"
]

# ===== 8) Check if columns exist =====
missing_cols = [col for col in validation_cols if col not in df.columns]
if missing_cols:
    print("These columns were not found in the Excel file:")
    for col in missing_cols:
        print(f" - {repr(col)}")
    print("\nAvailable columns are:")
    for col in df.columns:
        print(f" - {repr(col)}")
    raise KeyError("Some validation columns are missing. Check exact Excel header names.")

# ===== 9) Convert to long format =====
plot_df = mean_acc.melt(
    id_vars=["model", "stock"],
    value_vars=validation_cols,
    var_name="validation_method",
    value_name="accuracy"
)

plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors="coerce")

# Remove blank stock names if any
plot_df["stock"] = plot_df["stock"].astype(str).str.strip()
plot_df = plot_df[plot_df["stock"].notna()]
plot_df = plot_df[plot_df["stock"] != ""]

# ===== 10) Labels shown in the plot =====
label_map = {
    "Chronological Holdout Validation with Stratified K-Fold": "Chronological Holdout\nwith Stratified K-Fold",
    "Expanding-Window Walk-Forward Validation": "Expanding-Window\nWalk-Forward",
    "Nested Walk-Forward\nValidation": "Nested Walk-Forward",
    "Blocked Cross-\nValidation (BCV)": "Blocked Cross-Validation",
    "Blocked Cross-Validation with Partial Undersampling": "Blocked CV with\nPartial Undersampling"
}

methods = validation_cols
method_labels = [label_map[m] for m in methods]

# ===== 11) Use one common y-axis scale for all figures =====
y_min = 0.50
y_max = 1.00
y_ticks = np.arange(0.50, 1.01, 0.10)

# ===== 12) Create one figure per stock =====
stocks = plot_df["stock"].dropna().unique()

for stock_name in stocks:
    stock_df = plot_df[plot_df["stock"] == stock_name]

    cnn_tic_data = [
        stock_df[
            (stock_df["model"] == "CNN-TIC") &
            (stock_df["validation_method"] == method)
        ]["accuracy"].dropna()
        for method in methods
    ]

    cnn_mic_data = [
        stock_df[
            (stock_df["model"] == "CNN-MIC") &
            (stock_df["validation_method"] == method)
        ]["accuracy"].dropna()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    positions_tic = [i - 0.18 for i in range(1, len(methods) + 1)]
    positions_mic = [i + 0.18 for i in range(1, len(methods) + 1)]

    bp1 = ax.boxplot(
        cnn_tic_data,
        positions=positions_tic,
        widths=0.3,
        patch_artist=True,
        manage_ticks=False,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="navy",
            markeredgecolor="black",
            markersize=8
        )
    )

    bp2 = ax.boxplot(
        cnn_mic_data,
        positions=positions_mic,
        widths=0.3,
        patch_artist=True,
        manage_ticks=False,
        showmeans=True,
        meanprops=dict(
            marker="o",
            markerfacecolor="darkred",
            markeredgecolor="black",
            markersize=8
        )
    )

    # Colors
    for box in bp1["boxes"]:
        box.set(facecolor="skyblue", edgecolor="black")

    for box in bp2["boxes"]:
        box.set(facecolor="lightcoral", edgecolor="black")

    for element in ["whiskers", "caps", "medians"]:
        for item in bp1[element]:
            item.set(color="black", linewidth=1.5)
        for item in bp2[element]:
            item.set(color="black", linewidth=1.5)

    # Plot the actual values so the comparison stays visible even when
    # each method/model pair only has one observation.
    tic_values = [data.iloc[0] if not data.empty else np.nan for data in cnn_tic_data]
    mic_values = [data.iloc[0] if not data.empty else np.nan for data in cnn_mic_data]

    ax.scatter(
        positions_tic,
        tic_values,
        color="navy",
        edgecolors="black",
        s=90,
        zorder=3
    )
    ax.scatter(
        positions_mic,
        mic_values,
        color="darkred",
        edgecolors="black",
        s=90,
        zorder=3
    )

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(method_labels, rotation=0, ha="center")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("Validation Method")
    ax.set_title(f"Validation Accuracy Comparison of CNN-TIC and CNN-MIC, {stock_name}")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_elements = [
        Patch(facecolor="skyblue", edgecolor="black", label="CNN-TIC"),
        Patch(facecolor="lightcoral", edgecolor="black", label="CNN-MIC")
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()
