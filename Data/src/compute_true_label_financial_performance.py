from pathlib import Path

import numpy as np
import pandas as pd


STOCKS_ROOT = Path("Thesis") / "Stocks"
OUTPUT_PATH = Path("results") / "true_label_financial_performance.csv"

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-12-31")

INITIAL_BALANCE = 10_000.0
COMMISSION = 50.0
TAKE_PROFIT = 0.10
STOP_LOSS = 0.01

CATEGORY_NAMES = {
    "Blue chip stocks": "Blue-chip Stocks",
    "PennyStocks": "Penny Stocks",
    "Crypto": "Cryptocurrency",
}

OUTPUT_COLUMNS = [
    "Asset",
    "Category",
    "FinalEquity",
    "MeanAnnualReturn",
    "StdAnnualReturn",
]


def resolve_column(columns, expected):
    lookup = {str(col).strip().lower(): col for col in columns}
    key = expected.strip().lower()
    if key not in lookup:
        raise ValueError(f"Missing required column {expected!r}")
    return lookup[key]


def has_columns(csv_path, required):
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns
    except Exception:
        return False

    normalized = {str(col).strip().lower() for col in columns}
    return all(col.strip().lower() in normalized for col in required)


def parse_mixed_dates(values, source_name):
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        remaining = parsed.isna() & text.notna() & (text != "")
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(
            text.loc[remaining],
            format=fmt,
            errors="coerce",
        )

    remaining = parsed.isna() & text.notna() & (text != "")
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text.loc[remaining],
            dayfirst=True,
            errors="coerce",
        )

    failed = parsed.isna() & text.notna() & (text != "")
    if failed.any():
        examples = ", ".join(text.loc[failed].head(5).astype(str))
        raise ValueError(f"Could not parse dates in {source_name}: {examples}")

    return parsed.dt.normalize()


def find_label_files():
    label_files = []
    for csv_path in sorted(STOCKS_ROOT.rglob("*.csv")):
        if csv_path.parent.name.lower() != "labels":
            continue
        if has_columns(csv_path, ["Date", "Label"]):
            label_files.append(csv_path)

    if not label_files:
        raise FileNotFoundError(f"No true-label CSV files found under {STOCKS_ROOT}")

    return label_files


def asset_and_category(label_path):
    rel_parts = label_path.relative_to(STOCKS_ROOT).parts
    category_folder = rel_parts[0]
    if category_folder not in CATEGORY_NAMES:
        raise ValueError(f"Unknown category folder for {label_path}")

    asset = label_path.parent.parent.name
    return asset, CATEGORY_NAMES[category_folder]


def find_price_file(asset_root, asset, label_path):
    canonical = asset_root / asset / f"{asset}_TechnicalIndicators6-25.csv"
    if canonical.exists() and has_columns(canonical, ["Date", "Close"]):
        return canonical

    candidates = []
    for csv_path in asset_root.rglob("*.csv"):
        if csv_path == label_path:
            continue
        if not has_columns(csv_path, ["Date", "Close"]):
            continue

        name_lower = csv_path.name.lower()
        parts_lower = {part.lower() for part in csv_path.parts}
        score = 0

        if name_lower == f"{asset.lower()}_technicalindicators6-25.csv":
            score += 100
        if "technicalindicators6-25" in name_lower:
            score += 50
        if asset.lower() in name_lower:
            score += 20
        if csv_path.parent.name.lower() == asset.lower():
            score += 20
        if "raw" in parts_lower:
            score -= 20
        if "normalization" in parts_lower:
            score -= 30

        candidates.append((score, len(csv_path.parts), csv_path))

    if not candidates:
        raise FileNotFoundError(f"No matching price CSV found for {asset}")

    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2]).lower()))
    return candidates[0][2]


def load_labels(label_path):
    df = pd.read_csv(label_path)
    date_col = resolve_column(df.columns, "Date")
    label_col = resolve_column(df.columns, "Label")
    df = df[[date_col, label_col]].rename(columns={date_col: "Date", label_col: "Label"})
    df["Date"] = parse_mixed_dates(df["Date"], str(label_path))
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

    if df["Label"].isna().any():
        raise ValueError(f"Non-numeric labels found in {label_path}")

    df["Label"] = df["Label"].astype(int)
    invalid = sorted(set(df["Label"]) - {-1, 0, 1})
    if invalid:
        raise ValueError(f"Unexpected label values in {label_path}: {invalid}")

    return df.sort_values("Date").reset_index(drop=True)


def load_prices(price_path):
    df = pd.read_csv(price_path)
    date_col = resolve_column(df.columns, "Date")
    close_col = resolve_column(df.columns, "Close")
    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = parse_mixed_dates(df["Date"], str(price_path))
    df["Close"] = (
        df["Close"]
        .astype("string")
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    if df["Close"].isna().any():
        raise ValueError(f"Non-numeric close prices found in {price_path}")

    return df.sort_values("Date").reset_index(drop=True)


def align_test_window(labels, prices):
    df = pd.merge(labels, prices, on="Date", how="inner").sort_values("Date")
    df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No aligned rows in the 2021-2022 test period")

    return df


def financial_summary(df):
    balance = float(INITIAL_BALANCE)
    bought_amount = 0.0
    take_profit_price = None
    stop_loss_price = None

    equity_list = []

    for _, row in df.iterrows():
        price = float(row["Close"])
        action = int(row["Label"])

        if bought_amount != 0.0:
            if price > take_profit_price:
                balance = bought_amount * take_profit_price - COMMISSION
                bought_amount = 0.0
                take_profit_price = stop_loss_price = None
            elif price < stop_loss_price:
                balance = bought_amount * stop_loss_price - COMMISSION
                bought_amount = 0.0
                take_profit_price = stop_loss_price = None

        if action == 1 and balance > 0.0:
            bought_amount = (balance - COMMISSION) / price
            balance = 0.0
            take_profit_price = price * (1.0 + TAKE_PROFIT)
            stop_loss_price = price * (1.0 - STOP_LOSS)
        elif action == -1 and bought_amount != 0.0:
            balance = bought_amount * price - COMMISSION
            bought_amount = 0.0
            take_profit_price = stop_loss_price = None

        equity = balance + bought_amount * price
        equity_list.append(equity)

    last_price = float(df["Close"].iloc[-1])
    if bought_amount != 0.0:
        balance = bought_amount * last_price - COMMISSION
        bought_amount = 0.0

    df_out = df.copy()
    df_out["equity"] = equity_list
    df_out["year"] = df_out["Date"].dt.year

    yearly_rows = []
    for year, group in df_out.groupby("year"):
        start_eq = float(group["equity"].iloc[0])
        end_eq = float(group["equity"].iloc[-1])
        yearly_rows.append(
            {
                "year": int(year),
                "year_return": end_eq / start_eq - 1.0,
            }
        )

    yearly_df = pd.DataFrame(yearly_rows).sort_values("year")
    returns = yearly_df["year_return"].to_numpy()

    return {
        "FinalEquity": float(balance),
        "MeanAnnualReturn": float(returns.mean()),
        "StdAnnualReturn": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
    }


def main():
    rows = []
    for label_path in find_label_files():
        asset, category = asset_and_category(label_path)
        asset_root = label_path.parent.parent
        price_path = find_price_file(asset_root, asset, label_path)

        labels = load_labels(label_path)
        prices = load_prices(price_path)
        aligned = align_test_window(labels, prices)
        summary = financial_summary(aligned)

        rows.append(
            {
                "Asset": asset,
                "Category": category,
                **summary,
            }
        )

        print(
            f"{asset}: {len(aligned)} rows, {aligned['Date'].min().date()} "
            f"to {aligned['Date'].max().date()}, price={price_path}"
        )

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
