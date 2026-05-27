import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).parent / "content" / "gsm.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "inspection_data"


def load_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = {
        "",
        "-",
        "—",
        "–",
        "N/A",
        "n/a",
        "NA",
        "na",
        "None",
        "none",
        "null",
        "Null",
    }
    out = df.copy()
    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in obj_cols:
        out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        out[c] = out[c].replace(list(missing_tokens), np.nan)
    return out


def clean_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]", "_", name)
    return cleaned[:120]


def clean_mojibake(text: str) -> str:
    if pd.isna(text):
        return np.nan
    s = str(text)
    replacements = {
        "<e2><82><ac>": "EUR ",
        "<c2><a3>": "GBP ",
        "<e2><82><b9>": "INR ",
        "<e2><80><89>": " ",
        "<c2><a0>": " ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def parse_price_eur_from_misc(text: str) -> float:
    if pd.isna(text):
        return np.nan
    s = clean_mojibake(text).replace(",", "")
    currency_patterns = [
        (1.0, [r"([0-9]+(?:\.[0-9]+)?)\s*EUR\b", r"EUR\s*([0-9]+(?:\.[0-9]+)?)"]),
        (0.93, [r"\$\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*USD\b"]),
        (1.17, [r"GBP\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*GBP\b"]),
        (0.011, [r"INR\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*INR\b"]),
    ]
    for rate, patterns in currency_patterns:
        for pattern in patterns:
            m = re.search(pattern, s, flags=re.I)
            if m:
                return float(m.group(1)) * rate
    return np.nan


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_overview_missing(df: pd.DataFrame, out_path: Path) -> None:
    n = len(df)
    miss_ratio = (df.isna().sum() / n).sort_values(ascending=False)
    plt.figure(figsize=(16, 7))
    plt.bar(range(len(miss_ratio)), miss_ratio.values)
    plt.title("Missing Ratio by Column")
    plt.xlabel("Columns (sorted)")
    plt.ylabel("Missing ratio")
    plt.ylim(0, 1.0)
    save_fig(out_path)


def plot_overview_unique(df: pd.DataFrame, out_path: Path) -> None:
    n = len(df)
    uniq = df.nunique(dropna=True)
    non_missing = n - df.isna().sum()
    uniq_ratio = (uniq / non_missing.replace(0, np.nan)).fillna(0.0)
    miss_ratio = (df.isna().sum() / n).fillna(0.0)

    plt.figure(figsize=(9, 7))
    plt.scatter(miss_ratio.values, uniq_ratio.values, alpha=0.75)
    for col, x, y in zip(df.columns, miss_ratio.values, uniq_ratio.values):
        if x > 0.9 or y > 0.9:
            plt.annotate(col, (x, y), fontsize=7, alpha=0.8)
    plt.title("Column-level scatter: missing_ratio vs unique_ratio")
    plt.xlabel("missing_ratio")
    plt.ylabel("unique_ratio_among_non_missing")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    save_fig(out_path)


def plot_key_duplicates(df: pd.DataFrame, keys: list[str], out_path: Path) -> None:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return
    tmp = df[df[keys].notna().all(axis=1)].copy()
    if len(tmp) == 0:
        return
    grp = tmp.groupby(keys).size().reset_index(name="group_size")
    dist = grp["group_size"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(dist.index.astype(str), dist.values)
    plt.title(f"Duplicate group size distribution: {' + '.join(keys)}")
    plt.xlabel("group_size")
    plt.ylabel("number_of_groups")
    save_fig(out_path)


def plot_price_visuals(df: pd.DataFrame, out_dir: Path) -> None:
    if "misc_price" not in df.columns:
        return
    price_eur = df["misc_price"].apply(parse_price_eur_from_misc).dropna()
    if len(price_eur) == 0:
        return

    # Histogram
    plt.figure(figsize=(12, 5))
    plt.hist(price_eur.values, bins=60, edgecolor="black", alpha=0.8)
    plt.title("Price distribution (EUR parsed from misc_price)")
    plt.xlabel("price_eur")
    plt.ylabel("count")
    save_fig(out_dir / "price_histogram.png")

    # Dot plot (sorted values)
    sorted_vals = np.sort(price_eur.values)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(sorted_vals)), sorted_vals, ".", alpha=0.6)
    plt.title("Price dot plot (sorted)")
    plt.xlabel("sorted index")
    plt.ylabel("price_eur")
    save_fig(out_dir / "price_dotplot_sorted.png")

    # Tier counts
    tier = pd.cut(
        price_eur,
        bins=[0, 150, 400, 800, np.inf],
        labels=["budget", "mid_range", "premium", "flagship"],
        right=True,
    )
    vc = tier.value_counts().reindex(["budget", "mid_range", "premium", "flagship"]).fillna(0)
    plt.figure(figsize=(8, 5))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title("Price tier counts from misc_price")
    plt.xlabel("tier")
    plt.ylabel("count")
    save_fig(out_dir / "price_tier_counts.png")


def plot_column_visual(df: pd.DataFrame, column: str, out_dir: Path, top_k: int) -> None:
    s = df[column]
    missing = int(s.isna().sum())
    non_missing = int(len(s) - missing)
    unique = int(s.nunique(dropna=True))
    uniq_ratio = (unique / non_missing) if non_missing else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(column, fontsize=12)

    # Panel 1: missing/non-missing
    axes[0].bar(["non_missing", "missing"], [non_missing, missing], color=["#4c78a8", "#f58518"])
    axes[0].set_title("Missing count")
    axes[0].tick_params(axis="x", rotation=20)

    # Panel 2: unique ratio
    axes[1].bar(["unique_ratio"], [uniq_ratio], color=["#54a24b"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Unique ratio among non-missing")

    # Panel 3: distribution
    # Try numeric conversion
    numeric = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")
    numeric_non_na = numeric.dropna()
    if len(numeric_non_na) >= max(50, int(0.3 * non_missing)):
        axes[2].hist(numeric_non_na.values, bins=30, edgecolor="black", alpha=0.8)
        axes[2].set_title("Histogram (numeric-like)")
    else:
        vc = s.dropna().astype(str).value_counts().head(top_k)
        if len(vc):
            axes[2].bar(range(len(vc)), vc.values)
            axes[2].set_xticks(range(len(vc)))
            axes[2].set_xticklabels(vc.index, rotation=70, ha="right", fontsize=7)
            axes[2].set_title(f"Top {min(top_k, len(vc))} values")
        else:
            axes[2].text(0.5, 0.5, "No non-missing values", ha="center", va="center")
            axes[2].set_title("Distribution")
            axes[2].set_xticks([])
            axes[2].set_yticks([])

    save_fig(out_dir / f"{clean_name(column)}.png")


def write_summary(df: pd.DataFrame, out_dir: Path, input_path: Path) -> None:
    n_rows, n_cols = df.shape
    summary_path = out_dir / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("## GSM Inspection Summary\n\n")
        f.write(f"- input: `{input_path}`\n")
        f.write(f"- shape: {n_rows} rows x {n_cols} cols\n")
        f.write("- output: image-only (no csv)\n")
        f.write("- key visuals:\n")
        f.write("  - `overview/missing_ratio_all_columns.png`\n")
        f.write("  - `overview/missing_vs_unique_scatter.png`\n")
        f.write("  - `overview/duplicate_groups_oem_model.png`\n")
        f.write("  - `overview/duplicate_groups_model.png`\n")
        f.write("  - `overview/price_histogram.png`\n")
        f.write("  - `overview/price_dotplot_sorted.png`\n")
        f.write("  - `overview/price_tier_counts.png`\n")
        f.write("  - `columns/*.png` (all columns)\n")


def write_column_description(df: pd.DataFrame, out_dir: Path) -> None:
    n = len(df)
    path = out_dir / "COLUMN_DESCRIPTION.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("## Column Description (All Columns)\n\n")
        f.write("| column | dtype | missing | missing_ratio | unique | unique_ratio_non_missing |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for c in df.columns:
            s = df[c]
            missing = int(s.isna().sum())
            non_missing = n - missing
            unique = int(s.nunique(dropna=True))
            miss_r = missing / n if n else 0.0
            uniq_r = unique / non_missing if non_missing else 0.0
            f.write(
                f"| {c} | {s.dtype} | {missing} | {miss_r:.4f} | {unique} | {uniq_r:.4f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-friendly GSM inspection with images only.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to original gsm.csv")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output folder (default: /inspection_data in project root)",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Top-K values for categorical visuals")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    overview_dir = out_dir / "overview"
    columns_dir = out_dir / "columns"
    overview_dir.mkdir(parents=True, exist_ok=True)
    columns_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[INFO] Loading: {input_path}")
    df_raw = load_csv_robust(input_path)
    df = normalize_missing(df_raw)
    print(f"[INFO] Loaded shape: {df.shape}")

    # Text outputs (only two files)
    write_summary(df, out_dir, input_path)
    write_column_description(df, out_dir)

    # Overview visuals
    plot_overview_missing(df, overview_dir / "missing_ratio_all_columns.png")
    plot_overview_unique(df, overview_dir / "missing_vs_unique_scatter.png")
    plot_key_duplicates(df, ["oem", "model"], overview_dir / "duplicate_groups_oem_model.png")
    plot_key_duplicates(df, ["model"], overview_dir / "duplicate_groups_model.png")
    plot_price_visuals(df, overview_dir)

    # Per-column visuals (all columns)
    for col in df.columns:
        plot_column_visual(df, col, columns_dir, top_k=args.top_k)

    print(f"[INFO] Done. Output folder: {out_dir}")


if __name__ == "__main__":
    main()

