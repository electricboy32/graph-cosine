#!/usr/bin/env python3
"""
visualize_netflix_country_content.py

Visualize Netflix content availability by country from a dataset similar to data.csv.

Features:
- Explodes comma-separated country lists so each row is (title, country).
- Aggregates total titles per country and counts by type (Movie, TV Show).
- Generates:
    1. Top N countries by total titles (horizontal bar).
    2. Stacked horizontal bar for top N: Movie vs TV Show.
    3. (Optional) Heatmap of country vs type for top 30 countries.
- CLI: --csv, --top, --no-show
- Tabular console summary of top N countries with totals and breakdowns.

Author: CosineAI Genie
"""

import sys
import os
import argparse
from typing import List

try:
    import pandas as pd
except ImportError:
    print("Missing required dependency: pandas. Please install it with 'pip install pandas'.", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing required dependency: matplotlib. Please install it with 'pip install matplotlib'.", file=sys.stderr)
    sys.exit(1)

try:
    import seaborn as sns
except ImportError:
    print("Missing required dependency: seaborn. Please install it with 'pip install seaborn'.", file=sys.stderr)
    sys.exit(1)


REQUIRED_COLUMNS = ["title", "country", "type"]

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

def preprocess_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the 'country' field so each row is (title, single_country, ...).
    Drops rows where country is NaN/empty after trimming.
    """
    df = df.copy()
    # Fill country NaN with empty string for processing
    df['country'] = df['country'].fillna('')
    # Split, trim, and explode
    df['country'] = df['country'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip()])
    df = df.explode('country')
    df['country'] = df['country'].str.strip()
    # Drop rows with country empty after trimming
    df = df[df['country'].astype(bool)]
    return df

def aggregate_country_totals(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series: index=country, value=total titles
    """
    return df.groupby('country').size().sort_values(ascending=False)

def aggregate_country_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame: index=(country), columns=type, values=counts
    """
    return df.groupby(['country', 'type']).size().unstack(fill_value=0)

def get_top_countries(country_total: pd.Series, n: int) -> List[str]:
    return country_total.head(n).index.tolist()

def plot_top_countries_bar(country_total: pd.Series, top_n: int, outfile: str, show: bool) -> None:
    plt.figure(figsize=(10, max(6, top_n // 2)))
    top = country_total.head(top_n)[::-1]
    ax = top.plot(kind='barh', color='#2a9d8f', edgecolor='black')
    for i, v in enumerate(top):
        ax.text(v + max(top)*0.01, i, str(v), va='center', fontsize=10)
    plt.title(f"Top {top_n} Countries by Total Netflix Titles")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_top_countries_stacked_bar(country_type: pd.DataFrame, top_countries: List[str], outfile: str, show: bool) -> None:
    plt.figure(figsize=(12, max(6, len(top_countries)//2)))
    data = country_type.loc[top_countries].fillna(0)
    data = data.loc[::-1]  # reverse for top-to-bottom
    # Stacked bar
    colors = {'Movie': '#e76f51', 'TV Show': '#264653'}
    types = [col for col in ['Movie', 'TV Show'] if col in data.columns]
    left = pd.Series(0, index=data.index)
    ax = plt.gca()
    for t in types:
        bars = ax.barh(data.index, data[t], left=left, color=colors.get(t, None), label=t, edgecolor='black')
        for i, (country, val) in enumerate(zip(data.index, data[t])):
            if val > 0:
                ax.text(left[country] + val/2, i, str(int(val)), va='center', ha='center', color='w', fontsize=9, fontweight='bold')
        left += data[t]
    plt.title(f"Top {len(top_countries)} Countries: Movie vs TV Show Breakdown")
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_country_type_heatmap(country_type: pd.DataFrame, top_countries: List[str], outfile: str, show: bool) -> None:
    # Subset and fill missing with zero
    data = country_type.loc[top_countries].fillna(0)
    plt.figure(figsize=(10, max(8, len(top_countries)*0.35)))
    sns.heatmap(data, annot=True, fmt=".0f", cmap="YlOrRd", cbar=True, linewidths=0.5)
    plt.title(f"Netflix Content by Country & Type (Top {len(top_countries)})")
    plt.xlabel("Type")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def print_country_table(country_total: pd.Series, country_type: pd.DataFrame, top_countries: List[str]) -> None:
    header = f"{'Country':<24} {'Total':>7} {'Movies':>8} {'TV Shows':>10}"
    print("\n" + header)
    print("-" * len(header))
    for c in top_countries:
        total = country_total.get(c, 0)
        movies = country_type.loc[c]['Movie'] if 'Movie' in country_type.columns and c in country_type.index else 0
        tv = country_type.loc[c]['TV Show'] if 'TV Show' in country_type.columns and c in country_type.index else 0
        print(f"{c:<24} {total:>7} {movies:>8} {tv:>10}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Netflix content availability by country."
    )
    parser.add_argument("--csv", default="data.csv", help="Path to input CSV file (default: data.csv)")
    parser.add_argument("--top", type=int, default=20, help="Top N countries to show (default: 20)")
    parser.add_argument("--no-show", action="store_true", help="Suppress plt.show() (for headless runs)")
    args = parser.parse_args()

    infile = args.csv
    top_n = args.top
    show = not args.no_show

    if not os.path.isfile(infile):
        print(f"Error: CSV file not found: {infile}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(infile)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_columns(df, REQUIRED_COLUMNS)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Preprocess
    df_exp = preprocess_countries(df)

    if df_exp.empty:
        print("Error: No valid country data after preprocessing.", file=sys.stderr)
        sys.exit(1)

    # Aggregations
    country_total = aggregate_country_totals(df_exp)
    country_type = aggregate_country_type(df_exp)
    top_countries = get_top_countries(country_total, top_n)

    # Plot 1: Top countries by total titles
    plot_top_countries_bar(
        country_total, top_n,
        outfile="netflix_top_countries_total.png",
        show=show
    )
    # Plot 2: Stacked bar by type
    plot_top_countries_stacked_bar(
        country_type, top_countries,
        outfile="netflix_top_countries_type_breakdown.png",
        show=show
    )
    # Plot 3: Heatmap (nice-to-have)
    try:
        top_countries_heatmap = get_top_countries(country_total, max(30, top_n))
        plot_country_type_heatmap(
            country_type, top_countries_heatmap,
            outfile="netflix_country_type_heatmap.png",
            show=show
        )
    except Exception:
        pass  # Don't fail if heatmap fails

    # Console output
    print_country_table(country_total, country_type, top_countries)

if __name__ == "__main__":
    main()