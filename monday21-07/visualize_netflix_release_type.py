#!/usr/bin/env python3
"""
visualize_netflix_release_type.py

Visualize year-by-year Netflix release trends by type (e.g., Movie, TV Show) from a CSV dataset.
- Loads a CSV file with Netflix data.
- Validates required columns.
- Plots and saves a line chart showing yearly counts of each release type.
- Prints a year-by-year table of counts to the console.

Usage:
    python visualize_netflix_release_type.py [--csv path/to/file.csv] [--no-show]
"""

import sys

# --- Import required libraries, with graceful error handling ---

try:
    import pandas as pd
except ImportError:
    print(
        "Error: The 'pandas' library is required but not installed.\n"
        "Install it via pip:\n"
        "    pip install pandas"
    )
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Error: The 'matplotlib' library is required but not installed.\n"
        "Install it via pip:\n"
        "    pip install matplotlib"
    )
    sys.exit(1)

import argparse

# --- Required columns in the Netflix CSV ---
REQUIRED_COLUMNS = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "release_year", "rating", "duration",
    "listed_in", "description"
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load the Netflix data CSV and validate required columns.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded and validated DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(
            f"Error: The following required columns are missing from '{path}':\n"
            f"    {', '.join(missing)}"
        )
        sys.exit(1)

    return df


def plot_yearly_type_trend(
    df: pd.DataFrame,
    *,
    show: bool = True,
    save_path: str | None = "netflix_yearly_type_trend.png"
) -> pd.DataFrame:
    """
    Plot and save a line chart of the yearly counts of each release type (e.g. Movies, TV Shows).

    Args:
        df (pd.DataFrame): Netflix data.
        show (bool): Whether to display the plot window (default: True).
        save_path (str|None): Path to save the image (default: 'netflix_yearly_type_trend.png').

    Returns:
        pd.DataFrame: Pivoted table of counts (rows: years, columns: types).
    """
    # Prepare year/type counts
    grouped = df.groupby(['release_year', 'type']).size().reset_index(name='count')
    # Pivot so each type is a column, each row is a year
    pivot = grouped.pivot(index='release_year', columns='type', values='count').sort_index()
    # Fill missing years/types with zeros
    pivot = pivot.fillna(0).astype(int)

    # For consistent display: ensure all years between min and max are present
    all_years = range(pivot.index.min(), pivot.index.max() + 1)
    pivot = pivot.reindex(all_years, fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # extend if needed
    markers = ['o', 's', '^', 'D']

    for i, column in enumerate(pivot.columns):
        y = pivot[column]
        ax.plot(
            pivot.index, y,
            label=column,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6
        )
        # Annotate the last point of each series with the count value
        if len(y) > 0:
            ax.annotate(
                f"{y.iloc[-1]}",
                xy=(pivot.index[-1], y.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va='center', ha='left',
                fontsize=11, fontweight='bold',
                color=colors[i % len(colors)]
            )

    ax.set_title("Netflix Releases by Type per Year", fontsize=16, fontweight='bold')
    ax.set_xlabel("Release Year", fontsize=14)
    ax.set_ylabel("Number of Releases", fontsize=14)
    ax.legend(title="Type")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(list(pivot.index)[::max(1, len(pivot.index)//15)])  # only show ~15 ticks
    ax.set_xticklabels(list(pivot.index)[::max(1, len(pivot.index)//15)], rotation=45, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Line chart saved as '{save_path}'.")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pivot


def main():
    """
    Main routine: parse args, load data, plot yearly trend, print year-by-year counts table.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Netflix releases by type per year (Movie, TV Show, etc.)."
    )
    parser.add_argument(
        "--csv", type=str, default="data.csv",
        help="Path to the Netflix data CSV (default: data.csv)."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the chart window (for headless environments)."
    )
    args = parser.parse_args()

    # Load and validate data
    df = load_data(args.csv)

    # Plot and save yearly trend line chart
    yearly_counts = plot_yearly_type_trend(
        df,
        show=not args.no_show,
        save_path="netflix_yearly_type_trend.png"
    )

    # Print year-by-year table to console
    print("\nYear-by-year release counts (rows = year):")
    print(yearly_counts.to_string())
    """
    Main routine: parse args, load data, plot, print counts.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Netflix releases by type (Movie, TV Show, etc.)."
    )
    parser.add_argument(
        "--csv", type=str, default="data.csv",
        help="Path to the Netflix data CSV (default: data.csv)."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the chart window (for headless environments)."
    )
    args = parser.parse_args()

    # Load and validate data
    df = load_data(args.csv)

    # Plot and save chart
    plot_yearly_type_trend(
        df,
        show=not args.no_show,
        save_path="netflix_type_distribution.png"
    )

    # Print raw counts to console
    type_counts = df['type'].value_counts()
    print("Netflix release counts by type:")
    print(", ".join([f"{k}: {v}" for k, v in type_counts.items()]))


if __name__ == "__main__":
    main()