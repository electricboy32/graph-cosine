#!/usr/bin/env python3
"""
Analyze Netflix Content Similarity and Genre Distribution

Features:
- Cluster Netflix movies/TV shows by description similarity
- Summarize genres per cluster and overall
- Plot top genres bar chart (only if --genre is not provided)
- Retrieve all movies/shows by a specific genre (sorted by rating)
- Writes all available genres (for --genre option) to available_genres.txt

CLI options:
    --csv PATH         Path to input CSV (default: data.csv)
    --clusters N       Number of clusters for KMeans (default: 20)
    --top-genres M     Number of genres to plot (default: 20)
    --genre GENRE      Filter and list all titles in the specified genre (case-insensitive).
                       If set, skips genre bar chart.
    --no-show          Do not show plot (headless)

Outputs:
    - netflix_clusters.csv: Data with cluster labels (not produced if --genre is set)
    - Cluster summary printed to console (not shown if --genre is set)
    - netflix_top_genres.png: Bar chart of top genres (not produced if --genre is set)
    - titles_in_<genre>.csv: (only if --genre is specified) CSV of titles in that genre
    - available_genres.txt: All unique valid genres, one per line (auto-generated)

Examples:
    python analyze_netflix_similarity_and_genres.py --genre "Dramas"
"""

import sys
import argparse
import os

def import_or_die():
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances
        return pd, np, plt, sns, TfidfVectorizer, MiniBatchKMeans, pairwise_distances
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn.", file=sys.stderr)
        sys.exit(1)

def validate_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing required columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)

def preprocess_listed_in(listed_in_col):
    # listed_in: "Dramas, International Movies"
    return listed_in_col.apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])

def vectorize_descriptions(descriptions):
    _, _, _, _, TfidfVectorizer, _, _ = import_or_die()
    vec = TfidfVectorizer(
        stop_words='english',
        min_df=2,
        max_features=10000
    )
    X = vec.fit_transform(descriptions)
    return X, vec

def perform_clustering(X, n_clusters, random_state=42):
    _, _, _, _, _, MiniBatchKMeans, _ = import_or_die()
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=512)
    labels = km.fit_predict(X)
    return km, labels

def cluster_top_titles(X, km, df, cluster_id, top_n=10):
    # Find the top_n titles closest to the centroid (cosine similarity)
    _, np, _, _, _, _, _ = import_or_die()
    from sklearn.metrics import pairwise_distances
    indices = df.index[df['cluster'] == cluster_id].tolist()
    if not indices:
        return []
    X_cluster = X[indices]
    centroid = km.cluster_centers_[cluster_id].reshape(1, -1)
    # Cosine distances: lower = more similar
    dists = pairwise_distances(X_cluster, centroid, metric='cosine').reshape(-1)
    top_idx = np.argsort(dists)[:top_n]
    return df.iloc[indices].iloc[top_idx]['title'].tolist()

def cluster_predominant_genres(df, cluster_id, top_n=3):
    genres = df[df['cluster'] == cluster_id]['listed_in_exploded']
    if genres.empty:
        return []
    top_genres = genres.value_counts().nlargest(top_n).index.tolist()
    return top_genres

def plot_top_genres(df_exploded, top_m, output_path, show_plot=True):
    _, _, plt, sns, _, _, _ = import_or_die()
    genre_counts = df_exploded['listed_in_exploded'].value_counts().nlargest(top_m)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=genre_counts.index, x=genre_counts.values, palette="viridis")
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.title(f"Top {top_m} Genres on Netflix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    if show_plot:
        plt.show()
    plt.close()

def print_cluster_summary(cluster_infos):
    import tabulate
    headers = ["Cluster", "Count", "Top Genres"]
    rows = []
    for info in cluster_infos:
        rows.append([info["cluster"], info["count"], ", ".join(info["top_genres"])])
    print("\nCluster Summary:")
    print(tabulate.tabulate(rows, headers=headers, tablefmt="github"))

def filter_titles_by_genre(df_exploded, genre):
    """Filter all titles in the specified genre (case-insensitive, whitespace-trim) and return deduped DataFrame"""
    pd, *_ = import_or_die()
    # Normalize genre for comparison
    requested = genre.strip().lower()
    # Find available genres (normalized for lookup)
    genre_map = {g.strip().lower(): g for g in df_exploded['listed_in_exploded'].dropna().unique()}
    if requested not in genre_map:
        return None, sorted(genre_map.values())
    # Filter rows
    filtered = df_exploded[df_exploded['listed_in_exploded'].str.strip().str.lower() == requested]
    # Deduplicate by title (keep first occurrence)
    dedup_cols = ['title','type','rating','release_year','duration','description']
    available_cols = [col for col in dedup_cols if col in filtered.columns]
    if not available_cols:
        available_cols = ['title','type','rating']
        available_cols = [col for col in available_cols if col in filtered.columns]
    filtered_out = filtered.drop_duplicates(subset=['title'])[available_cols].copy()
    return filtered_out, None

def write_available_genres(genres, path="available_genres.txt"):
    """Write unique genres to a text file, one per line."""
    with open(path, "w", encoding="utf-8") as f:
        for g in genres:
            f.write(f"{g}\n")

def main():
    pd, np, plt, sns, TfidfVectorizer, MiniBatchKMeans, pairwise_distances = import_or_die()
    parser = argparse.ArgumentParser(
        description="Analyze Netflix similarity, genres, and list titles by genre. Writes available genres to available_genres.txt automatically."
    )
    parser.add_argument("--csv", type=str, default="data.csv", help="Path to CSV file [default: data.csv]")
    parser.add_argument("--clusters", type=int, default=20, help="Number of clusters for KMeans [default: 20]")
    parser.add_argument("--top-genres", type=int, default=20, help="Number of top genres to plot [default: 20]")
    parser.add_argument("--genre", type=str, default=None, help="Specify a genre to list all titles in that genre (case-insensitive)")
    parser.add_argument("--no-show", action="store_true", help="Suppress plot display")
    args = parser.parse_args()

    # 1. Load CSV
    if not os.path.exists(args.csv):
        print(f"CSV file '{args.csv}' not found.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv)
    required_cols = ['title', 'type', 'description', 'listed_in']
    validate_columns(df, required_cols)

    # 2. Preprocess genres & descriptions
    df['listed_in_list'] = preprocess_listed_in(df['listed_in'])
    df_exploded = df.explode('listed_in_list').rename(columns={'listed_in_list': 'listed_in_exploded'})

    # 2a. Write available genres file
    unique_genres = sorted({g for g in df_exploded['listed_in_exploded'].dropna().unique() if g})
    write_available_genres(unique_genres, "available_genres.txt")
    print(f"Wrote {len(unique_genres)} available genres to available_genres.txt")

    # 2b. If --genre: filter and output
    if args.genre:
        print(f"\nFiltering titles for genre: '{args.genre}'")
        filtered, genre_list = filter_titles_by_genre(df_exploded, args.genre)
        if filtered is None or filtered.empty:
            print(f"Genre '{args.genre}' not found. Available genres:")
            print(", ".join(genre_list))
            sys.exit(1)
        # Try to sort by 'rating' column if present
        if 'rating' in filtered.columns:
            try:
                filtered['rating_numeric'] = pd.to_numeric(filtered['rating'], errors='coerce')
                if filtered['rating_numeric'].notna().any():
                    # Sort numeric descending, NaN last
                    filtered = filtered.sort_values(by=['rating_numeric','rating'], ascending=[False, True])
                else:
                    filtered = filtered.sort_values(by=['rating'], ascending=True)
                filtered = filtered.drop(columns=['rating_numeric'])
            except Exception:  # fallback
                filtered = filtered.sort_values(by=['rating'], ascending=True)
        # Print results with row numbers
        print(f"\nFound {len(filtered)} titles in genre '{args.genre}':")
        print(filtered.reset_index(drop=True).to_string(index=True))
        # Save to CSV
        genre_fname = args.genre.strip().replace(" ", "_").lower()
        out_path = f"titles_in_{genre_fname}.csv"
        filtered.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")
        # Continue with rest of analysis as usual

    # 3. TF-IDF vectorization
    X, tfidf_vec = vectorize_descriptions(df['description'].fillna(''))

    # 4. Clustering
    km, labels = perform_clustering(X, args.clusters)
    df['cluster'] = labels

    # 5. Cluster summaries
    cluster_infos = []
    for cluster_id in range(args.clusters):
        count = (df['cluster'] == cluster_id).sum()
        # Titles closest to centroid (by cosine similarity)
        top_titles = cluster_top_titles(X, km, df, cluster_id, top_n=10)
        # Top genres in cluster
        genres_in_cluster = df[df['cluster'] == cluster_id].explode('listed_in_list')
        top_genres = genres_in_cluster['listed_in_list'].value_counts().head(3).index.tolist()
        cluster_infos.append({
            "cluster": cluster_id,
            "count": count,
            "top_titles": top_titles,
            "top_genres": top_genres,
        })

    # 6. Save CSV (only if --genre not provided)
    df_out = df[['title', 'type', 'description', 'listed_in', 'cluster']]
    if args.genre is None:
        df_out.to_csv("netflix_clusters.csv", index=False)
        print("Saved clustering results to netflix_clusters.csv")
    else:
        print("Skipping clusters CSV because --genre flag was provided.")

    # 7. Print summary table (only if --genre not provided)
    if args.genre is None:
        try:
            print_cluster_summary(cluster_infos)
        except ImportError:
            # Fallback if tabulate not installed
            print("\nCluster Summary:")
            print("Cluster | Count | Top Genres")
            for info in cluster_infos:
                print(f"{info['cluster']:7} | {info['count']:5} | {', '.join(info['top_genres'])}")
    else:
        print("Skipping cluster summary because --genre flag was provided.")

    # 8. Plot genre distribution (only if --genre not provided)
    if args.genre is None:
        plot_top_genres(df_exploded, args.top_genres, "netflix_top_genres.png", show_plot=not args.no_show)
    else:
        print("Skipping top-genre chart because --genre flag was provided.")

if __name__ == "__main__":
    main()