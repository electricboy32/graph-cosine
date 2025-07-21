#!/usr/bin/env python3
"""
Analyze Netflix Content Similarity and Genre Distribution

Features:
- Cluster Netflix movies/TV shows by description similarity
- Summarize genres per cluster and overall
- Plot top genres bar chart (always generated for UI, only shown in Matplotlib if --no-show is not set)
- Retrieve all movies/shows by a specific genre (sorted by rating)
- Modern PyQt6-based interactive UI: genre picker, title viewer, navigation, action buttons (unless --no-ui)
- Writes all available genres (for --genre option) to available_genres.txt

CLI options:
    --csv PATH         Path to input CSV (default: data.csv)
    --clusters N       Number of clusters for KMeans (default: 20)
    --top-genres M     Number of genres to plot (default: 20)
    --genre GENRE      Filter and list all titles in the specified genre (case-insensitive).
                       If set, skips cluster summary/table/auto-CSV, but enables GUI.
    --no-ui            Suppress PyQt6 GUI; just print results.
    --no-show          Suppress Matplotlib pop-up windows (does not affect GUI)

Outputs:
    - netflix_clusters.csv: Data with cluster labels (auto if no --genre or no UI; else only by UI button)
    - Cluster summary printed to console (not shown if --genre is set)
    - netflix_top_genres.png: Bar chart of top genres (always generated for UI)
    - titles_in_<genre>.csv: (only if --genre is specified and saved via UI or auto if --no-ui)
    - available_genres.txt: All unique valid genres, one per line (auto-generated)

Examples:
    python analyze_netflix_similarity_and_genres.py --genre "Dramas"
    python analyze_netflix_similarity_and_genres.py --genre "Comedies" --no-ui
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
        description="Analyze Netflix similarity, genres, and list titles by genre. Writes available genres to available_genres.txt automatically. If --genre is set, opens interactive UI browser for titles unless --no-ui or --no-show."
    )
    parser.add_argument("--csv", type=str, default="data.csv", help="Path to CSV file [default: data.csv]")
    parser.add_argument("--clusters", type=int, default=20, help="Number of clusters for KMeans [default: 20]")
    parser.add_argument("--top-genres", type=int, default=20, help="Number of top genres to plot [default: 20]")
    parser.add_argument("--genre", type=str, default=None, help="Specify a genre to list all titles in that genre (case-insensitive). Shows interactive UI unless --no-ui.")
    parser.add_argument("--no-ui", action="store_true", help="Suppress interactive UI for --genre (just print table)")
    parser.add_argument("--no-show", action="store_true", help="Suppress plot display and UI (headless)")
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
    filtered = None
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
                    filtered = filtered.sort_values(by=['rating_numeric','rating'], ascending=[False, True])
                else:
                    filtered = filtered.sort_values(by=['rating'], ascending=True)
                filtered = filtered.drop(columns=['rating_numeric'])
            except Exception:
                filtered = filtered.sort_values(by=['rating'], ascending=True)
        print(f"\nFound {len(filtered)} titles in genre '{args.genre}':")
        print(filtered.reset_index(drop=True).to_string(index=True))
        # Optionally save CSV if no UI
        if args.no_ui:
            genre_fname = args.genre.strip().replace(" ", "_").lower()
            out_path = f"titles_in_{genre_fname}.csv"
            filtered.to_csv(out_path, index=False)
            print(f"\nResults saved to {out_path}")

    # 3. TF-IDF vectorization
    X, tfidf_vec = vectorize_descriptions(df['description'].fillna(''))

    # 4. Clustering
    km, labels = perform_clustering(X, args.clusters)
    df['cluster'] = labels

    # 5. Cluster summaries, clusters_df
    cluster_infos = []
    for cluster_id in range(args.clusters):
        count = (df['cluster'] == cluster_id).sum()
        top_titles = cluster_top_titles(X, km, df, cluster_id, top_n=10)
        genres_in_cluster = df[df['cluster'] == cluster_id].explode('listed_in_list')
        top_genres = genres_in_cluster['listed_in_list'].value_counts().head(3).index.tolist()
        cluster_infos.append({
            "cluster": cluster_id,
            "count": count,
            "top_titles": top_titles,
            "top_genres": top_genres,
        })
    clusters_df = df[['title', 'type', 'description', 'listed_in', 'cluster']]

    # 6. Save CSV (only if --genre not provided and no UI)
    if args.genre is None and args.no_ui:
        clusters_df.to_csv("netflix_clusters.csv", index=False)
        print("Saved clustering results to netflix_clusters.csv")
    elif args.genre is not None and args.no_ui:
        print("Skipping clusters CSV because --genre flag was provided.")
    # If UI is to be launched, user saves CSV from UI

    # 7. Print summary table (only if --genre not provided)
    if args.genre is None:
        try:
            print_cluster_summary(cluster_infos)
        except ImportError:
            print("\nCluster Summary:")
            print("Cluster | Count | Top Genres")
            for info in cluster_infos:
                print(f"{info['cluster']:7} | {info['count']:5} | {', '.join(info['top_genres'])}")
    else:
        print("Skipping cluster summary because --genre flag was provided.")

    # 8. Plot genre distribution (always generate image for UI, only show if --no-show is not set and --genre is None)
    chart_path = "netflix_top_genres.png"
    plot_top_genres(df_exploded, args.top_genres, chart_path, show_plot=(args.genre is None and not args.no_show))

    # 9. If UI is desired, launch PyQt6 UI
    if args.genre and not args.no_ui:
        launch_pyqt_ui(df_exploded, clusters_df, available_genres, chart_path, initial_genre=args.genre)

if __name__ == "__main__":
    main()


def launch_pyqt_ui(df_exploded, clusters_df, available_genres, chart_path, initial_genre=None):
    """
    Launches a PyQt6 interactive UI for browsing Netflix titles and genres.
    - df_exploded: DataFrame of exploded genres
    - clusters_df: DataFrame with cluster assignment
    - available_genres: list of genre strings
    - chart_path: path to genre bar chart PNG (optional)
    - initial_genre: genre string for initial selection (optional)
    """
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
            QListWidget, QTextEdit, QFileDialog, QMessageBox, QComboBox, QSpacerItem, QSizePolicy
        )
        from PyQt6.QtGui import QPixmap, QFont
        from PyQt6.QtCore import Qt
    except ImportError:
        print("PyQt6 not available. Please install it with pip install PyQt6 to use this UI.")
        return

    import sys

    class MainWindow(QMainWindow):
        def __init__(self, df_exploded, clusters_df, available_genres, chart_path, initial_genre):
            super().__init__()
            self.df_exploded = df_exploded
            self.clusters_df = clusters_df
            self.available_genres = available_genres
            self.chart_path = chart_path
            self.setWindowTitle("Netflix Explorer")
            self.setMinimumSize(800, 600)

            self.genre = initial_genre or (available_genres[0] if available_genres else "")
            self.titles = []
            self.idx = 0

            # Widgets
            self.genre_combo = QComboBox()
            self.genre_combo.addItems(self.available_genres)
            if self.genre:
                self.genre_combo.setCurrentText(self.genre)
            self.genre_combo.currentTextChanged.connect(self.change_genre)

            self.lbl_title = QLabel("")
            self.lbl_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            self.lbl_title.setWordWrap(True)
            self.lbl_rating = QLabel("")
            self.lbl_year = QLabel("")
            self.lbl_duration = QLabel("")
            self.lbl_idx = QLabel("")

            self.txt_desc = QTextEdit("")
            self.txt_desc.setReadOnly(True)
            self.txt_desc.setMinimumHeight(100)
            self.txt_desc.setFont(QFont("Arial", 11))
            self.txt_desc.setWordWrapMode(True)

            self.btn_prev = QPushButton("Prev")
            self.btn_next = QPushButton("Next")
            self.btn_prev.clicked.connect(self.prev_title)
            self.btn_next.clicked.connect(self.next_title)

            self.btn_save = QPushButton("Save List...")
            self.btn_save.clicked.connect(self.save_file)
            self.btn_chart = QPushButton("Show Genre Chart")
            self.btn_chart.clicked.connect(self.show_chart)

            # Layout
            lay_main = QVBoxLayout()
            lay_main.addWidget(QLabel("Select Genre:"))
            lay_main.addWidget(self.genre_combo)
            lay_main.addSpacing(8)
            lay_main.addWidget(self.lbl_title)
            info_row = QHBoxLayout()
            info_row.addWidget(self.lbl_rating)
            info_row.addWidget(self.lbl_year)
            info_row.addWidget(self.lbl_duration)
            info_row.addStretch()
            lay_main.addLayout(info_row)
            lay_main.addWidget(self.txt_desc)
            nav_row = QHBoxLayout()
            nav_row.addWidget(self.btn_prev)
            nav_row.addWidget(self.btn_next)
            nav_row.addWidget(self.lbl_idx)
            nav_row.addStretch()
            nav_row.addWidget(self.btn_save)
            nav_row.addWidget(self.btn_chart)
            lay_main.addLayout(nav_row)

            central = QWidget()
            central.setLayout(lay_main)
            self.setCentralWidget(central)

            self.change_genre(self.genre)
            self.update_display()

        def get_field(self, rec, key):
            val = rec.get(key, "")
            return "" if val is None else str(val)

        def change_genre(self, genre):
            genre = genre.strip()
            self.genre = genre
            df = self.df_exploded[self.df_exploded['listed_in_exploded'] == genre]
            self.titles = df.drop_duplicates(subset=['title']).to_dict('records')
            self.idx = 0
            self.update_display()

        def update_display(self):
            if not self.titles:
                self.lbl_title.setText("No titles found.")
                self.lbl_rating.setText("")
                self.lbl_year.setText("")
                self.lbl_duration.setText("")
                self.txt_desc.setText("")
                self.lbl_idx.setText("")
                return
            rec = self.titles[self.idx]
            self.lbl_title.setText(self.get_field(rec, 'title'))
            self.lbl_rating.setText(f"Rating: {self.get_field(rec, 'rating')}")
            self.lbl_year.setText(f"Year: {self.get_field(rec, 'release_year')}")
            self.lbl_duration.setText(f"Duration: {self.get_field(rec, 'duration')}")
            self.txt_desc.setText(self.get_field(rec, 'description'))
            self.lbl_idx.setText(f"{self.idx + 1} / {len(self.titles)}")

        def prev_title(self):
            if self.idx > 0:
                self.idx -= 1
                self.update_display()

        def next_title(self):
            if self.idx < len(self.titles) - 1:
                self.idx += 1
                self.update_display()

        def keyPressEvent(self, event):
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_A):
                self.prev_title()
            elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_D):
                self.next_title()
            else:
                super().keyPressEvent(event)

        def save_file(self):
            if not self.titles:
                QMessageBox.information(self, "No Data", "No titles to save.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Titles", f"{self.genre}_titles.csv", "CSV Files (*.csv);;All Files (*)")
            if path:
                import pandas as pd
                pd.DataFrame(self.titles).to_csv(path, index=False)
                QMessageBox.information(self, "Saved", f"Saved {len(self.titles)} titles to {path}")

        def show_chart(self):
            if not self.chart_path:
                QMessageBox.warning(self, "No Chart", "No chart image available.")
                return
            chart_win = QWidget()
            chart_win.setWindowTitle("Top Genres Chart")
            vbox = QVBoxLayout()
            pixmap = QPixmap(self.chart_path)
            if pixmap.isNull():
                lbl = QLabel("Could not load chart image.")
            else:
                lbl = QLabel()
                lbl.setPixmap(pixmap.scaledToWidth(700, Qt.TransformationMode.SmoothTransformation))
            vbox.addWidget(lbl)
            chart_win.setLayout(vbox)
            chart_win.setMinimumWidth(720)
            chart_win.show()
            # Keep open until closed
            chart_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.chart_win = chart_win # ref

    # ---- App launch ----
    app = QApplication(sys.argv)
    mw = MainWindow(df_exploded, clusters_df, available_genres, chart_path, initial_genre)
    mw.show()
    app.exec()