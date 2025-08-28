import os
import pandas as pd

def load_movielens(path: str):
    """
    Load MovieLens CSVs from `path` (directory that contains `ratings.csv`, `movies.csv`, `tags.csv`).
    Returns (ratings_df, movies_df, tags_df_or_None).
    """
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(path, "movies.csv"))
    tags_path = os.path.join(path, "tags.csv")
    tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else None
    return ratings, movies, tags

def ensure_ids_as_int(df, col):
    if df[col].dtype != "int64" and df[col].dtype != "int32":
        df[col] = df[col].astype(int)
    return df
