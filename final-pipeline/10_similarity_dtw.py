import os
import csv
import numpy as np
import librosa.sequence
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple

CSV_FILE = "processed_dataset.csv"
FEATURE_DIR = "features"


def feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(FEATURE_DIR, f"{stem}_chroma.npy")


def load_metadata(csv_file: str = CSV_FILE) -> List[Dict]:
    with open(csv_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_chroma(file_name: str) -> np.ndarray | None:
    path = feature_path(file_name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def clean_chroma(chroma: np.ndarray) -> np.ndarray:
    """
    Make chroma safe for distance calculations:
    - replace NaN/inf with 0
    - ensure float32
    - drop any completely invalid columns
    """
    chroma = np.asarray(chroma, dtype=np.float32)
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)

    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError(f"Expected chroma shape (12, T), got {chroma.shape}")

    keep = np.any(np.abs(chroma) > 1e-12, axis=0)
    if not np.any(keep):
        return np.zeros((12, 1), dtype=np.float32)

    chroma = chroma[:, keep]
    return chroma


def normalize_chroma_columns(chroma: np.ndarray) -> np.ndarray:
    """
    Normalize each time frame (column) to unit norm.
    Zero columns stay zero.
    """
    chroma = clean_chroma(chroma)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    chroma = chroma / norms
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    return chroma.astype(np.float32)


def downsample_chroma(chroma: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Reduce time resolution to speed up DTW.
    """
    factor = max(1, int(factor))
    return chroma[:, ::factor]


def dtw_distance(chroma1: np.ndarray, chroma2: np.ndarray, downsample_factor: int = 4) -> float:
    """
    Compute DTW cost using a precomputed cosine distance matrix.
    Lower = more similar.
    """
    chroma1 = normalize_chroma_columns(chroma1)
    chroma2 = normalize_chroma_columns(chroma2)

    chroma1 = downsample_chroma(chroma1, downsample_factor)
    chroma2 = downsample_chroma(chroma2, downsample_factor)

    X = chroma1.T
    Y = chroma2.T

    C = cdist(X, Y, metric="cosine")
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)

    C = np.clip(C, 0.0, None)

    D, wp = librosa.sequence.dtw(C=C)

    path_len = len(wp) if len(wp) > 0 else 1
    return float(D[-1, -1] / path_len)


def build_feature_index(rows: List[Dict]) -> List[Dict]:
    indexed = []

    for row in rows:
        file_name = row["file_name"].strip()
        chroma = load_chroma(file_name)

        if chroma is None:
            print(f"[SKIP] Missing feature for {file_name}")
            continue

        try:
            chroma = clean_chroma(chroma)
        except Exception as e:
            print(f"[SKIP] Bad feature for {file_name}: {e}")
            continue

        indexed.append({
            "song_id": row["song_id"].strip(),
            "title": row["title"].strip(),
            "artist": row["artist"].strip(),
            "file_name": file_name,
            "song_version": row["song_version"].strip(),
            "chroma": chroma
        })

    return indexed


def rank_similar_songs(query_idx: int, indexed: List[Dict]) -> List[Tuple[float, Dict]]:
    query = indexed[query_idx]
    results = []

    print(f"\nComputing DTW similarities for query: {query['artist']} — {query['title']} ({query['song_version']})")

    for i, candidate in enumerate(indexed):
        if i == query_idx:
            continue

        try:
            score = dtw_distance(query["chroma"], candidate["chroma"], downsample_factor=4)
            results.append((score, candidate))
            print(f"[OK] compared with {candidate['artist']} — {candidate['title']} | dtw={score:.4f}")
        except Exception as e:
            print(f"[ERROR] DTW failed for {candidate['file_name']}: {e}")

    results.sort(key=lambda x: x[0])
    return results


def print_top_matches(query_idx: int, indexed: List[Dict], top_k: int = 10) -> None:
    query = indexed[query_idx]
    results = rank_similar_songs(query_idx, indexed)

    print("\nQUERY")
    print(f"{query['artist']} — {query['title']} ({query['song_version']})")
    print(f"song_id = {query['song_id']}")

    print(f"\nTOP {top_k} MATCHES")
    for rank, (score, item) in enumerate(results[:top_k], start=1):
        correct = "✓" if item["song_id"] == query["song_id"] else " "
        print(
            f"{rank:2d}. [{correct}] "
            f"{item['artist']} — {item['title']} ({item['song_version']}) | "
            f"song_id={item['song_id']} | dtw_score={score:.4f}"
        )


def main():
    rows = load_metadata()
    indexed = build_feature_index(rows)

    print(f"Loaded {len(indexed)} tracks with chroma features.")

    if len(indexed) == 0:
        print("No features found.")
        return

    query_idx = 0
    print_top_matches(query_idx, indexed, top_k=10)


if __name__ == "__main__":
    main()