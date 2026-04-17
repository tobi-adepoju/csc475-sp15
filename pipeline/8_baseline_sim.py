import os
import csv
import numpy as np
from typing import List, Dict, Tuple

CSV_FILE = "processed_dataset.csv"
FEATURE_DIR = "features"


def feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(FEATURE_DIR, f"{stem}_chroma.npy")


def load_metadata(csv_file: str = CSV_FILE) -> List[Dict]:
    with open(csv_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_chroma(file_name: str) -> np.ndarray | None:
    path = feature_path(file_name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def aggregate_chroma(chroma: np.ndarray) -> np.ndarray:
    vec = np.mean(chroma, axis=1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))


def build_feature_index(rows: List[Dict]) -> List[Dict]:
    indexed = []

    for row in rows:
        file_name = row["file_name"].strip()
        chroma = load_chroma(file_name)

        if chroma is None:
            print(f"[SKIP] Missing feature for {file_name}")
            continue

        vec = aggregate_chroma(chroma)

        indexed.append({
            "song_id": row["song_id"].strip(),
            "title": row["title"].strip(),
            "artist": row["artist"].strip(),
            "file_name": file_name,
            "song_version": row["song_version"].strip(),
            "vec": vec
        })

    return indexed


def rank_similar_songs(query_idx: int, indexed: List[Dict]) -> List[Tuple[float, Dict]]:
    query = indexed[query_idx]
    results = []

    for i, candidate in enumerate(indexed):
        if i == query_idx:
            continue

        score = cosine_similarity(query["vec"], candidate["vec"])
        results.append((score, candidate))

    results.sort(key=lambda x: x[0], reverse=True)
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
            f"song_id={item['song_id']} | score={score:.4f}"
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