import os
import csv
import numpy as np
import librosa.sequence
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple

CSV_FILE = "processed_dataset.csv"
BEAT_FEATURE_DIR = "features_beat"
REGULAR_FEATURE_DIR = "features"

SHORTLIST_K = 15
DTW_WEIGHT = 0.70
GLOBAL_WEIGHT = 0.30
TRIM_START = 0.20
TRIM_END = 0.80
DOWNSAMPLE_FACTOR = 1
LENGTH_PENALTY_STRENGTH = 0.25
USE_BEAT_FEATURES_FOR_SEQUENCE = True


def beat_feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(BEAT_FEATURE_DIR, f"{stem}_beat_chroma.npy")


def regular_feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(REGULAR_FEATURE_DIR, f"{stem}_chroma.npy")


def load_metadata(csv_file: str = CSV_FILE) -> List[Dict]:
    with open(csv_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_npy_if_exists(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    return np.load(path)


def clean_chroma(chroma: np.ndarray) -> np.ndarray:
    chroma = np.asarray(chroma, dtype=np.float32)
    chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError(f"Expected chroma shape (12, T), got {chroma.shape}")
    keep = np.any(np.abs(chroma) > 1e-12, axis=0)
    if not np.any(keep):
        return np.zeros((12, 1), dtype=np.float32)
    return chroma[:, keep]


def normalize_chroma_columns(chroma: np.ndarray) -> np.ndarray:
    chroma = clean_chroma(chroma)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    chroma = chroma / norms
    return np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def aggregate_chroma(chroma: np.ndarray) -> np.ndarray:
    chroma = normalize_chroma_columns(chroma)
    vec = np.mean(chroma, axis=1)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec.astype(np.float32)


def trim_chroma(chroma: np.ndarray, start_frac: float = TRIM_START, end_frac: float = TRIM_END) -> np.ndarray:
    T = chroma.shape[1]
    start_idx = int(T * start_frac)
    end_idx = int(T * end_frac)
    start_idx = max(0, min(start_idx, T - 1))
    end_idx = max(start_idx + 1, min(end_idx, T))
    return chroma[:, start_idx:end_idx]


def downsample_chroma(chroma: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    factor = max(1, int(factor))
    return chroma[:, ::factor]


def best_transposed_global_distance(vec1: np.ndarray, vec2: np.ndarray, shifts=range(12)) -> Tuple[float, int]:
    best_dist = float("inf")
    best_shift = 0
    for shift in shifts:
        shifted = np.roll(vec2, shift=shift)
        sim = float(np.dot(vec1, shifted))
        dist = 1.0 - sim
        if dist < best_dist:
            best_dist = dist
            best_shift = shift
    return best_dist, best_shift


def dtw_distance_from_cost_matrix(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    X = chroma1.T
    Y = chroma2.T
    C = cdist(X, Y, metric="cosine")
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)
    C = np.clip(C, 0.0, None)
    D, wp = librosa.sequence.dtw(C=C)
    path_len = len(wp) if len(wp) > 0 else 1
    raw_score = float(D[-1, -1] / path_len)
    ratio = max(X.shape[0], Y.shape[0]) / max(1, min(X.shape[0], Y.shape[0]))
    penalty = np.log1p(ratio - 1.0)
    return raw_score + (LENGTH_PENALTY_STRENGTH * penalty)


def transposed_dtw_distance(chroma1: np.ndarray, chroma2: np.ndarray, shifts=range(12)) -> Tuple[float, int]:
    chroma1 = downsample_chroma(trim_chroma(normalize_chroma_columns(chroma1)))
    chroma2 = downsample_chroma(trim_chroma(normalize_chroma_columns(chroma2)))
    best_score = float("inf")
    best_shift = 0
    for shift in shifts:
        shifted = np.roll(chroma2, shift=shift, axis=0)
        score = dtw_distance_from_cost_matrix(chroma1, shifted)
        if score < best_score:
            best_score = score
            best_shift = shift
    return best_score, best_shift


def build_feature_index(rows: List[Dict]) -> List[Dict]:
    indexed = []
    for row in rows:
        file_name = row["file_name"].strip()
        beat_chroma = load_npy_if_exists(beat_feature_path(file_name))
        regular_chroma = load_npy_if_exists(regular_feature_path(file_name))
        if beat_chroma is None and regular_chroma is None:
            continue
        try:
            seq_chroma = beat_chroma if (USE_BEAT_FEATURES_FOR_SEQUENCE and beat_chroma is not None) else regular_chroma
            if seq_chroma is None:
                seq_chroma = beat_chroma
            seq_chroma = clean_chroma(seq_chroma)
            global_source = regular_chroma if regular_chroma is not None else seq_chroma
            global_vec = aggregate_chroma(global_source)
        except Exception:
            continue
        indexed.append({
            "song_id": row["song_id"].strip(),
            "title": row["title"].strip(),
            "artist": row["artist"].strip(),
            "song_version": row["song_version"].strip(),
            "seq_chroma": seq_chroma,
            "global_vec": global_vec,
        })
    return indexed


def rank_similar_songs(query_idx: int, indexed: List[Dict]) -> List[Dict]:
    query = indexed[query_idx]
    coarse = []
    for i, candidate in enumerate(indexed):
        if i == query_idx:
            continue
        global_dist, _ = best_transposed_global_distance(query["global_vec"], candidate["global_vec"])
        coarse.append({"candidate": candidate, "global_dist": global_dist})
    coarse.sort(key=lambda x: x["global_dist"])
    shortlist = coarse[: max(1, min(SHORTLIST_K, len(coarse)))]

    results = []
    for item in shortlist:
        candidate = item["candidate"]
        dtw_score, shift = transposed_dtw_distance(query["seq_chroma"], candidate["seq_chroma"])
        final_score = (DTW_WEIGHT * dtw_score) + (GLOBAL_WEIGHT * item["global_dist"])
        results.append({
            "final_score": final_score,
            "dtw_score": dtw_score,
            "shift": shift,
            "global_dist": item["global_dist"],
            "candidate": candidate,
        })
    results.sort(key=lambda x: x["final_score"])
    return results


def reciprocal_rank(results: List[Dict], query_song_id: str) -> float:
    for rank, item in enumerate(results, start=1):
        if item["candidate"]["song_id"] == query_song_id:
            return 1.0 / rank
    return 0.0


def hit_at_k(results: List[Dict], query_song_id: str, k: int) -> int:
    for item in results[:k]:
        if item["candidate"]["song_id"] == query_song_id:
            return 1
    return 0


def evaluate(indexed: List[Dict]) -> None:
    n = len(indexed)
    if n == 0:
        print("No indexed tracks found.")
        return

    top1_hits = top3_hits = top5_hits = 0
    mrr_total = 0.0

    print(f"\nEvaluating hybrid retrieval on {n} queries...\n")
    for query_idx, query in enumerate(indexed):
        results = rank_similar_songs(query_idx, indexed)
        rr = reciprocal_rank(results, query["song_id"])
        top1_hits += hit_at_k(results, query["song_id"], 1)
        top3_hits += hit_at_k(results, query["song_id"], 3)
        top5_hits += hit_at_k(results, query["song_id"], 5)
        mrr_total += rr

        first_correct_rank = int(round(1.0 / rr)) if rr > 0 else None
        print(
            f"[{query_idx + 1}/{n}] {query['artist']} — {query['title']} ({query['song_version']}) | "
            f"first_correct_rank={first_correct_rank}"
        )

    print("\nFINAL RESULTS")
    print(f"Number of queries: {n}")
    print(f"Top-1 Accuracy: {top1_hits / n:.4f}")
    print(f"Top-3 Accuracy: {top3_hits / n:.4f}")
    print(f"Top-5 Accuracy: {top5_hits / n:.4f}")
    print(f"MRR:            {mrr_total / n:.4f}")


def main() -> None:
    rows = load_metadata()
    indexed = build_feature_index(rows)
    print(f"Loaded {len(indexed)} tracks with usable features.")
    evaluate(indexed)


if __name__ == "__main__":
    main()
