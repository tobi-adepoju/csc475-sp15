import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import librosa.sequence
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional

CSV_FILE = "processed_dataset.csv"
REGULAR_FEATURE_DIR = "features"
BEAT_FEATURE_DIR = "features_beat"
PLOT_DIR = "plots"

SHORTLIST_K = 15
DTW_WEIGHT = 0.70
GLOBAL_WEIGHT = 0.30
TRIM_START = 0.20
TRIM_END = 0.80
DOWNSAMPLE_FACTOR = 1
LENGTH_PENALTY_STRENGTH = 0.25
USE_BEAT_FEATURES_FOR_HYBRID = True

os.makedirs(PLOT_DIR, exist_ok=True)

def load_metadata(csv_file: str = CSV_FILE) -> List[Dict]:
    with open(csv_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def regular_feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(REGULAR_FEATURE_DIR, f"{stem}_chroma.npy")


def beat_feature_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(BEAT_FEATURE_DIR, f"{stem}_beat_chroma.npy")


def load_npy_if_exists(path: str) -> Optional[np.ndarray]:
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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))

def dtw_distance(chroma1: np.ndarray, chroma2: np.ndarray, downsample_factor: int = 4) -> float:
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


def build_index(rows: List[Dict]) -> List[Dict]:
    indexed = []

    for row in rows:
        file_name = row["file_name"].strip()

        regular_chroma = load_npy_if_exists(regular_feature_path(file_name))
        beat_chroma = load_npy_if_exists(beat_feature_path(file_name))

        if regular_chroma is None and beat_chroma is None:
            print(f"[SKIP] Missing all features for {file_name}")
            continue

        try:
            regular_clean = clean_chroma(regular_chroma) if regular_chroma is not None else None
            beat_clean = clean_chroma(beat_chroma) if beat_chroma is not None else None

            global_source = regular_clean if regular_clean is not None else beat_clean
            seq_source = beat_clean if (USE_BEAT_FEATURES_FOR_HYBRID and beat_clean is not None) else regular_clean
            if seq_source is None:
                seq_source = beat_clean

            indexed.append({
                "song_id": row["song_id"].strip(),
                "title": row["title"].strip(),
                "artist": row["artist"].strip(),
                "file_name": file_name,
                "song_version": row["song_version"].strip(),
                "regular_chroma": regular_clean,
                "beat_chroma": beat_clean,
                "global_vec": aggregate_chroma(global_source),
            })
        except Exception as e:
            print(f"[SKIP] Bad feature for {file_name}: {e}")

    return indexed

def rank_baseline(query_idx: int, indexed: List[Dict]) -> List[Dict]:
    query = indexed[query_idx]
    results = []

    for i, candidate in enumerate(indexed):
        if i == query_idx:
            continue
        score = cosine_similarity(query["global_vec"], candidate["global_vec"])
        results.append({
            "candidate": candidate,
            "sort_score": score,
            "higher_is_better": True,
        })

    results.sort(key=lambda x: x["sort_score"], reverse=True)
    return results


def rank_dtw(query_idx: int, indexed: List[Dict]) -> List[Dict]:
    query = indexed[query_idx]
    results = []

    q_chroma = query["regular_chroma"]
    if q_chroma is None:
        return results

    for i, candidate in enumerate(indexed):
        if i == query_idx or candidate["regular_chroma"] is None:
            continue
        try:
            score = dtw_distance(q_chroma, candidate["regular_chroma"], downsample_factor=4)
            results.append({
                "candidate": candidate,
                "sort_score": score,
                "higher_is_better": False,
            })
        except Exception as e:
            print(f"[WARN] DTW failed for {candidate['file_name']}: {e}")

    results.sort(key=lambda x: x["sort_score"])
    return results


def rank_hybrid(query_idx: int, indexed: List[Dict]) -> List[Dict]:
    query = indexed[query_idx]
    coarse = []

    q_seq = query["beat_chroma"] if (USE_BEAT_FEATURES_FOR_HYBRID and query["beat_chroma"] is not None) else query["regular_chroma"]
    if q_seq is None:
        return []

    for i, candidate in enumerate(indexed):
        if i == query_idx:
            continue
        global_dist, _ = best_transposed_global_distance(query["global_vec"], candidate["global_vec"])
        coarse.append({
            "candidate_idx": i,
            "global_dist": global_dist,
        })

    coarse.sort(key=lambda x: x["global_dist"])
    shortlist = coarse[:max(1, min(SHORTLIST_K, len(coarse)))]

    results = []
    for item in shortlist:
        candidate = indexed[item["candidate_idx"]]
        c_seq = candidate["beat_chroma"] if (USE_BEAT_FEATURES_FOR_HYBRID and candidate["beat_chroma"] is not None) else candidate["regular_chroma"]
        if c_seq is None:
            continue

        try:
            dtw_score, shift = transposed_dtw_distance(q_seq, c_seq)
            final_score = (DTW_WEIGHT * dtw_score) + (GLOBAL_WEIGHT * item["global_dist"])
            results.append({
                "candidate": candidate,
                "sort_score": final_score,
                "higher_is_better": False,
                "dtw_score": dtw_score,
                "global_dist": item["global_dist"],
                "shift": shift,
            })
        except Exception as e:
            print(f"[WARN] Hybrid DTW failed for {candidate['file_name']}: {e}")

    results.sort(key=lambda x: x["sort_score"])
    return results


def reciprocal_rank(results: List[Dict], query_song_id: str) -> float:
    for rank, item in enumerate(results, start=1):
        if item["candidate"]["song_id"] == query_song_id:
            return 1.0 / rank
    return 0.0


def average_precision(results: List[Dict], query_song_id: str) -> float:
    hits = 0
    running_sum = 0.0
    total_relevant = sum(1 for item in results if item["candidate"]["song_id"] == query_song_id)

    if total_relevant == 0:
        return 0.0

    for rank, item in enumerate(results, start=1):
        if item["candidate"]["song_id"] == query_song_id:
            hits += 1
            running_sum += hits / rank

    return running_sum / total_relevant


def hit_at_k(results: List[Dict], query_song_id: str, k: int) -> int:
    return int(any(item["candidate"]["song_id"] == query_song_id for item in results[:k]))


def evaluate_method(indexed: List[Dict], method_name: str) -> Dict[str, float]:
    method_map = {
        "baseline": rank_baseline,
        "dtw": rank_dtw,
        "hybrid": rank_hybrid,
    }
    rank_fn = method_map[method_name]

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    rr_total = 0.0
    ap_total = 0.0
    n = 0

    for query_idx, query in enumerate(indexed):
        results = rank_fn(query_idx, indexed)
        if not results:
            continue

        qid = query["song_id"]
        top1_hits += hit_at_k(results, qid, 1)
        top3_hits += hit_at_k(results, qid, 3)
        top5_hits += hit_at_k(results, qid, 5)
        rr_total += reciprocal_rank(results, qid)
        ap_total += average_precision(results, qid)
        n += 1

    if n == 0:
        return {
            "Top-1": 0.0,
            "Top-3": 0.0,
            "Top-5": 0.0,
            "MRR": 0.0,
            "MAP": 0.0,
        }

    return {
        "Top-1": top1_hits / n,
        "Top-3": top3_hits / n,
        "Top-5": top5_hits / n,
        "MRR": rr_total / n,
        "MAP": ap_total / n,
    }


def plot_results(results_by_method: Dict[str, Dict[str, float]], out_path: str) -> None:
    metrics = ["Top-1", "Top-3", "Top-5", "MRR", "MAP"]
    methods = list(results_by_method.keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))

    for i, method in enumerate(methods):
        values = [results_by_method[method][m] for m in metrics]
        ax.bar(x + (i - 1) * width, values, width, label=method.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Cover Song Retrieval Methods")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_results_table(results_by_method: Dict[str, Dict[str, float]], out_path: str) -> None:
    metrics = ["Top-1", "Top-3", "Top-5", "MRR", "MAP"]
    methods = list(results_by_method.keys())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(metrics) + "\n")
        for method in methods:
            vals = [results_by_method[method][m] for m in metrics]
            f.write(method + "," + ",".join(f"{v:.4f}" for v in vals) + "\n")


def main() -> None:
    rows = load_metadata()
    indexed = build_index(rows)

    print(f"Loaded {len(indexed)} usable tracks.")

    if not indexed:
        print("No usable tracks found.")
        return

    results_by_method = {}
    for method in ["baseline", "dtw", "hybrid"]:
        print(f"\nEvaluating {method}...")
        results_by_method[method] = evaluate_method(indexed, method)
        for metric, value in results_by_method[method].items():
            print(f"  {metric}: {value:.4f}")

    fig_path = os.path.join(PLOT_DIR, "retrieval_method_comparison.png")
    csv_path = os.path.join(PLOT_DIR, "retrieval_method_comparison.csv")

    plot_results(results_by_method, fig_path)
    save_results_table(results_by_method, csv_path)

    print(f"\nSaved figure to: {fig_path}")
    print(f"Saved table to:  {csv_path}")


if __name__ == "__main__":
    main()