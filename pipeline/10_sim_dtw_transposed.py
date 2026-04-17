import os
import numpy as np
import pandas as pd
import librosa
from scipy.spatial.distance import cosine

DATASET_CSV = "processed_dataset.csv"

CHROMA_DIR = "features"
BEAT_CHROMA_DIR = "features_beat"

USE_BEAT_FEATURES_FOR_SEQUENCE = False   

DTW_WEIGHT = 0.60
GLOBAL_WEIGHT = 0.40
SHORTLIST_SIZE = 15

def normalize_chroma_columns(C):
    norms = np.linalg.norm(C, axis=0, keepdims=True) + 1e-12
    return C / norms

def aggregate_chroma(chroma):
    chroma = normalize_chroma_columns(chroma)

    mean = np.mean(chroma, axis=1)
    std = np.std(chroma, axis=1)

    vec = np.concatenate([mean, std])
    vec /= np.linalg.norm(vec) + 1e-12
    return vec.astype(np.float32)

def load_chroma(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

def dtw_distance_transposed(X, Y):
    X = normalize_chroma_columns(X)
    Y = normalize_chroma_columns(Y)

    best_cost = np.inf
    best_shift = 0

    for shift in range(12):
        Y_shift = np.roll(Y, shift, axis=0)

        C = 1 - np.dot(X.T, Y_shift)

        D, wp = librosa.sequence.dtw(
            C=C,
            global_constraints=True,
            band_rad=0.25
        )
        
        raw_cost = D[-1, -1]
        path_len = len(wp)

        expected_len = (X.shape[1] + Y.shape[1]) / 2
        length_penalty = path_len / expected_len
        cost = raw_cost * length_penalty / path_len

        len_ratio = max(X.shape[1], Y.shape[1]) / min(X.shape[1], Y.shape[1])
        duration_penalty = np.log(len_ratio + 1e-12)
        cost *= (1 + 0.5 * duration_penalty)

        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    return best_cost, best_shift

df = pd.read_csv(DATASET_CSV)

tracks = []

for _, row in df.iterrows():
    base = os.path.splitext(row["file_name"])[0]

    chroma_path = os.path.join(CHROMA_DIR, base + "_chroma.npy")
    beat_path   = os.path.join(BEAT_CHROMA_DIR, base + "_beat_chroma.npy")

    chroma = load_chroma(chroma_path)
    beat   = load_chroma(beat_path)

    if chroma is None and beat is None:
        print(f"[SKIP] Missing features for {row['file_name']}")
        continue

    is_cover = row["song_version"] == "cover"

    global_vec = aggregate_chroma(chroma if chroma is not None else beat)

    tracks.append({
        "song_id": row["song_id"],
        "artist": row["artist"],
        "title": row["title"],
        "is_cover": is_cover,
        "chroma": chroma,
        "beat": beat,
        "global": global_vec
    })

print(f"\nLoaded {len(tracks)} tracks with usable features.")


def global_distance(a, b):
    return cosine(a, b)

def retrieve(query_index):
    query = tracks[query_index]
    print(f"\nComputing hybrid retrieval for query: "
          f"{query['artist']} — {query['title']} "
          f"[sequence_feature={'beat' if USE_BEAT_FEATURES_FOR_SEQUENCE else 'regular'}]")

    global_scores = []
    for i, track in enumerate(tracks):
        if i == query_index:
            continue
        d = global_distance(query["global"], track["global"])
        global_scores.append((i, d))

    global_scores.sort(key=lambda x: x[1])
    shortlist = global_scores[:SHORTLIST_SIZE]

    print(f"Shortlist size after global similarity pruning: {len(shortlist)}")

    results = []

    for idx, global_dist in shortlist:
        candidate = tracks[idx]

        X = query["beat"] if USE_BEAT_FEATURES_FOR_SEQUENCE else query["chroma"]
        Y = candidate["beat"] if USE_BEAT_FEATURES_FOR_SEQUENCE else candidate["chroma"]

        dtw_cost, shift = dtw_distance_transposed(X, Y)

        final_score = DTW_WEIGHT * dtw_cost + GLOBAL_WEIGHT * global_dist

        print(f"[OK] {candidate['artist']} — {candidate['title']} | "
              f"final={final_score:.4f} | dtw={dtw_cost:.4f} | "
              f"global={global_dist:.4f} | dtw_shift={shift}")

        results.append((idx, final_score, dtw_cost, global_dist, shift))

    results.sort(key=lambda x: x[1])
    return results[:10]


QUERY_INDEX = 0  

top_matches = retrieve(QUERY_INDEX)

print("\nQUERY")
print(f"{tracks[QUERY_INDEX]['artist']} — {tracks[QUERY_INDEX]['title']}")
print(f"song_id = {tracks[QUERY_INDEX]['song_id']}")

print("\nTOP 10 MATCHES")
for rank, (idx, final, dtw, glob, shift) in enumerate(top_matches, 1):
    t = tracks[idx]
    tag = "cover" if t["is_cover"] else "original"
    print(f"{rank:2d}. [{tag}] {t['artist']} — {t['title']} "
          f"| final={final:.4f} | dtw={dtw:.4f} | global={glob:.4f} | shift={shift}")