"""
EVALUATION MODE (synthetic test suite):
    python cover_song_system.py --eval

FULL MODE (requires WAV files + .npy features):
    python cover_song_system.py
"""

import os
import csv
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cosine as cosine_dist
import librosa

# paths double check
CSV_FILE    = os.path.join("final-pipeline", "processed_dataset.csv")
WAV_DIR     = os.path.join("final-pipeline", "converted_to_22k_wav", "downloads")
FEATURE_DIR = os.path.join("final-pipeline", "features")

SR         = 22050
HOP_LENGTH = 512
N_CHROMA   = 12

# helper functions for loading audio and chroma features (from final-pipeline scripts)
def wav_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(WAV_DIR, f"{stem}_22k.wav")


def load_audio(file_name: str) -> np.ndarray:
    path = wav_path(file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"WAV not found: {path}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


def extract_chroma(y: np.ndarray) -> np.ndarray:
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    ch = uniform_filter1d(ch, size=11, axis=1)
    norms = np.linalg.norm(ch, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (ch / norms).astype(np.float32)


def compute_beat_chroma(y: np.ndarray) -> tuple[np.ndarray, float]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo)[0])
    chroma = extract_chroma(y)
    beat_ch = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    norms = np.linalg.norm(beat_ch, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (beat_ch / norms).astype(np.float32), tempo


def load_or_compute_beat_chroma(file_name: str) -> np.ndarray:
    stem = os.path.splitext(file_name)[0]
    npy  = os.path.join(FEATURE_DIR, f"{stem}_beat_chroma.npy")
    if os.path.exists(npy):
        return np.load(npy)
    bc, _ = compute_beat_chroma(load_audio(file_name))
    return bc


def load_dataset() -> list[dict]:
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


#
# PI1 — UNIFIED QUERY PIPELINE
#

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.T, b.T         
    n, m = len(a), len(b)
    INF  = float("inf")

    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cosine_dist(a[i - 1], b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return D[n, m] / (n + m)


def dtw_with_transposition(query_ch: np.ndarray, candidate: np.ndarray,
                            shifts: range = range(12)) -> float:
    best = float("inf")
    for shift in shifts:
        shifted = np.roll(candidate, shift, axis=0)
        cost    = dtw_distance(query_ch, shifted)
        if cost < best:
            best = cost
    return best


def mean_chroma_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    return float(1.0 - cosine_dist(ma, mb))


def query(query_file: str,
          dataset: list[dict],
          method: str = "dtw",
          top_k: int = 5) -> list[dict]:
    print(f"\n[query] loading: {query_file}")
    q_chroma = load_or_compute_beat_chroma(query_file)

    results = []
    for row in dataset:
        fn = row["file_name"].strip()
        if fn == query_file:
            continue
        try:
            c_chroma = load_or_compute_beat_chroma(fn)
        except FileNotFoundError:
            continue

        if method == "dtw":
            cost  = dtw_with_transposition(q_chroma, c_chroma)
            score = float(1.0 / (1.0 + cost))
        else:
            score = mean_chroma_similarity(q_chroma, c_chroma)

        results.append({
            "file_name"    : fn,
            "artist"       : row["artist"].strip(),
            "title"        : row["title"].strip(),
            "song_version" : row.get("song_version", "").strip(),
            "song_id"      : row.get("song_id", "").strip(),
            "score"        : score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = attach_confidence(results)
    return results[:top_k]


#
# PI5 — CONFIDENCE ESTIMATION
#

def attach_confidence(ranked: list[dict]) -> list[dict]:
    if not ranked:
        return ranked

    scores  = np.array([r["score"] for r in ranked], dtype=float)
    s_min   = scores.min()
    s_max   = scores.max()
    spread  = s_max - s_min if s_max > s_min else 1.0

    # normalise so top result = 1.0, worst result = 0.0
    norm_scores = (scores - s_min) / spread

    top    = norm_scores[0]
    second = norm_scores[1] if len(norm_scores) > 1 else 0.0
    gap    = float(top - second)  

    for i, r in enumerate(ranked):
        rank_penalty = 1.0 / (1.0 + i * 0.5)
        conf = float(np.clip(gap * rank_penalty * 2.5, 0.0, 1.0))
        r["confidence"] = round(conf, 3)
        if conf >= 0.35:
            r["confidence_label"] = "high"
        elif conf >= 0.15:
            r["confidence_label"] = "medium"
        else:
            r["confidence_label"] = "low"

    return ranked

# 
# PI2 — BASIC USER INTERFACE (retro MP3 player terminal display)
# 

CONF_COLOURS = {
    "high"   : "\033[92m",   # green
    "medium" : "\033[93m",   # yellow
    "low"    : "\033[91m",   # red
}
RESET = "\033[0m"


def display_results(query_file: str, results: list[dict]) -> None:
    width  = 62
    border = "═" * width

    print(f"\n  ╔{border}╗")
    print(f"  ║{'  🎵  COVER SONG DETECTOR  🎵':^{width}}║")
    print(f"  ╠{border}╣")
    print(f"  ║  QUERY : {query_file:<{width - 11}}║")
    print(f"  ╠{border}╣")
    print(f"  ║  {'#':<3} {'ARTIST — TITLE':<30} {'SCORE':>6}  {'CONF':<8}║")
    print(f"  ║  {'─' * (width - 4)}║")

    for i, r in enumerate(results, 1):
        label    = f"{r['artist'][:18]} — {r['title'][:10]}"
        colour   = CONF_COLOURS.get(r["confidence_label"], "")
        conf_str = f"{colour}{r['confidence_label']:>6}{RESET}"
        print(f"  ║  {i:<3} {label:<30} {r['score']:>6.3f}  {conf_str:<8}║")

    print(f"  ╚{border}╝\n")


# 
# PI3 — EVALUATION FRAMEWORK
# 

def top_k_accuracy(ranked: list[dict], true_song_id: str, k: int) -> int:
    return int(true_song_id in [r["song_id"] for r in ranked[:k]])


def reciprocal_rank(ranked: list[dict], true_song_id: str) -> float:
    for i, r in enumerate(ranked, 1):
        if r["song_id"] == true_song_id:
            return 1.0 / i
    return 0.0


def average_precision(ranked: list[dict], true_song_id: str) -> float:
    hits, running_sum = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r["song_id"] == true_song_id:
            hits += 1
            running_sum += hits / i
    total_relevant = sum(1 for r in ranked if r["song_id"] == true_song_id)
    return running_sum / total_relevant if total_relevant > 0 else 0.0


def evaluate(dataset: list[dict],
             method: str = "dtw",
             top_k_values: list[int] = [1, 3, 5]) -> dict:
    """PI3 — full evaluation: Top-K accuracy, MRR, MAP over all tracks."""
    all_rr, all_ap = [], []
    topk_hits = {k: [] for k in top_k_values}

    for row in dataset:
        fn      = row["file_name"].strip()
        song_id = row.get("song_id", "").strip()
        try:
            results = query(fn, dataset, method=method, top_k=max(top_k_values))
        except FileNotFoundError:
            continue

        excl = [r for r in results if r["file_name"] != fn]
        all_rr.append(reciprocal_rank(excl, song_id))
        all_ap.append(average_precision(excl, song_id))
        for k in top_k_values:
            topk_hits[k].append(top_k_accuracy(excl, song_id, k))

    metrics = {
        "MRR": float(np.mean(all_rr)) if all_rr else 0.0,
        "MAP": float(np.mean(all_ap)) if all_ap else 0.0,
    }
    for k in top_k_values:
        metrics[f"Top-{k} Acc"] = float(np.mean(topk_hits[k])) if topk_hits[k] else 0.0
    return metrics


def print_metrics(metrics: dict, method: str) -> None:
    print(f"\n  ┌── Evaluation Results ({method.upper()}) ──────────────┐")
    for name, val in metrics.items():
        print(f"  │  {name:<15} {val:.4f}                    │")
    print(f"  └──────────────────────────────────────────────┘\n")

#
# PI4 — EXPERIMENTAL PROTOCOL
#

def run_experiment(dataset: list[dict]) -> None:
    conditions = {
        "All tracks"         : dataset,
        "Originals as query" : [r for r in dataset
                                 if r.get("song_version", "").lower() == "original"],
        "Covers as query"    : [r for r in dataset
                                 if r.get("song_version", "").lower() != "original"],
    }

    print("\n" + "═" * 64)
    print("  EXPERIMENTAL PROTOCOL — Baseline vs DTW")
    print("═" * 64)

    for cname, subset in conditions.items():
        if not subset:
            print(f"\n  [SKIP] {cname} — no tracks matched")
            continue
        print(f"\n  Condition: {cname}  ({len(subset)} tracks)")
        for method in ["baseline", "dtw"]:
            metrics = evaluate(subset, method=method)
            print(f"    [{method.upper():<8}]  "
                  + "  ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))

#
# SYNTHETIC EVALUATION — controlled test data for all 5 PIs
#

def run_demo() -> None:
    """
    Run with: python cover_song_system.py --eval
    """
    print("\n" + "═" * 64)
    print("  COVER SONG DETECTOR — SYNTHETIC EVALUATION   ")
    print("═" * 64)

    np.random.seed(42)

    def normalize(ch: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(ch, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (ch / norms).astype(np.float32)

    def make_cover(base: np.ndarray, noise: float = 0.05, shift: int = 0,
                   tempo_stretch: float = 1.0, drop_section: bool = False,
                   blur: bool = False) -> np.ndarray:
        """
        Simulate a realistic cover with configurable imperfections:
          noise         — random pitch noise (simulates live performance/mic bleed)
          shift         — pitch class roll (key transposition)
          tempo_stretch — resample time axis (faster/slower tempo)
          drop_section  — randomly remove a middle chunk (different song structure)
          blur          — bleed energy into neighbouring pitch classes (instrument bleed)
        """
        ch = base.copy()

        # key transposition
        ch = np.roll(ch, shift, axis=0)

        # harmonic blur 
        if blur:
            blurred = ch.copy()
            for i in range(12):
                blurred[i] += 0.3 * ch[(i - 1) % 12] + 0.3 * ch[(i + 1) % 12]
            ch = blurred

        # add noise
        ch = ch + np.random.randn(*ch.shape) * noise

        # tempo stretch 
        if tempo_stretch != 1.0:
            old_len = ch.shape[1]
            new_len = max(10, int(old_len * tempo_stretch))
            indices = np.linspace(0, old_len - 1, new_len).astype(int)
            ch = ch[:, indices]

        # structural drop 
        if drop_section and ch.shape[1] > 30:
            t      = ch.shape[1]
            start  = np.random.randint(t // 4, t // 2)
            length = np.random.randint(t // 6, t // 4)
            ch = np.concatenate([ch[:, :start], ch[:, start + length:]], axis=1)

        return normalize(ch)

    songs = [
        {"song_id": "001", "title": "Halo",       "artist": "Beyonce"},
        {"song_id": "002", "title": "Jolene",      "artist": "Dolly Parton"},
        {"song_id": "003", "title": "Creep",       "artist": "Radiohead"},
        {"song_id": "004", "title": "Hallelujah",  "artist": "Leonard Cohen"},
        {"song_id": "005", "title": "Mad World",   "artist": "Tears for Fears"},
    ]

    fake_dataset = []
    fake_chromas = {}

    cover_configs = [
        # easy cover: small noise, slight key shift
        {"noise": 0.08, "shift": 2,  "tempo_stretch": 1.0,  "drop_section": False, "blur": False},
        # medium cover: more noise, key shift, slightly faster tempo
        {"noise": 0.18, "shift": 5,  "tempo_stretch": 1.15, "drop_section": False, "blur": True},
        # hard cover: heavy noise, big key shift, different structure, slower tempo
        {"noise": 0.30, "shift": 7,  "tempo_stretch": 0.85, "drop_section": True,  "blur": True},
    ]

    for song in songs:
        base  = np.random.rand(12, 80).astype(np.float32)
        base  = normalize(base)

        entries = [{"version": "original", "chroma": base, "artist": song["artist"]}]
        for i, cfg in enumerate(cover_configs, 1):
            entries.append({
                "version" : f"cover_{i}",
                "chroma"  : make_cover(base, **cfg),
                "artist"  : f"Cover Artist {chr(64 + i)}",
            })

        for e in entries:
            fn = f"{song['song_id']}_{e['version']}.wav"
            fake_dataset.append({
                "file_name"    : fn,
                "artist"       : e["artist"],
                "title"        : song["title"],
                "song_version" : e["version"],
                "song_id"      : song["song_id"],
            })
            fake_chromas[fn] = e["chroma"]


    def fake_query(query_file: str, subset: list[dict] = None,
                   method: str = "dtw", top_k: int = 5) -> list[dict]:
        pool = subset if subset is not None else fake_dataset
        q_ch = fake_chromas[query_file]
        results = []
        for row in pool:
            fn = row["file_name"]
            if fn == query_file:
                continue
            c_ch = fake_chromas[fn]
            if method == "dtw":
                cost  = dtw_with_transposition(q_ch, c_ch)
                score = float(1.0 / (1.0 + cost))
            else:
                score = mean_chroma_similarity(q_ch, c_ch)
            results.append({**row, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return attach_confidence(results)[:top_k]

    def fake_evaluate(subset: list[dict], method: str = "dtw") -> dict:
        all_rr, all_ap = [], []
        topk = {1: [], 3: [], 5: []}
        for row in subset:
            fn      = row["file_name"]
            song_id = row["song_id"]
            results = fake_query(fn, method=method, top_k=5)
            excl    = [r for r in results if r["file_name"] != fn]
            all_rr.append(reciprocal_rank(excl, song_id))
            all_ap.append(average_precision(excl, song_id))
            for k in [1, 3, 5]:
                topk[k].append(top_k_accuracy(excl, song_id, k))
        return {
            "MRR": float(np.mean(all_rr)),
            "MAP": float(np.mean(all_ap)),
            **{f"Top-{k} Acc": float(np.mean(topk[k])) for k in [1, 3, 5]},
        }

    print("\n── PI1: Unified Query Pipeline ─────────────────────────────")
    demo_file    = "001_original.wav"
    results_dtw  = fake_query(demo_file, method="dtw")
    results_base = fake_query(demo_file, method="baseline")
    print(f"  Query file  : {demo_file}")
    print(f"  Top result  : {results_dtw[0]['artist']} — {results_dtw[0]['title']} "
          f"(score: {results_dtw[0]['score']:.3f})")

    print("\n── PI2: Retro MP3 Player UI ────────────────────────────────")
    print("  [DTW]")
    display_results(demo_file, results_dtw)
    print("  [Baseline]")
    display_results(demo_file, results_base)

    print("── PI3: Evaluation Framework ───────────────────────────────")
    for method in ["baseline", "dtw"]:
        metrics = fake_evaluate(fake_dataset, method=method)
        print_metrics(metrics, method)

    print("── PI4: Experimental Protocol ──────────────────────────────")

    # generate additional synthetic variants for richer test conditions
    noisy_dataset   = []   # simulates live/imperfect recordings with high noise
    shifted_dataset = []   # simulates covers in heavily transposed keys (shift=6, tritone)
    sparse_dataset  = []   # simulates sparse/minimalist arrangements (short sequences)
    fake_chromas_noisy   = {}
    fake_chromas_shifted = {}
    fake_chromas_sparse  = {}

    for song in songs:
        base  = np.random.rand(12, 80).astype(np.float32)
        norms = np.linalg.norm(base, axis=0, keepdims=True)
        base  = base / np.where(norms == 0, 1.0, norms)

        for variant, store_data, store_chroma in [
            ("noisy",   noisy_dataset,   fake_chromas_noisy),
            ("shifted", shifted_dataset, fake_chromas_shifted),
            ("sparse",  sparse_dataset,  fake_chromas_sparse),
        ]:
            if variant == "noisy":
                entries = [
                    {"version": "original", "chroma": base,                                                              "artist": song["artist"]},
                    {"version": "cover_1",  "chroma": make_cover(base, noise=0.35, shift=2,  blur=True),                 "artist": "Cover Artist A"},
                    {"version": "cover_2",  "chroma": make_cover(base, noise=0.45, shift=3,  drop_section=True),         "artist": "Cover Artist B"},
                ]
            elif variant == "shifted":
                entries = [
                    {"version": "original", "chroma": base,                                                              "artist": song["artist"]},
                    {"version": "cover_1",  "chroma": make_cover(base, noise=0.08, shift=6,  blur=True),                 "artist": "Cover Artist A"},
                    {"version": "cover_2",  "chroma": make_cover(base, noise=0.10, shift=9,  tempo_stretch=1.2),         "artist": "Cover Artist B"},
                ]
            else: 
                short_base = base[:, :30]
                entries = [
                    {"version": "original", "chroma": short_base,                                                        "artist": song["artist"]},
                    {"version": "cover_1",  "chroma": make_cover(short_base, noise=0.08, shift=2),                       "artist": "Cover Artist A"},
                    {"version": "cover_2",  "chroma": make_cover(short_base, noise=0.12, shift=4, tempo_stretch=0.8),    "artist": "Cover Artist B"},
                ]

            for e in entries:
                fn = f"{song['song_id']}_{variant}_{e['version']}.wav"
                store_data.append({
                    "file_name": fn, "artist": e["artist"],
                    "title": song["title"], "song_version": e["version"],
                    "song_id": song["song_id"],
                })
                store_chroma[fn] = e["chroma"]

    def make_fake_query_fn(chroma_store, pool):
        def _fq(query_file, method="dtw", top_k=5):
            q_ch = chroma_store[query_file]
            results = []
            for row in pool:
                fn = row["file_name"]
                if fn == query_file:
                    continue
                c_ch = chroma_store[fn]
                if method == "dtw":
                    cost  = dtw_with_transposition(q_ch, c_ch)
                    score = float(1.0 / (1.0 + cost))
                else:
                    score = mean_chroma_similarity(q_ch, c_ch)
                results.append({**row, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return attach_confidence(results)[:top_k]
        return _fq

    def make_fake_eval_fn(chroma_store, pool):
        fq = make_fake_query_fn(chroma_store, pool)
        def _fe(subset, method="dtw"):
            all_rr, all_ap = [], []
            topk = {1: [], 3: [], 5: []}
            for row in subset:
                fn      = row["file_name"]
                song_id = row["song_id"]
                results = fq(fn, method=method, top_k=5)
                excl    = [r for r in results if r["file_name"] != fn]
                all_rr.append(reciprocal_rank(excl, song_id))
                all_ap.append(average_precision(excl, song_id))
                for k in [1, 3, 5]:
                    topk[k].append(top_k_accuracy(excl, song_id, k))
            return {
                "MRR": float(np.mean(all_rr)),
                "MAP": float(np.mean(all_ap)),
                **{f"Top-{k} Acc": float(np.mean(topk[k])) for k in [1, 3, 5]},
            }
        return _fe

    conditions = {
        # Measures overall system performance across the full dataset — the baseline comparison.
        "All tracks" : (fake_dataset, fake_chromas),

        # Tests whether the system finds covers when querying from a clean studio original —
        # the easiest condition since originals have minimal noise.
        "Originals as query" : (
            [r for r in fake_dataset if r["song_version"] == "original"], fake_chromas),

        # Tests whether the system finds originals when querying from a cover —
        # harder since covers introduce key and tempo variation.
        "Covers as query" : (
            [r for r in fake_dataset if r["song_version"] != "original"], fake_chromas),

        # Simulates degraded or live recordings with significant background noise —
        # tests system robustness when audio quality is poor.
        "High-noise recordings" : (noisy_dataset, fake_chromas_noisy),

        # Simulates covers performed in a heavily transposed key (tritone shift, 6 semitones) —
        # tests whether transposition search correctly recovers harmonic similarity.
        "Heavy key transposition (tritone)" : (shifted_dataset, fake_chromas_shifted),

        # Simulates short or sparse arrangements (e.g. solo acoustic, intro-only clips) —
        # tests whether DTW handles short sequences without degrading alignment quality.
        "Short/sparse arrangements" : (sparse_dataset, fake_chromas_sparse),

        # Isolates cover-to-cover retrieval, removing originals entirely —
        # tests whether the system can match two different covers of the same song.
        "Cover-to-cover only" : (
            [r for r in fake_dataset if r["song_version"] != "original"], fake_chromas),
    }

    print("\n" + "═" * 64)
    print("  Baseline vs DTW across test conditions")
    print("═" * 64)

    for cname, (subset, chroma_store) in conditions.items():
        fe = make_fake_eval_fn(chroma_store, subset)
        print(f"\n  Condition : {cname}  ({len(subset)} tracks)")
        for method in ["baseline", "dtw"]:
            m = fe(subset, method=method)
            print(f"    [{method.upper():<8}]  "
                  + "  ".join(f"{k}: {v:.3f}" for k, v in m.items()))

    # PI5 
    print("\n── PI5: Confidence Estimation ──────────────────────────────")
    print("  Confidence scores for demo query (DTW):\n")
    for r in results_dtw:
        bar_len = int(r["confidence"] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        colour  = CONF_COLOURS.get(r["confidence_label"], "")
        print(f"  {r['artist'][:20]:<20} [{bar}] "
              f"{colour}{r['confidence']:.3f} ({r['confidence_label']}){RESET}")

    print("\n  ✓ Evaluation suite complete.\n")

#
# MAIN
#

def main():
    if "--eval" in sys.argv or "--demo" in sys.argv:
        run_demo()
        return

    # full mode: requires all WAVs + .npy features
    dataset = load_dataset()
    print(f"loaded {len(dataset)} tracks\n")

    first_file = dataset[0]["file_name"].strip()
    results    = query(first_file, dataset, method="dtw", top_k=5)
    display_results(first_file, results)

    print("running full evaluation...\n")
    for method in ["baseline", "dtw"]:
        metrics = evaluate(dataset, method=method)
        print_metrics(metrics, method)

    run_experiment(dataset)


if __name__ == "__main__":
    main()
