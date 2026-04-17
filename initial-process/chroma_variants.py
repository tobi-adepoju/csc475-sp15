import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cosine as cosine_dist
 


CSV_FILE    = "processed_dataset.csv"
WAV_DIR     = "converted_to_22k_wav/downloads"
PLOT_DIR    = "plots"
SR          = 22050
HOP_LENGTH  = 512
N_CHROMA    = 12
PITCH_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
FONT_SIZE   = 14

'''
compares STFT, CQT, and CENS chroma variants for every track.
 
for each track, extracts all three variants, prints a quantitative
comparison table, and saves side-by-side chromagram plots so differences
in pitch resolution and temporal smoothness are visible.
'''

# requires: WAV files in converted_to_22k_wav/ (run 3_convert_to_22k_wave.sh first)


os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": FONT_SIZE})
 
 
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
 
 
def extract_chroma(y: np.ndarray, variant: str) -> np.ndarray:
    """(12, T) L2-normalized chroma for a given variant."""
    if variant == "stft":
        ch = librosa.feature.chroma_stft(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    elif variant == "cqt":
        ch = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    elif variant == "cens":
        ch = librosa.feature.chroma_cens(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    else:
        raise ValueError(variant)
 
    # smooth STFT and CQT, CENS handles smoothing internally
    if variant != "cens":
        ch = uniform_filter1d(ch, size=11, axis=1)
 
    norms = np.linalg.norm(ch, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (ch / norms).astype(np.float32)
 
 
def compare_variants(y: np.ndarray, file_name: str, artist: str, title: str) -> dict:
    """
    Extract all three variants, print stats table, save comparison plot.
    Returns dict of variant -> chroma array.
    """
    VARIANTS = ["stft", "cqt", "cens"]
    results  = {v: extract_chroma(y, v) for v in VARIANTS}
 
    # quantitative stats table 
    print(f"\n  {artist} — {title}")
    print(f"  {'Variant':<6}  {'Shape':>14}  {'Mean':>7}  {'Std':>7}  {'Sparsity%':>10}")
    print(f"  {'─' * 50}")
    for v, ch in results.items():
        sparsity = np.mean(ch < 0.05) * 100
        print(f"  {v.upper():<6}  {str(ch.shape):>14}  {ch.mean():>7.4f}  "
              f"{ch.std():>7.4f}  {sparsity:>9.1f}%")
 
    # pairwise cosine similarity between variants
    print(f"  Pairwise cosine similarity:")
    for i in range(len(VARIANTS)):
        for j in range(i + 1, len(VARIANTS)):
            a, b = results[VARIANTS[i]], results[VARIANTS[j]]
            T    = min(a.shape[1], b.shape[1])
            step = max(1, T // 200)
            sims = [1 - cosine_dist(a[:, t], b[:, t]) for t in range(0, T, step)
                     if np.any(a[:, t]) and np.any(b[:, t])]
            print(f"    {VARIANTS[i].upper()} vs {VARIANTS[j].upper()}: {np.mean(sims):.4f}")
 
    # side by side chromagram plot 
    descriptions = {
        "stft": "STFT: fast, lower pitch resolution",
        "cqt" : "CQT: better pitch resolution, recommended",
        "cens": "CENS: quantized + smoothed, most robust",
    }
 
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)
    fig.suptitle(f"Chroma Variant Comparison: {artist} / {title}",
                 fontsize=13, fontweight="bold")
 
    for ax, v in zip(axes, VARIANTS):
        librosa.display.specshow(
            results[v], y_axis="chroma", x_axis="time",
            sr=SR, hop_length=HOP_LENGTH, ax=ax, cmap="magma"
        )
        ax.set_yticks(range(12))
        ax.set_yticklabels(PITCH_NAMES, fontsize=9)
        ax.set_title(descriptions[v], fontsize=10)
        ax.set_ylabel("Pitch class")
        plt.colorbar(ax.collections[0], ax=ax)
 
    plt.tight_layout()
    stem     = os.path.splitext(file_name)[0]
    out_path = os.path.join(PLOT_DIR, f"{stem}_variant_comparison.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"  plot saved: {out_path}")
 
    return results
 
 
def plot_summary(all_stats: list[dict]) -> None:
    # bar chart of CQT vs CENS similarity per track, lower = representations differ more."""
    if not all_stats:
        print("  [SKIP] no data to plot summary")
        return
    labels = [s["label"] for s in all_stats]
    sims   = [s["cqt_vs_cens"] for s in all_stats]
 
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4))
    ax.bar(range(len(labels)), sims, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(np.mean(sims), color="orange", linestyle="--", linewidth=1.2,
               label=f"mean = {np.mean(sims):.3f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("CQT vs CENS Similarity per Track", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
 
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "variant_summary_cqt_vs_cens.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"\nsummary plot saved: {out_path}")
 
 
def main():
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
 
    print(f"comparing chroma variants for {len(rows)} tracks\n")
    
 
    all_stats = []
    for row in rows:
        file_name = row["file_name"].strip()
        try:
            y = load_audio(file_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue
 
        variants = compare_variants(y, file_name, row["artist"].strip(), row["title"].strip())
 
        # collect CQT vs CENS similarity for summary chart
        a, b = variants["cqt"], variants["cens"]
        T    = min(a.shape[1], b.shape[1])
        step = max(1, T // 200)
        sims = [1 - cosine_dist(a[:, t], b[:, t]) for t in range(0, T, step)]
        all_stats.append({
            "label"       : f"{row['artist'][:20]} ({row['song_version']})",
            "cqt_vs_cens" : float(np.mean(sims)),
        })
 
    if all_stats:
        plot_summary(all_stats)
 
 
if __name__ == "__main__":
    main()