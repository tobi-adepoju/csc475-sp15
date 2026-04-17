import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa.display
from collections import defaultdict

CSV_FILE    = "processed_dataset.csv"
FEATURE_DIR = "features"
PLOT_DIR    = "plots"
SR          = 22050
HOP_LENGTH  = 512
PITCH_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def load_chroma(file_name: str) -> np.ndarray | None:
    stem = os.path.splitext(file_name)[0]
    path = os.path.join(FEATURE_DIR, f"{stem}_chroma.npy")
    if not os.path.exists(path):
        print(f"  [SKIP] feature file not found: {path}")
        return None
    return np.load(path)

# plot chromagram heatmaps and 
def plot_track(chroma: np.ndarray, label: str, save_path: str) -> None:
    
    # create two plots
    # top: chromagram heatmap
    
    fig, (ax_chroma, ax_energy) = plt.subplots(
        2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [4, 1]})
    
    fig.suptitle(label, fontsize=12, fontweight="bold")
 
    librosa.display.specshow(
        chroma, y_axis="chroma", x_axis="time",
        sr=SR, hop_length=HOP_LENGTH, ax=ax_chroma, cmap="magma"
    )
    ax_chroma.set_yticks(range(12))
    ax_chroma.set_yticklabels(PITCH_NAMES, fontsize=9)
    ax_chroma.set_title("CQT Chromagram", fontsize=10)
    ax_chroma.set_ylabel("Pitch class")
    plt.colorbar(ax_chroma.collections[0], ax=ax_chroma, label="Intensity")
 
    # bottom: frame energy - quality check
    # mean energy plotted as line, low mean detected as silence
    # red dots mark silent frames

    energy    = chroma.sum(axis=0)
    threshold = energy.mean() * 0.1
    silent    = np.where(energy < threshold)[0]
    ax_energy.fill_between(range(len(energy)), energy, alpha=0.7, color="steelblue")
    # ax_energy.axhline(energy.mean(), color="orange", linewidth=1, linestyle="--",
    #                  label=f"mean={energy.mean():.2f}")
    if len(silent):
        ax_energy.scatter(silent, energy[silent], color="red", s=5, zorder=5,
                          label=f"{len(silent)} low-energy frames")
    ax_energy.set_xlabel("Frame")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Frame Energy — quality check", fontsize=9)
    # ax_energy.legend(fontsize=16)
 
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"  saved: {save_path}")

# plot originals and covers per song
def plot_group(group_rows: list[dict], save_path: str) -> None:
    # load chroma for each track in group, skip missing files
    loaded = [(r, load_chroma(r["file_name"])) for r in group_rows]
    loaded = [(r, ch) for r, ch in loaded if ch is not None]
    if len(loaded) < 2:
        return
    
    # stack one chromagram per versions to compare similarities
    # cover in same key will look identical, key shifted will be shifted up or down

    fig, axes = plt.subplots(len(loaded), 1, figsize=(15, 4 * len(loaded)))
    if len(loaded) == 1:
        axes = [axes]
    fig.suptitle(f"Cover Group: {group_rows[0]['title']}", fontsize=13, fontweight="bold")

    for ax, (row, chroma) in zip(axes, loaded):
        librosa.display.specshow(
            chroma, y_axis="chroma", x_axis="time", sr=SR, hop_length=HOP_LENGTH, ax=ax)
        
        ax.set_yticks(range(12))
        ax.set_yticklabels(PITCH_NAMES, fontsize=16)
        ax.set_title(f"[{row['song_version'].upper()}]  {row['artist']} — {row['title']}", fontsize=10)
        ax.set_ylabel("Pitch class")
        plt.colorbar(ax.collections[0], ax=ax, label="Intensity")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"  saved: {save_path}")

    
def main():
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
 
    print(f"visualizing {len(rows)} tracks\n")
 
    # chromagram + energy per track
    for row in rows:
        chroma = load_chroma(row["file_name"])
        if chroma is None:
            continue
        stem  = os.path.splitext(row["file_name"])[0]
        label = f"{row['artist']} — {row['title']} ({row['song_version']})"
        plot_track(chroma, label, os.path.join(PLOT_DIR, f"{stem}_chromagram.png"))

    # song group comparisons

    groups = defaultdict(list)
    for row in rows:
        groups[row["song_id"].strip()].append(row)
    for song_id, group_rows in groups.items():
        safe = group_rows[0]["title"].lower().replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        plot_group(group_rows, os.path.join(PLOT_DIR, f"group_{song_id}_{safe}.png"))
 
    print(f"\nall group plots saved to '{PLOT_DIR}/'")

if __name__ == "__main__":
    main()