import os
import csv
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d

CSV_FILE    = "processed_dataset.csv"
WAV_DIR     = "converted_to_22k_wav/downloads"
FEATURE_DIR = "features"
SR          = 22050
HOP_LENGTH  = 512
N_CHROMA    = 12

"""
puts chroma into one vector per detected beat using median aggregation
requires: WAV files in converted_to_22k_wav (run 3_convert_to_22k_wave.sh first)
          CQT chroma in features (run extract_chroma.py first)

"""

os.makedirs(FEATURE_DIR, exist_ok=True)
 
 
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
    """Frame-level CQT chroma, L2-normalized."""
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)
    ch = uniform_filter1d(ch, size=11, axis=1)
    norms = np.linalg.norm(ch, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (ch / norms).astype(np.float32)
 
 
def compute_beat_chroma(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    collapse frame-level chroma into one vector per detected beat.
 
    beat-sync chroma reduces a ~3000-frame track to ~100 beat vectors,
    making DTW faster and less sensitive to tempo differences between covers.
 
    returns:
        beat_chroma : (12, n_beats) L2-normalized
        beat_times  : (n_beats,) seconds per beat
        tempo       : estimated BPM
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP_LENGTH)
 
    # frame-level chroma without smoothing, sync aggregation handles it
    chroma = extract_chroma(y)
 
    # median aggregation, can handlenoisy or transient frames within a beat
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
 
    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    beat_chroma = (beat_chroma / norms).astype(np.float32)
 
    return beat_chroma, beat_times, tempo
 
 
def main():
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
 
    print(f"computing beat-sync chroma for {len(rows)} tracks\n")
 
    ok, skipped = 0, 0
    for row in rows:
        file_name = row["file_name"].strip()
        stem      = os.path.splitext(file_name)[0]
 
        try:
            y = load_audio(file_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            skipped += 1
            continue
 
        beat_chroma, beat_times, tempo = compute_beat_chroma(y)
 
        np.save(os.path.join(FEATURE_DIR, f"{stem}_beat_chroma.npy"), beat_chroma)
        np.save(os.path.join(FEATURE_DIR, f"{stem}_beat_times.npy"),  beat_times)
 
        n_frames = int(len(y) / SR * SR / HOP_LENGTH)
        print(f"  {row['artist']:<30}  tempo={tempo:5.1f} BPM  "
              f"{n_frames:>5} frames -> {beat_chroma.shape[1]:>4} beats")
        ok += 1
 
    print(f"\nbeat-sync chroma saved for {ok} tracks, skipped {skipped}")
    print(f"features saved to '{FEATURE_DIR}/'")
 
 
if __name__ == "__main__":
    main()
 