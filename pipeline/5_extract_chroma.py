import os
import csv
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d

# load 22k WAV files and computer CQT chroma features
# saves one .npy file for each track


CSV_FILE   = "processed_dataset.csv"
WAV_DIR    = "converted_to_22k_wav\downloads"
OUTPUT_DIR = "features"
SR         = 22050
HOP_LENGTH = 512
N_CHROMA   = 12 
VARIANT    = "cqt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def wav_path(file_name: str) -> str:
    stem = os.path.splitext(file_name)[0]
    return os.path.join(WAV_DIR, f"{stem}_22k.wav")

def load_audio(path: str) -> np.ndarray:
    # load wav as mono float32, normalized
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)

def extract_chroma(y: np.ndarray, variant: str = VARIANT) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=N_CHROMA, hop_length=HOP_LENGTH)

    # temporal smoothing
    chroma = uniform_filter1d(chroma, size=11, axis=1)

    # normalize so loudness doesn't affect similarity
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)

    return (chroma/norms).astype(np.float32)

def main():
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
 
    print(f"found {len(rows)} tracks in {CSV_FILE}\n")
 
    ok, skipped = 0, 0
    for row in rows:
        file_name = row["file_name"].strip()
        song_id   = row["song_id"].strip()
        artist    = row["artist"].strip()
        path      = wav_path(file_name)
 
        if not os.path.exists(path):
            print(f"  [SKIP] WAV not found: {path}")
            skipped += 1
            continue
 
        y      = load_audio(path)
        chroma = extract_chroma(y, variant=VARIANT)

        # save as .npy
        stem     = os.path.splitext(file_name)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_chroma.npy")
        np.save(out_path, chroma)
        
        duration = len(y) / SR
        print(f"  [OK] {artist:<30}  shape={chroma.shape}  duration={duration:.1f}s  -> {out_path}")
        ok += 1
 
    print(f"\nextracted chroma for {ok} tracks, skipped {skipped}")
    print(f"features saved to '{OUTPUT_DIR}/'")
 
 
if __name__ == "__main__":
    main()



