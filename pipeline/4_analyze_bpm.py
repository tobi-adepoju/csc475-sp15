import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

INPUT_CSV = "processed_dataset.csv"
OUTPUT_CSV = "dataset_with_bpm.csv"
AUDIO_DIR = os.path.join(os.getcwd(), "downloads", "converted_to_22k_wav") 

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"error: {INPUT_CSV} not found")
        return
    
    df = pd.read_csv(INPUT_CSV)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_name = str(row['file_name']).strip()
        base_name = os.path.splitext(raw_name)[0]
        file_name = f"{base_name}_22k.wav"
        file_path = os.path.join(AUDIO_DIR, file_name)
        
        current_row = row.to_dict()
        
        if os.path.exists(file_path):
            try:
                y, sr = librosa.load(file_path, sr=None)
                
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                
                current_row['bpm'] = round(float(tempo), 2)
                
                beat_save_path = os.path.join(AUDIO_DIR, f"{base_name}_beats.npy")
                np.save(beat_save_path, beat_times)
                
            except Exception as e:
                current_row['bpm'] = f"ERROR: {str(e)}"
        else:
            current_row['bpm'] = "FILE_NOT_FOUND"
            
        results.append(current_row)

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBPMs saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()