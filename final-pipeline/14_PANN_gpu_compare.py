import os
import shutil
import pandas as pd
import torch
import lancedb
import numpy as np
import librosa
from panns_inference import AudioTagging
from tqdm import tqdm

CSV_PATH = 'processed_dataset.csv'
DB_PATH = './song_similarity_db'
AUDIO_DIR = './downloads/converted_to_22k_wav/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'


if os.path.exists(DB_PATH):
    try:
        shutil.rmtree(DB_PATH)
        print(f"Successfully deleted {DB_PATH}")
    except Exception as e:
        print(f"Error deleting database: {e}")
else:
    print("Directory does not exist, nothing to delete.")
    
db = lancedb.connect(DB_PATH)

def setup_db():

    print(f"Connected to LanceDB at {DB_PATH}")
        
model = AudioTagging(checkpoint_path=None, device=DEVICE)

def get_embedding(file_path):
    """
    Extracts 2048-d embedding using CNN14 global pooling.
    Optimized for RTX 5080/WSL2.
    """
    try:
        audio, _ = librosa.load(file_path, sr=32000, mono=True)

        audio_input = audio[None, :]
        _, emb = model.inference(audio_input) 

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
            
        return emb.flatten().tolist()
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset():
    df = pd.read_csv(CSV_PATH)
    
    originals_df = df[df['song_version'].str.lower() == 'original']
    covers_df = df[df['song_version'].str.lower() == 'cover']

    print(f"Processing {len(originals_df)} Originals and {len(covers_df)} Covers...")

    def process_and_create(sub_df, table_name):
        data_to_ingest = []
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
            stem = os.path.splitext(row['file_name'])[0]
            file_full_path = os.path.join(AUDIO_DIR, f"{stem}_22k.wav")

            if not os.path.exists(file_full_path):
                print(f"Missing audio: {file_full_path}")
                continue

            vector = get_embedding(file_full_path)

            if vector:
                record = row.to_dict()
                record['vector'] = vector
                data_to_ingest.append(record)
            
        if data_to_ingest:
            return db.create_table(table_name, data=data_to_ingest, mode="overwrite")
        else:
            print(f"No data found for {table_name}, skipping table creation.")
            return None

    print("Ingesting Originals...")
    process_and_create(originals_df, "originals")

    print("Ingesting Covers...")
    process_and_create(covers_df, "covers")

def compute_similarity_results():
    orig_table = db.open_table("originals")
    cover_table = db.open_table("covers")
    
    originals = orig_table.to_pandas()
    final_results = []

    print("Computing Cross-Similarity using GPU...")
    for _, orig in tqdm(originals.iterrows(), total=len(originals)):
        matches = cover_table.search(orig['vector']) \
            .metric("cosine") \
            .limit(1000) \
            .to_pandas()

        for _, match in matches.iterrows():
            final_results.append({
                "fkid_original": str(orig['pkid']),
                "fkid_cover": str(match['pkid']),
                "match_value": float(1 - match['_distance']), 
                "raw": f"Match: {orig['title']} vs {match['title']}"
            })

    if final_results:
        db.create_table("similarity_results", data=final_results, mode="overwrite")
        print(f"Success! {len(final_results)} matches stored in 'similarity_results'.")
    else:
        print("No matches generated. Check vector dimensions.")

def show_results(): 
    originals = db.open_table("originals").to_pandas()
    results = db.open_table("similarity_results").to_pandas()
    covers = db.open_table("covers").to_pandas()
    view = results.merge(originals[['pkid', 'title', 'artist']], left_on='fkid_original', right_on='pkid')
    view = view.rename(columns={'title': 'orig_title', 'artist': 'orig_artist'})
    view = view.merge(covers[['pkid', 'title', 'artist']], left_on='fkid_cover', right_on='pkid')
    view = view.rename(columns={'title': 'cover_title', 'artist': 'cover_artist'})
    top_matches = view[['orig_title', 'cover_artist', 'match_value']].sort_values(by='match_value', ascending=False)

    print("\n--- TOP 100 CROSS-VERSION MATCHES ---")
    print(top_matches.head(100).to_string(index=False))

def query_and_report():
    db = lancedb.connect("./song_similarity_db")
    try:
        originals_table = db.open_table("originals")
        covers_table = db.open_table("covers")
        results_table = db.open_table("similarity_results")
    except Exception as e:
        print(f"Error opening tables: {e}")
        return

    originals = originals_table.to_pandas()
    results = results_table.to_pandas()
    covers = covers_table.to_pandas()

    view = results.merge(
        originals[['pkid', 'title', 'artist']], 
        left_on='fkid_original', 
        right_on='pkid'
    ).rename(columns={'title': 'orig_title', 'artist': 'orig_artist'})

    view = view.merge(
        covers[['pkid', 'title', 'artist']], 
        left_on='fkid_cover', 
        right_on='pkid'
    ).rename(columns={'title': 'cover_title', 'artist': 'cover_artist'})

    report_df = view[['orig_title', 'cover_artist', 'cover_title', 'match_value']].sort_values(
        by='match_value', ascending=False
    )

    header = "AUDIO SIMILARITY REPORT (PANNs CNN14 + LanceDB)"
    border = "=" * 80
    print(f"\n{border}\n{header}\n{border}")
    print(report_df.to_string(index=False))
    print(f"{border}\n")

    output_filename = "similarity_results_report.txt"
    with open(output_filename, "w") as f:
        f.write(f"{header}\n{border}\n")
        f.write(report_df.to_string(index=False))
        f.write(f"\n{border}\n")
    
    print(f"Report saved to {output_filename}")

if __name__ == "__main__":
    setup_db()
    process_dataset()
    compute_similarity_results()
    query_and_report()
