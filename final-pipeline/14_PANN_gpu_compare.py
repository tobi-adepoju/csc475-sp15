import os
import shutil
import pandas as pd
import torch
import lancedb
import numpy as np
import librosa
from panns_inference import AudioTagging
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = 'processed_dataset.csv'
DB_PATH = './song_similarity_db'
AUDIO_DIR = './downloads/converted_to_22k_wav/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'

# Database Initialization ---
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
    # Initialize connection and ensure we start fresh or append
    print(f"Connected to LanceDB at {DB_PATH}")
        
# Embedding Generation Logic ---
# Load PANNs CNN14 to GPU
# Note: panns_inference handles model downloading automatically
model = AudioTagging(checkpoint_path=None, device=DEVICE)

def get_embedding(file_path):
    """
    Extracts 2048-d embedding using CNN14 global pooling.
    Optimized for RTX 5080/WSL2.
    """
    try:
        # 1. Load audio and resample to 32k (required by PANNs)
        # We use duration=None to load the whole song, or limit it for speed
        audio, _ = librosa.load(file_path, sr=32000, mono=True)

        # 2. Reshape for the model: (batch_size, samples)
        audio_input = audio[None, :]

        # 3. The 'inference' method returns (clipwise_output, embedding)
        _, emb = model.inference(audio_input) 

        # 4. Prep embedding for DB
        # Convert from torch/numpy to a flat list for LanceDB
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
            
        return emb.flatten().tolist()
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main Processing Loop ---
def process_dataset():
    df = pd.read_csv(CSV_PATH)
    
    # Split into Originals and Covers
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

# Cross-Comparison
def compute_similarity_results():
    orig_table = db.open_table("originals")
    cover_table = db.open_table("covers")
    
    # Convert originals to pandas to iterate easily
    originals = orig_table.to_pandas()
    final_results = []

    print("Computing Cross-Similarity using GPU...")
    for _, orig in tqdm(originals.iterrows(), total=len(originals)):
        # Vector search in the covers table
        # We use .limit(len(cover_table)) if you want to rank EVERY cover
        matches = cover_table.search(orig['vector']) \
            .metric("cosine") \
            .limit(1000) \
            .to_pandas()

        for _, match in matches.iterrows():
            final_results.append({
                "fkid_original": str(orig['pkid']),
                "fkid_cover": str(match['pkid']),
                # Distance is 0 to 2; 1 - distance gives a rough similarity
                "match_value": float(1 - match['_distance']), 
                "raw": f"Match: {orig['title']} vs {match['title']}"
            })

    if final_results:
        db.create_table("similarity_results", data=final_results, mode="overwrite")
        print(f"Success! {len(final_results)} matches stored in 'similarity_results'.")
    else:
        print("No matches generated. Check vector dimensions.")

def show_results(): # test function
    # Open tables
    originals = db.open_table("originals").to_pandas()
    results = db.open_table("similarity_results").to_pandas()
    covers = db.open_table("covers").to_pandas()

    # Merge results with original metadata for readability
    # Join on the 'pkid' and 'fkid' columns
    view = results.merge(originals[['pkid', 'title', 'artist']], left_on='fkid_original', right_on='pkid')
    view = view.rename(columns={'title': 'orig_title', 'artist': 'orig_artist'})

    # Merge with cover metadata
    view = view.merge(covers[['pkid', 'title', 'artist']], left_on='fkid_cover', right_on='pkid')
    view = view.rename(columns={'title': 'cover_title', 'artist': 'cover_artist'})

    # Select and sort the best matches
    top_matches = view[['orig_title', 'cover_artist', 'match_value']].sort_values(by='match_value', ascending=False)

    print("\n--- TOP 100 CROSS-VERSION MATCHES ---")
    print(top_matches.head(100).to_string(index=False))

def query_and_report():
    # Connect to the local LanceDB
    db = lancedb.connect("./song_similarity_db")

    # Open tables
    try:
        originals_table = db.open_table("originals")
        covers_table = db.open_table("covers")
        results_table = db.open_table("similarity_results")
    except Exception as e:
        print(f"Error opening tables: {e}")
        return

    # Load into memory for reporting
    originals = originals_table.to_pandas()
    results = results_table.to_pandas()
    covers = covers_table.to_pandas()

    # Merge results with original metadata
    view = results.merge(
        originals[['pkid', 'title', 'artist']], 
        left_on='fkid_original', 
        right_on='pkid'
    ).rename(columns={'title': 'orig_title', 'artist': 'orig_artist'})

    # Merge with cover metadata
    view = view.merge(
        covers[['pkid', 'title', 'artist']], 
        left_on='fkid_cover', 
        right_on='pkid'
    ).rename(columns={'title': 'cover_title', 'artist': 'cover_artist'})

    # Sort by similarity match
    report_df = view[['orig_title', 'cover_artist', 'cover_title', 'match_value']].sort_values(
        by='match_value', ascending=False
    )

    # Print to screen
    header = "AUDIO SIMILARITY REPORT (PANNs CNN14 + LanceDB)"
    border = "=" * 80
    print(f"\n{border}\n{header}\n{border}")
    print(report_df.to_string(index=False))
    print(f"{border}\n")

    # Save to text file
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
