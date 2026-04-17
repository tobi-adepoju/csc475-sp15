#!/bin/bash

DATASET_FILE="dataset_fixed.csv"
MARKDOWN_FILE="download.md"
OUTPUT_FILE="processed_dataset.csv"

echo "pkid,url,song_id,title,artist,file_name,song_version" > "$OUTPUT_FILE"

if [ ! -f "$DATASET_FILE" ]; then
    echo "$DATASET_FILE not found, using dataset.csv instead"
    DATASET_FILE="dataset.csv"
fi

tail -n +2 "$DATASET_FILE" | while read -r line; do
    [ -z "$line" ] && continue
    
    if command -v uuidgen >/dev/null 2>&1; then
        pkid=$(uuidgen | tr '[:upper:]' '[:lower:]')
    else
        pkid=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo $RANDOM-$RANDOM)
    fi
    
    file_name=$(echo "$line" | cut -d',' -f4)
    
    url=""
    if [ -n "$file_name" ]; then
        url=$(grep -A 1 -F "Filename: $file_name" "$MARKDOWN_FILE" | grep "Link:" | sed 's/.*Link: //' | tr -d '\r' | xargs)
    fi
    
    if [ -z "$url" ]; then
        url="URL_NOT_FOUND"
    fi
    
    echo "$pkid,$url,$line" >> "$OUTPUT_FILE"

done

echo "processing complete, URL column populated"
echo "results saved to: $OUTPUT_FILE"