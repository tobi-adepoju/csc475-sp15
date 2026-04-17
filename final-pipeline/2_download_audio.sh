#!/bin/bash

INPUT_FILE="processed_dataset.csv"
DOWNLOAD_FOLDER="downloads"
YTDLP="./yt-dlp" 


mkdir -p "$DOWNLOAD_FOLDER"

tail -n +2 "$INPUT_FILE" | while IFS=, read -r pkid url song_id title artist file_name song_version; do
    
    clean_url=$(echo "$url" | tr -d '\r')
    clean_file=$(echo "$file_name" | tr -d '\r')

    if [ "$clean_url" != "URL_NOT_FOUND" ] && [ -n "$clean_url" ]; then
        echo "downloading: $clean_file"
        
        yt-dlp -o "$DOWNLOAD_FOLDER/$clean_file" "$clean_url"
        
    else
        echo "skipping $clean_file: no URL found in the dataset"
    fi

done

echo "downloads complete, check '$DOWNLOAD_FOLDER' directory"