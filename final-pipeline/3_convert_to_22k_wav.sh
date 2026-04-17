#!/bin/bash

mkdir -p converted_to_22k_wav

shopt -s nocaseglob

for file in *.{flac,wav,mp3,m4a}; do
    [[ -e "$file" ]] || continue

    filename="${file%.*}"
    
    echo "Processing: $file"

    ffmpeg -i "$file" -ar 22050 -ac 1 "converted_to_22k_wav/${filename}_22k.wav" -loglevel error

done

echo "converted files in the 'converted_to_22k_wav' folder"