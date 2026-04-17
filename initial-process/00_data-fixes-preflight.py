import csv
import re
import difflib

CSV_FILE = 'dataset.csv'
MD_FILE = 'download.md'
FIXED_CSV_FILE = 'dataset_fixed.csv'

def main():
    md_filenames = []
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(r'- Filename:\s*(.+\.mp3)', content)
        md_filenames = [m.strip() for m in matches]

    fixed_rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        fixed_rows.append(header)
        
        for row in reader:
            if not row or len(row) < 4:
                continue
            
            original_filename = row[3]
            
            closest_match = difflib.get_close_matches(original_filename, md_filenames, n=1, cutoff=0.7)
            
            if closest_match and closest_match[0] != original_filename:
                print(f"fixed typo: '{original_filename}' -> '{closest_match[0]}'")
                row[3] = closest_match[0]
            
            if len(row) > 4 and row[4] == 'covercr':
                row[4] = 'cover'
                print("fixed typo: 'covercr' -> 'cover'")
                
            fixed_rows.append(row)

    with open(FIXED_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)

    print(f"\n cleaned data saved to '{FIXED_CSV_FILE}'.")

if __name__ == '__main__':
    main()