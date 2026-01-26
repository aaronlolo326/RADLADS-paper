#!/usr/bin/env python3
"""
Convert results JSON to CSV format.
Extracts ctx_len from keys like "32768,none" and ignores stderr entries.
"""

import json
import csv
import sys
import re
from pathlib import Path


def extract_ctx_len(key):
    """Extract ctx_len from keys like '32768,none' or '4096_stderr,none'."""
    # Match the number at the start of the key (before comma or underscore)
    match = re.match(r'^(\d+)', key)
    if match:
        return int(match.group(1))
    return None


def json_results_to_csv(json_file, output_file=None):
    """
    Convert results JSON to CSV.
    
    Args:
        json_file: Path to input JSON file
        output_file: Path to output CSV file (default: same name as JSON with .csv extension)
    
    Returns:
        True if successful, False otherwise
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        return False
    
    try:
        # Read JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get results section
        if 'results' not in data:
            print(f"Warning: 'results' key not found in {json_path.name}, skipping", file=sys.stderr)
            return False
        
        results = data['results']
        
        # Prepare CSV data
        csv_rows = []
        
        for task_name, task_data in results.items():
            for key, value in task_data.items():
                # Skip stderr entries
                if 'stderr' in key:
                    continue
                
                # Skip 'alias' key
                if key == 'alias':
                    continue
                
                # Extract ctx_len from key
                ctx_len = extract_ctx_len(key)
                if ctx_len is None:
                    # Skip if we can't extract ctx_len
                    continue
                
                # Only include numeric values
                if isinstance(value, (int, float)):
                    csv_rows.append({
                        'task': task_name,
                        'ctx_len': ctx_len,
                        'score': value
                    })
        
        # Sort by task name, then by ctx_len
        csv_rows.sort(key=lambda x: (x['task'], x['ctx_len']))
        
        # Determine output file
        if output_file is None:
            output_file = json_path.with_suffix('.csv')
        else:
            output_file = Path(output_file)
        
        # Write CSV
        if csv_rows:
            fieldnames = ['task', 'ctx_len', 'score']
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"Converted {len(csv_rows)} rows: {json_path.name} -> {output_file.name}")
            return True
        else:
            print(f"Warning: No data found to convert in {json_path.name}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Error processing {json_path.name}: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: json_to_csv.py <input_json_or_dir> [output_csv]", file=sys.stderr)
        print("  If input is a directory, processes all results_*.json files in it", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    if input_path.is_dir():
        # Process all results_*.json files in the directory
        json_files = sorted(input_path.glob('results_*.json'))
        if not json_files:
            print(f"Warning: No results_*.json files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(json_files)} JSON file(s) to process...\n")
        success_count = 0
        for json_file in json_files:
            if json_results_to_csv(json_file):
                success_count += 1
        
        print(f"\nProcessed {success_count}/{len(json_files)} file(s) successfully")
    else:
        # Process single file
        if not json_results_to_csv(input_path, output_csv):
            sys.exit(1)
