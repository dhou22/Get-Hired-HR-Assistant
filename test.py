"""
Debug script to inspect the structure of your JSON resume files
"""

import json
import os
from pathlib import Path

def inspect_json_file(filepath):
    """Load and inspect a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def analyze_structure(data, indent=0):
    """Recursively analyze the structure of data"""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        print(f"{prefix}Dict with {len(data)} keys:")
        for key in list(data.keys())[:10]:  # Show first 10 keys
            value = data[key]
            if isinstance(value, dict):
                print(f"{prefix}  {key}: Dict")
            elif isinstance(value, list):
                print(f"{prefix}  {key}: List ({len(value)} items)")
            elif isinstance(value, str):
                preview = value[:60].replace('\n', ' ')
                print(f"{prefix}  {key}: String - '{preview}...'")
            else:
                print(f"{prefix}  {key}: {type(value).__name__}")
    
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items")
        if data:
            print(f"{prefix}First item:")
            analyze_structure(data[0], indent + 1)

def main():
    # Check for resume files
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return
    
    # Find JSON files (prioritize step1 or step2 output)
    json_files = sorted(data_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files\n")
    
    # Inspect a few files from different steps
    files_to_inspect = [
        "step1_fetch",  # Original fetched resumes
        "step2_extraction",  # After text extraction
        "step4_anonymization",  # After anonymization
    ]
    
    for pattern in files_to_inspect:
        matching_files = [f for f in json_files if pattern in f.name]
        if matching_files:
            filepath = matching_files[-1]  # Get the latest one
            print(f"\n{'='*70}")
            print(f"FILE: {filepath.name}")
            print(f"{'='*70}")
            
            data = inspect_json_file(filepath)
            if data:
                analyze_structure(data)
                
                # If it's a list, show more detail about first item
                if isinstance(data, list) and data:
                    print(f"\n{'-'*70}")
                    print("FIRST ITEM DETAILED:")
                    print(f"{'-'*70}")
                    first = data[0]
                    if isinstance(first, dict):
                        print(json.dumps(first, indent=2)[:1000])
                        print("... [truncated]")
    
    # Also show what's in step5 (chunking output)
    step5_files = [f for f in json_files if "step5" in f.name]
    if step5_files:
        print(f"\n{'='*70}")
        print(f"STEP5 OUTPUT (chunking)")
        print(f"{'='*70}")
        latest_step5 = step5_files[-1]
        data = inspect_json_file(latest_step5)
        if data:
            analyze_structure(data)

if __name__ == "__main__":
    main()