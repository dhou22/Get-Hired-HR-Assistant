"""
Pipeline utility to preserve raw_data through all processing steps
This ensures that structured resume data is maintained for later use
"""

def preserve_raw_data(input_list, output_list):
    """
    Merge raw_data from input to output
    
    Args:
        input_list: Original list with raw_data
        output_list: Processed list without raw_data
    
    Returns:
        Updated output_list with raw_data preserved
    """
    # Create a map of candidate IDs to raw_data from input
    raw_data_map = {}
    for item in input_list:
        cid = item.get('candidate_id') or item.get('id')
        if cid and 'raw_data' in item:
            raw_data_map[cid] = item['raw_data']
    
    # Add raw_data to output items
    for item in output_list:
        cid = item.get('candidate_id') or item.get('id')
        if cid in raw_data_map:
            item['raw_data'] = raw_data_map[cid]
        else:
            item['raw_data'] = None
    
    return output_list


# ============================================================
# STEP 2: PRESERVE RAW_DATA IN EXTRACTION
# ============================================================
# Modify step2_extract_text.py - Update the process_resumes function:

def process_resumes_step2_FIXED(resumes):
    """
    After extraction, preserve raw_data from input
    """
    # ... existing extraction code ...
    results = []
    
    for resume in resumes:
        try:
            extracted_text = extract_text(resume)
            results.append({
                'candidate_id': resume.get('id'),
                'raw_text': extracted_text,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'extraction_status': 'success',
                'category': resume.get('category', 'Unknown'),
                'status': 'extracted',
                'raw_data': resume.get('raw_data')  # ✅ PRESERVE
            })
        except Exception as e:
            results.append({
                'candidate_id': resume.get('id'),
                'extraction_status': 'failed',
                'error': str(e),
                'raw_data': resume.get('raw_data')  # ✅ PRESERVE
            })
    
    return results


# ============================================================
# STEP 3: PRESERVE RAW_DATA IN CLEANING
# ============================================================
# Modify step3_clean_text.py - Update the process_resumes function:

def process_resumes_step3_FIXED(resumes):
    """
    After cleaning, preserve raw_data from input
    """
    # ... existing cleaning code ...
    results = []
    
    for resume in resumes:
        try:
            cleaned_text = clean_text(resume['raw_text'])
            results.append({
                'candidate_id': resume.get('candidate_id'),
                'cleaned_text': cleaned_text,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'cleaning_status': 'success',
                'category': resume.get('category', 'Unknown'),
                'status': 'cleaned',
                'raw_data': resume.get('raw_data'),  # ✅ PRESERVE
                'raw_text': resume.get('raw_text')   # Also preserve for reference
            })
        except Exception as e:
            results.append({
                'candidate_id': resume.get('candidate_id'),
                'cleaning_status': 'failed',
                'error': str(e),
                'raw_data': resume.get('raw_data')  # ✅ PRESERVE
            })
    
    return results


# ============================================================
# QUICK FIX: One-liner to add raw_data to existing checkpoints
# ============================================================
import json
from pathlib import Path

def add_raw_data_to_checkpoint(step_input_file, step_output_file):
    """
    Add raw_data from input checkpoint to output checkpoint
    Use this as a quick fix on existing files
    
    Args:
        step_input_file: Path to input checkpoint (e.g., step1_fetch_*.json)
        step_output_file: Path to output checkpoint (e.g., step4_anonymization_*.json)
    """
    # Load input with raw_data
    with open(step_input_file, 'r') as f:
        input_data = json.load(f)
    
    # Load output without raw_data
    with open(step_output_file, 'r') as f:
        output_data = json.load(f)
    
    # Create map
    raw_data_map = {}
    if isinstance(input_data, list):
        for item in input_data:
            cid = item.get('candidate_id') or item.get('id')
            if cid:
                raw_data_map[cid] = item.get('raw_data')
    else:
        raw_data_map[input_data.get('candidate_id')] = input_data.get('raw_data')
    
    # Add to output
    if isinstance(output_data, list):
        for item in output_data:
            cid = item.get('candidate_id')
            if cid in raw_data_map:
                item['raw_data'] = raw_data_map[cid]
    else:
        output_data['raw_data'] = raw_data_map.get(output_data.get('candidate_id'))
    
    # Save updated output
    with open(step_output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Updated {step_output_file} with raw_data from {step_input_file}")


if __name__ == "__main__":
    # QUICK FIX: Run this to add raw_data to your existing checkpoints
    # Example:
    input_checkpoint = "data/processed/step1_fetch_20251014_234004.json"
    output_checkpoint = "data/processed/step4_anonymization_20251014_235015.json"
    
    add_raw_data_to_checkpoint(input_checkpoint, output_checkpoint)
    
    print("\nNow run step5_semantic_chunking.py again!")