"""
Step 3: Text Cleaning and Preprocessing
Technique: NLP-based preprocessing pipeline
Libraries: spaCy (tokenization, lemmatization), regex
Reference: Phase 1 document - Section 3.2
"""

import re
import time
import spacy
from typing import Dict, List, Set
try:
    from .utils import get_logger, save_checkpoint, calculate_statistics, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, calculate_statistics, format_report

logger = get_logger(__name__)

class TextCleaner:
    def __init__(self):
        """Initialize spaCy model with error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            logger.info("Falling back to basic cleaning")
            self.nlp = None
        
        # Domain-specific terms to preserve
        self.preserve_terms = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'node',
            'sql', 'postgresql', 'mongodb', 'redis', 'mysql',
            'machine learning', 'data science', 'devops', 'ci/cd',
            'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy',
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum'
        }
        
        # Basic stopwords (used if spaCy not available)
        self.basic_stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 
            'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean resume text using NLP pipeline
        
        Steps:
        1. Lowercase normalization
        2. URL/Email masking
        3. Special character removal
        4. Tokenization
        5. Stopword removal (preserving technical terms)
        6. Lemmatization
        """
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Mask URLs and emails (preserve for PII extraction later)
        text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Step 3: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 4: Remove special characters (keep alphanumeric + basic punctuation)
        text = re.sub(r'[^a-z0-9\s\.\,\-\+\#\/]', ' ', text)
        
        if self.nlp:
            # Step 5-6: spaCy processing
            doc = self.nlp(text)
            
            tokens = []
            for token in doc:
                # Check if token should be preserved
                if token.text in self.preserve_terms:
                    tokens.append(token.text)
                elif not token.is_stop and not token.is_punct and len(token.text) > 2:
                    tokens.append(token.lemma_)
            
            cleaned_text = ' '.join(tokens)
        else:
            # Fallback: Basic stopword removal
            words = text.split()
            cleaned_text = ' '.join([
                w for w in words 
                if w not in self.basic_stopwords and len(w) > 2
            ])
        
        return cleaned_text.strip()

def process_resumes(resumes: List[Dict]) -> List[Dict]:
    """
    Process all resumes for text cleaning
    
    Args:
        resumes: List of resume dictionaries from Step 2
        
    Returns:
        List of dictionaries with cleaned text
    """
    logger.info(f"Starting text cleaning for {len(resumes)} resumes")
    start_time = time.time()
    
    cleaner = TextCleaner()
    results = []
    
    for resume in resumes:
        if resume.get('extraction_status') == 'success':
            try:
                cleaned_text = cleaner.clean_text(resume['raw_text'])
                
                results.append({
                    'candidate_id': resume['candidate_id'],
                    'cleaned_text': cleaned_text,
                    'original_text': resume['raw_text'],
                    'original_length': len(resume['raw_text']),
                    'cleaned_length': len(cleaned_text),
                    'reduction_ratio': round(
                        (1 - len(cleaned_text) / len(resume['raw_text'])) * 100, 2
                    ) if len(resume['raw_text']) > 0 else 0,
                    'cleaning_status': 'success',
                    'category': resume.get('category', 'Unknown'),
                    'status': 'cleaned'
                })
                
            except Exception as e:
                logger.error(f"Cleaning failed for {resume['candidate_id']}: {e}")
                results.append({
                    'candidate_id': resume['candidate_id'],
                    'cleaned_text': '',
                    'cleaning_status': 'failed',
                    'error': str(e),
                    'status': 'failed'
                })
        else:
            # Skip failed extractions
            results.append({
                'candidate_id': resume['candidate_id'],
                'cleaned_text': '',
                'cleaning_status': 'skipped',
                'status': 'skipped'
            })
    
    duration = time.time() - start_time
    stats = calculate_statistics(results)
    logger.info(format_report("TEXT CLEANING", stats, duration))
    
    # Save checkpoint
    checkpoint_file = save_checkpoint(results, 'step3_cleaning')
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    return results

def run(input_data: List[Dict]) -> List[Dict]:
    """Execute Step 3: Text cleaning"""
    return process_resumes(input_data)

if __name__ == "__main__":
    # Test with sample data
    from step1_fetch_data import run as fetch_data
    from step2_extract_text import run as extract_text
    
    resumes = fetch_data(limit=3)
    extracted = extract_text(resumes)
    results = run(extracted)
    
    print(f"\n--- CLEANING RESULTS ---")
    for r in results[:2]:
        if r['cleaning_status'] == 'success':
            print(f"Candidate: {r['candidate_id']}")
            print(f"Original Length: {r['original_length']}")
            print(f"Cleaned Length: {r['cleaned_length']}")
            print(f"Reduction: {r['reduction_ratio']}%")
            print(f"Preview: {r['cleaned_text'][:200]}...")
            print("-" * 60)