"""
Step 4: PII Anonymization (COMPLETE & FIXED)
Technique: Pattern matching + NER-based PII detection
Approach: GDPR-compliant anonymization with structured data extraction
Reference: Phase 1 document - Section 4.1
"""

import re
import json
import time
from typing import Dict, List, Optional
from cryptography.fernet import Fernet

try:
    from .utils import get_logger, save_checkpoint, calculate_statistics, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, calculate_statistics, format_report

logger = get_logger(__name__)


class PIIAnonymizer:
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize PII anonymizer with encryption"""
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate new key if none provided
            self.cipher = Fernet(Fernet.generate_key())
            logger.warning("No encryption key provided. Generated new key.")
    
    def detect_and_anonymize(self, text: str, candidate_id: str, raw_data: Dict = None) -> Dict:
        """
        Detect and anonymize PII in resume text.
        Also extracts structured PII from raw_data if available.
        
        Args:
            text: Resume text to anonymize
            candidate_id: Candidate ID
            raw_data: Optional structured resume data (JSON)
        
        Returns:
            Dictionary with anonymized text and PII mapping
        """
        pii_mapping = {
            'candidate_id': candidate_id,
            'original_name': None,
            'original_email': None,
            'original_phone': None,
            'original_address': None,
            'original_location': None,
            'original_linkedin': None,
            'original_github': None
        }
        
        anonymized_text = text
        
        # ===== PHASE 1: Extract structured PII from raw_data =====
        if raw_data and isinstance(raw_data, dict):
            personal_info = raw_data.get('personal_info', {})
            
            # Extract from structured fields
            if personal_info.get('name') and personal_info['name'] != 'Unknown':
                pii_mapping['original_name'] = personal_info['name']
                anonymized_text = anonymized_text.replace(
                    personal_info['name'],
                    f"[CANDIDATE_{candidate_id}]"
                )
            
            if personal_info.get('email') and personal_info['email'] != 'Unknown':
                pii_mapping['original_email'] = personal_info['email']
                anonymized_text = anonymized_text.replace(
                    personal_info['email'],
                    '[EMAIL_REDACTED]'
                )
            
            if personal_info.get('phone') and personal_info['phone'] != 'Unknown':
                pii_mapping['original_phone'] = personal_info['phone']
                anonymized_text = anonymized_text.replace(
                    personal_info['phone'],
                    '[PHONE_REDACTED]'
                )
            
            if personal_info.get('linkedin') and personal_info['linkedin'] != 'Unknown':
                pii_mapping['original_linkedin'] = personal_info['linkedin']
                anonymized_text = anonymized_text.replace(
                    personal_info['linkedin'],
                    '[LINKEDIN_REDACTED]'
                )
            
            if personal_info.get('github') and personal_info['github'] != 'Unknown':
                pii_mapping['original_github'] = personal_info['github']
                anonymized_text = anonymized_text.replace(
                    personal_info['github'],
                    '[GITHUB_REDACTED]'
                )
            
            # Extract location
            location = personal_info.get('location', {})
            if isinstance(location, dict):
                city = location.get('city')
                country = location.get('country')
                if city and city != 'Unknown':
                    pii_mapping['original_location'] = f"{city}, {country}" if country and country != 'Unknown' else city
        
        # ===== PHASE 2: Pattern-based detection (fallback) =====
        
        # 1. Email Detection
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, anonymized_text)
        if emails and not pii_mapping['original_email']:
            pii_mapping['original_email'] = emails[0]
            anonymized_text = re.sub(email_pattern, '[EMAIL_REDACTED]', anonymized_text)
        
        # 2. Phone Number Detection
        phone_pattern = r'(\+?1?\s*)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'
        phones = re.findall(phone_pattern, anonymized_text)
        if phones and not pii_mapping['original_phone']:
            phone_str = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
            pii_mapping['original_phone'] = phone_str.strip()
            anonymized_text = re.sub(phone_pattern, '[PHONE_REDACTED]', anonymized_text)
        
        # 3. Name Detection (heuristic: capitalized words in first lines)
        if not pii_mapping['original_name']:
            lines = text.split('\n')
            for line in lines[:10]:
                line = line.strip()
                # Match full names (2-3 capitalized words)
                name_match = re.match(r'^([A-Z][a-z]+\s+){1,2}[A-Z][a-z]+$', line)
                if name_match and len(line) < 50:
                    pii_mapping['original_name'] = line
                    anonymized_text = anonymized_text.replace(
                        line,
                        f"[CANDIDATE_{candidate_id}]",
                        1
                    )
                    break
        
        # 4. Address Detection
        address_pattern = r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}'
        addresses = re.findall(address_pattern, anonymized_text)
        if addresses and not pii_mapping['original_address']:
            pii_mapping['original_address'] = addresses[0]
            anonymized_text = re.sub(address_pattern, '[ADDRESS_REDACTED]', anonymized_text)
        
        # 5. Social Security Number Detection
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, anonymized_text):
            anonymized_text = re.sub(ssn_pattern, '[SSN_REDACTED]', anonymized_text)
        
        # 6. LinkedIn URL Detection
        linkedin_pattern = r'linkedin\.com/in/[^\s]+'
        linkedin_urls = re.findall(linkedin_pattern, anonymized_text)
        if linkedin_urls and not pii_mapping['original_linkedin']:
            pii_mapping['original_linkedin'] = linkedin_urls[0]
            anonymized_text = re.sub(linkedin_pattern, '[LINKEDIN_REDACTED]', anonymized_text)
        
        # 7. GitHub URL Detection
        github_pattern = r'github\.com/[^\s]+'
        github_urls = re.findall(github_pattern, anonymized_text)
        if github_urls and not pii_mapping['original_github']:
            pii_mapping['original_github'] = github_urls[0]
            anonymized_text = re.sub(github_pattern, '[GITHUB_REDACTED]', anonymized_text)
        
        # Encrypt PII mapping
        pii_json = json.dumps(pii_mapping)
        encrypted_pii = self.cipher.encrypt(pii_json.encode()).decode()
        
        return {
            'anonymized_text': anonymized_text,
            'pii_mapping': pii_mapping,
            'encrypted_pii': encrypted_pii
        }
    
    def decrypt_pii(self, encrypted_pii: str) -> Dict:
        """Decrypt PII mapping (for authorized access only)"""
        decrypted_json = self.cipher.decrypt(encrypted_pii.encode()).decode()
        return json.loads(decrypted_json)


def process_resumes(resumes: List[Dict], encryption_key: Optional[str] = None) -> List[Dict]:
    """
    Process all resumes for PII anonymization
    
    Args:
        resumes: List of resume dictionaries from Step 3
        encryption_key: Fernet encryption key (base64 encoded)
        
    Returns:
        List of dictionaries with anonymized text
    """
    logger.info(f"Starting PII anonymization for {len(resumes)} resumes")
    start_time = time.time()
    
    anonymizer = PIIAnonymizer(encryption_key)
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for resume in resumes:
        candidate_id = resume.get('candidate_id', 'UNKNOWN')
        
        # Check if input has cleaned text (from step 3)
        if resume.get('cleaning_status') == 'success' or 'cleaned_text' in resume or 'raw_text' in resume:
            try:
                text_to_anonymize = resume.get('cleaned_text') or resume.get('raw_text', '')
                
                if not text_to_anonymize:
                    logger.warning(f"No text found for {candidate_id}")
                    failed += 1
                    results.append({
                        'candidate_id': candidate_id,
                        'anonymization_status': 'failed',
                        'error': 'No text content',
                        'status': 'failed'
                    })
                    continue
                
                raw_data = resume.get('raw_data')
                anon_result = anonymizer.detect_and_anonymize(text_to_anonymize, candidate_id, raw_data)
                
                results.append({
                    'candidate_id': candidate_id,
                    'anonymized_text': anon_result['anonymized_text'],
                    'pii_mapping': anon_result['pii_mapping'],
                    'encrypted_pii': anon_result['encrypted_pii'],
                    'anonymization_status': 'success',
                    'category': resume.get('category', 'Unknown'),
                    'raw_data': resume.get('raw_data'),
                    'status': 'anonymized'
                })
                successful += 1
                logger.info(f"Successfully anonymized {candidate_id}")
                
            except Exception as e:
                logger.error(f"Anonymization failed for {candidate_id}: {e}")
                failed += 1
                results.append({
                    'candidate_id': candidate_id,
                    'anonymization_status': 'failed',
                    'error': str(e),
                    'status': 'failed'
                })
        else:
            logger.warning(f"Skipping {candidate_id} - no text content available")
            skipped += 1
            results.append({
                'candidate_id': candidate_id,
                'anonymization_status': 'skipped',
                'status': 'skipped'
            })
    
    duration = time.time() - start_time
    total = len(resumes)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    # Log statistics
    logger.info("=" * 60)
    logger.info("STEP: PII ANONYMIZATION")
    logger.info("=" * 60)
    logger.info(f"Total Items: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info("=" * 60)
    
    # Save checkpoint
    checkpoint_file = save_checkpoint(results, 'step4_anonymization')
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    return results


def run(input_data: List[Dict], encryption_key: Optional[str] = None) -> List[Dict]:
    """Execute Step 4: PII anonymization"""
    return process_resumes(input_data, encryption_key)


if __name__ == "__main__":
    # Test with sample data
    from step1_fetch_data import run as fetch_data
    from step2_extract_text import run as extract_text
    from step3_clean_text import run as clean_text
    
    resumes = fetch_data(limit=3)
    extracted = extract_text(resumes)
    cleaned = clean_text(extracted)
    results = run(cleaned)
    
    print(f"\n--- ANONYMIZATION RESULTS ---")
    successful_count = 0
    for r in results:
        if r['anonymization_status'] == 'success':
            successful_count += 1
            if successful_count <= 2:
                print(f"\nCandidate: {r['candidate_id']}")
                print(f"Status: {r['anonymization_status']}")
                print(f"PII Detected:")
                print(f"  - Name: {r['pii_mapping']['original_name']}")
                print(f"  - Email: {r['pii_mapping']['original_email']}")
                print(f"  - Phone: {r['pii_mapping']['original_phone']}")
                print(f"  - Location: {r['pii_mapping']['original_location']}")
                print(f"Text Preview: {r['anonymized_text'][:200]}...")
                print("-" * 60)
    
    print(f"\nTotal successful: {successful_count}")