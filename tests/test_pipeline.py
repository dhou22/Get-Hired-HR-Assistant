"""
Pipeline Testing Suite
Quick validation tests for each processing step
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)

def test_step1_fetch():
    """Test Step 1: Data fetching"""
    logger.info("=" * 60)
    logger.info("TEST: Step 1 - Fetch Data")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run
    
    try:
        results = run(limit=5)
        
        assert len(results) > 0, "No resumes fetched"
        assert 'id' in results[0], "Missing candidate ID"
        assert 'resume_text' in results[0], "Missing resume text"
        
        logger.info(f"- Fetched: {len(results)} resumes")
        logger.info(f"- Sample ID: {results[0]['id']}")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def test_step2_extract():
    """Test Step 2: Text extraction"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Step 2 - Text Extraction")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run as fetch
    from src.step2_extract_text import run as extract
    
    try:
        resumes = fetch(limit=3)
        results = extract(resumes)
        
        successful = sum(1 for r in results if r.get('extraction_status') == 'success')
        
        assert len(results) > 0, "No extractions performed"
        assert successful > 0, "No successful extractions"
        
        logger.info(f"- Processed: {len(results)} resumes")
        logger.info(f"- Successful: {successful}")
        logger.info(f"- Avg word count: {sum(r.get('word_count', 0) for r in results) / len(results):.0f}")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def test_step3_clean():
    """Test Step 3: Text cleaning"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Step 3 - Text Cleaning")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run as fetch
    from src.step2_extract_text import run as extract
    from src.step3_clean_text import run as clean
    
    try:
        resumes = fetch(limit=3)
        extracted = extract(resumes)
        results = clean(extracted)
        
        successful = sum(1 for r in results if r.get('cleaning_status') == 'success')
        
        assert len(results) > 0, "No cleaning performed"
        assert successful > 0, "No successful cleaning"
        
        avg_reduction = sum(r.get('reduction_ratio', 0) for r in results if r.get('cleaning_status') == 'success') / successful
        
        logger.info(f"- Processed: {len(results)} resumes")
        logger.info(f"- Successful: {successful}")
        logger.info(f"- Avg text reduction: {avg_reduction:.2f}%")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def test_step4_anonymize():
    """Test Step 4: PII anonymization"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Step 4 - PII Anonymization")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run as fetch
    from src.step2_extract_text import run as extract
    from src.step3_clean_text import run as clean
    from src.step4_anonymize_pii import run as anonymize
    
    try:
        resumes = fetch(limit=3)
        extracted = extract(resumes)
        cleaned = clean(extracted)
        results = anonymize(cleaned)
        
        successful = sum(1 for r in results if r.get('anonymization_status') == 'success')
        pii_detected = sum(1 for r in results if r.get('pii_mapping', {}).get('original_email'))
        
        assert len(results) > 0, "No anonymization performed"
        assert successful > 0, "No successful anonymization"
        
        logger.info(f"- Processed: {len(results)} resumes")
        logger.info(f"- Successful: {successful}")
        logger.info(f"- PII detected: {pii_detected}")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def test_step5_chunk():
    """Test Step 5: Semantic chunking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Step 5 - Semantic Chunking")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run as fetch
    from src.step2_extract_text import run as extract
    from src.step3_clean_text import run as clean
    from src.step4_anonymize_pii import run as anonymize
    from src.step5_semantic_chunking import run as chunk
    
    try:
        resumes = fetch(limit=3)
        extracted = extract(resumes)
        cleaned = clean(extracted)
        anonymized = anonymize(cleaned)
        results = chunk(anonymized)
        
        assert len(results) > 0, "No chunks created"
        
        # Analyze chunks
        sections = {}
        for chunk in results:
            section = chunk.get('section_type', 'unknown')
            sections[section] = sections.get(section, 0) + 1
        
        logger.info(f"- Total chunks: {len(results)}")
        logger.info(f"- Avg chunks per resume: {len(results) / len(resumes):.2f}")
        logger.info(f"- Section distribution:")
        for section, count in sections.items():
            logger.info(f"  - {section}: {count}")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def test_step6_embeddings():
    """Test Step 6: Embedding generation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Step 6 - Embedding Generation")
    logger.info("=" * 60)
    
    from src.step1_fetch_data import run as fetch
    from src.step2_extract_text import run as extract
    from src.step3_clean_text import run as clean
    from src.step4_anonymize_pii import run as anonymize
    from src.step5_semantic_chunking import run as chunk
    from src.step6_generate_embeddings import run as embed
    
    try:
        resumes = fetch(limit=2)
        extracted = extract(resumes)
        cleaned = clean(extracted)
        anonymized = anonymize(cleaned)
        chunks = chunk(anonymized)
        results = embed(chunks)
        
        assert len(results) > 0, "No embeddings generated"
        assert 'embedding' in results[0], "Missing embedding"
        assert 'embedding_dim' in results[0], "Missing embedding dimension"
        
        logger.info(f"- Total embeddings: {len(results)}")
        logger.info(f"- Embedding dimension: {results[0]['embedding_dim']}")
        logger.info(f"- Sample embedding preview: {results[0]['embedding'][:3]}")
        logger.info(f"- Status: PASS")
        return True
        
    except Exception as e:
        logger.error(f"- Status: FAIL - {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE TEST SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Step 1: Fetch Data", test_step1_fetch),
        ("Step 2: Text Extraction", test_step2_extract),
        ("Step 3: Text Cleaning", test_step3_clean),
        ("Step 4: PII Anonymization", test_step4_anonymize),
        ("Step 5: Semantic Chunking", test_step5_chunk),
        ("Step 6: Embedding Generation", test_step6_embeddings)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"- {test_name}: {status}")
    
    logger.info("-" * 80)
    logger.info(f"Total: {passed_count}/{total_count} tests passed")
    logger.info(f"Success Rate: {(passed_count/total_count)*100:.1f}%")
    logger.info("=" * 80)
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)