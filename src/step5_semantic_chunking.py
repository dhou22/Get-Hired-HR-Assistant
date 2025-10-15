"""
Step 5: Semantic Chunking for JSON-Structured Resumes (DEBUGGED)
Technique: Section-aware chunking aligned with resume JSON structure
Strategy: Extract from JSON structure and create semantic chunks by section
"""

import json
import time
from typing import Dict, List, Any

try:
    from .utils import get_logger, save_checkpoint, calculate_statistics, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, calculate_statistics, format_report

logger = get_logger(__name__)


class JSONSemanticChunker:
    def __init__(self, max_chunk_size: int = 400, min_chunk_size: int = 50):
        """
        Initialize chunker for JSON-structured resumes
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text by words"""
        if not isinstance(text, str):
            return []
        return text.split()
    
    def _token_count(self, text: str) -> int:
        """Count tokens in text"""
        return len(self._tokenize(text))
    
    def _flatten_dict(self, d: Any, prefix: str = "") -> str:
        """
        Recursively flatten dictionary/list to readable text
        """
        if isinstance(d, dict):
            parts = []
            for key, value in d.items():
                if isinstance(value, (dict, list)) and value:
                    flattened = self._flatten_dict(value, f"{prefix}{key}")
                    if flattened:
                        parts.append(flattened)
                elif isinstance(value, str) and value and value != "Unknown":
                    parts.append(value)
            return "\n".join(parts)
        elif isinstance(d, list):
            parts = []
            for item in d:
                if isinstance(item, dict):
                    flattened = self._flatten_dict(item, prefix)
                    if flattened:
                        parts.append(flattened)
                elif isinstance(item, str) and item and item != "Unknown":
                    parts.append(item)
            return "\n".join(parts)
        elif isinstance(d, str) and d and d != "Unknown":
            return d
        return ""
    
    def _extract_section_text(self, data: Dict, section_key: str) -> str:
        """Extract text from a specific section of resume"""
        if not isinstance(data, dict) or section_key not in data:
            return ""
        
        section_data = data[section_key]
        if not section_data:
            return ""
        
        return self._flatten_dict(section_data)
    
    def _create_chunk_from_text(self, text: str, section_name: str, 
                                candidate_id: str, chunk_index: int,
                                pii_mapping: Dict = None,
                                encrypted_pii: str = None) -> Dict:
        """Create a chunk from text"""
        if not text or self._token_count(text) < self.min_chunk_size:
            return None
        
        return {
            'chunk_id': f"{candidate_id}_{chunk_index:03d}",
            'candidate_id': candidate_id,
            'section_type': section_name,
            'chunk_text': text.strip(),
            'position': chunk_index,
            'token_count': self._token_count(text),
            'pii_mapping': pii_mapping or {},
            'encrypted_pii': encrypted_pii or '',
            'status': 'success'
        }
    
    def _split_large_text(self, text: str, section_name: str, candidate_id: str,
                         start_index: int, pii_mapping: Dict = None,
                         encrypted_pii: str = None) -> List[Dict]:
        """Split text that exceeds max_chunk_size into multiple chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        chunk_index = start_index
        
        for line in lines:
            test_text = current_chunk + '\n' + line if current_chunk else line
            token_count = self._token_count(test_text)
            
            if token_count > self.max_chunk_size and current_chunk:
                chunk = self._create_chunk_from_text(
                    current_chunk, section_name, candidate_id, chunk_index,
                    pii_mapping, encrypted_pii
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                current_chunk = line
            else:
                current_chunk = test_text
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_from_text(
                current_chunk, section_name, candidate_id, chunk_index,
                pii_mapping, encrypted_pii
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_json_resume(self, resume_data: Dict, candidate_id: str,
                          pii_mapping: Dict = None,
                          encrypted_pii: str = None) -> List[Dict]:
        """
        Create semantic chunks from JSON-structured resume
        
        Extracts data from raw_data if available, otherwise uses top-level keys
        Creates separate chunks for each section
        """
        chunks = []
        chunk_index = 0
        
        # Extract raw_data if available (structured resume)
        raw_data = None
        if 'raw_data' in resume_data and resume_data['raw_data']:
            raw_data = resume_data['raw_data']
            logger.debug(f"Found raw_data with keys: {list(raw_data.keys())}")
        else:
            logger.debug(f"No raw_data found. Resume keys: {list(resume_data.keys())}")
        
        # Process sections in logical order
        section_order = [
            ('personal_info', 'Contact Information'),
            ('experience', 'Work Experience'),
            ('education', 'Education'),
            ('skills', 'Skills'),
            ('projects', 'Projects'),
            ('certifications', 'Certifications'),
            ('publications', 'Publications'),
            ('internships', 'Internships'),
            ('achievements', 'Achievements'),
            ('workshops', 'Workshops'),
            ('teaching_experience', 'Teaching Experience'),
        ]
        
        # Try structured extraction first
        if raw_data:
            for section_key, section_name in section_order:
                section_text = self._extract_section_text(raw_data, section_key)
                
                if section_text and self._token_count(section_text) >= self.min_chunk_size:
                    logger.debug(f"Extracted {section_name}: {self._token_count(section_text)} tokens")
                    
                    # Check if section needs to be split
                    if self._token_count(section_text) > self.max_chunk_size:
                        section_chunks = self._split_large_text(
                            section_text, section_name, candidate_id,
                            chunk_index, pii_mapping, encrypted_pii
                        )
                        chunks.extend(section_chunks)
                        chunk_index = len(chunks)
                    else:
                        chunk = self._create_chunk_from_text(
                            section_text, section_name, candidate_id,
                            chunk_index, pii_mapping, encrypted_pii
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
        
        # Fallback: if no chunks created, try text fields
        if not chunks:
            logger.debug("No structured chunks created. Trying text fallback...")
            
            text = resume_data.get('anonymized_text') or \
                   resume_data.get('cleaned_text') or \
                   resume_data.get('raw_text') or \
                   resume_data.get('resume_text', '')
            
            logger.debug(f"Text fallback found: {len(text)} characters")
            
            if text and self._token_count(text) >= self.min_chunk_size:
                if self._token_count(text) > self.max_chunk_size:
                    section_chunks = self._split_large_text(
                        text, 'Full Resume', candidate_id, 0,
                        pii_mapping, encrypted_pii
                    )
                    chunks.extend(section_chunks)
                else:
                    chunk = self._create_chunk_from_text(
                        text, 'Full Resume', candidate_id, 0,
                        pii_mapping, encrypted_pii
                    )
                    if chunk:
                        chunks.append(chunk)
        
        if not chunks:
            logger.warning(f"Could not create any chunks for {candidate_id}")
        
        return chunks


def process_resumes(resumes: List[Dict]) -> List[Dict]:
    """
    Process all JSON resumes for semantic chunking
    
    Args:
        resumes: List of resume dictionaries (JSON format)
        
    Returns:
        List of chunk dictionaries
    """
    if not resumes:
        logger.warning("No resumes provided for chunking")
        return []
    
    logger.info(f"Starting semantic chunking for {len(resumes)} resumes")
    start_time = time.time()
    
    chunker = JSONSemanticChunker()
    all_chunks = []
    successful = 0
    failed = 0
    
    for resume in resumes:
        try:
            candidate_id = resume.get('candidate_id') or resume.get('id', 'UNKNOWN')
            pii_mapping = resume.get('pii_mapping', {})
            encrypted_pii = resume.get('encrypted_pii', '')
            
            chunks = chunker.chunk_json_resume(
                resume, candidate_id, pii_mapping, encrypted_pii
            )
            
            if chunks:
                all_chunks.extend(chunks)
                successful += 1
                logger.info(f"Created {len(chunks)} chunks for {candidate_id}")
            else:
                logger.warning(f"No chunks generated for {candidate_id}")
                failed += 1
        
        except Exception as e:
            logger.error(f"Chunking failed for {resume.get('candidate_id', 'unknown')}: {e}", exc_info=True)
            failed += 1
            continue
    
    duration = time.time() - start_time
    
    # Calculate statistics
    total = len(resumes)
    success_rate = (successful / total * 100) if total > 0 else 0
    
    logger.info("=" * 60)
    logger.info("STEP: SEMANTIC CHUNKING")
    logger.info("=" * 60)
    logger.info(f"Total Items: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total Chunks Created: {len(all_chunks)}")
    avg_chunks = round(len(all_chunks) / successful, 2) if successful > 0 else 0
    logger.info(f"Avg Chunks per Resume: {avg_chunks}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info("=" * 60)
    
    # Save checkpoint
    checkpoint_file = save_checkpoint(all_chunks, 'step5_chunking')
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    return all_chunks


def run(input_data: List[Dict]) -> List[Dict]:
    """Execute Step 5: Semantic chunking"""
    return process_resumes(input_data)


if __name__ == "__main__":
    # Test with sample JSON resume
    test_resume = {
        "candidate_id": "TEST_001",
        "raw_data": {
            "personal_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1-555-0123",
                "location": {"city": "San Francisco", "country": "USA"},
                "summary": "Experienced Python Developer with 5 years in full-stack development and strong background in microservices architecture"
            },
            "experience": [
                {
                    "company": "Tech Corp",
                    "title": "Senior Developer",
                    "level": "senior",
                    "responsibilities": [
                        "Led development of microservices architecture using Python and Kubernetes",
                        "Managed team of 5 developers across multiple projects",
                        "Improved system performance by 40% through optimization and caching strategies",
                        "Implemented CI/CD pipelines using Jenkins and Docker"
                    ]
                },
                {
                    "company": "StartUp Inc",
                    "title": "Junior Developer",
                    "level": "entry",
                    "responsibilities": [
                        "Built customer-facing web applications using React and Django",
                        "Implemented automated testing suite with pytest and Jest",
                        "Contributed to database schema design and optimization"
                    ]
                }
            ],
            "education": [
                {
                    "degree": {"level": "B.S.", "field": "Computer Science"},
                    "institution": {"name": "University of Technology", "location": "San Francisco"},
                    "dates": {"end": "2018"},
                    "achievements": {"gpa": 3.8}
                }
            ],
            "skills": {
                "technical": {
                    "programming_languages": [
                        {"name": "Python", "level": "expert"},
                        {"name": "JavaScript", "level": "intermediate"},
                        {"name": "Go", "level": "beginner"}
                    ],
                    "frameworks": [
                        {"name": "Django", "level": "expert"},
                        {"name": "React", "level": "intermediate"},
                        {"name": "FastAPI", "level": "intermediate"}
                    ],
                    "databases": [
                        {"name": "PostgreSQL", "level": "expert"},
                        {"name": "MongoDB", "level": "intermediate"}
                    ]
                }
            }
        }
    }
    
    chunks = run([test_resume])
    
    print(f"\n--- CHUNKING RESULTS ---")
    print(f"Total chunks created: {len(chunks)}")
    
    if chunks:
        for chunk in chunks:
            print(f"\nChunk {chunk['position']}: {chunk['section_type']}")
            print(f"   Tokens: {chunk['token_count']}")
            print(f"   Preview: {chunk['chunk_text'][:100]}...")
            print("-" * 60)
    else:
        print("No chunks generated!")