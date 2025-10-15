import base64
import json
import time
import re
from io import BytesIO
from typing import Dict, List
from PyPDF2 import PdfReader
from docx import Document

try:
    from .utils import get_logger, save_checkpoint, calculate_statistics, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, calculate_statistics, format_report

logger = get_logger(__name__)

class TextExtractor:
    @staticmethod
    def extract_from_pdf(pdf_bytes: bytes) -> str:
        """Extract text from PDF binary data"""
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()

    @staticmethod
    def extract_from_docx(docx_bytes: bytes) -> str:
        """Extract text from DOCX binary data"""
        doc = Document(BytesIO(docx_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()

    @staticmethod
    def extract_from_json(json_bytes: bytes) -> str:
        """Extract text from JSON binary data (actual .docx JSON files)"""
        data = json.loads(json_bytes.decode('utf-8'))
        texts = []
        def traverse(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)
            elif isinstance(obj, str):
                texts.append(obj)
        traverse(data)
        return "\n".join(texts).strip()

    @staticmethod
    def extract_from_text(resume_data: Dict) -> str:
        """
        Extract text from resume data
        Handles PDF, DOCX, JSON, and plain text formats
        """
        # Already extracted text
        if 'resume_text' in resume_data and resume_data['resume_text']:
            return resume_data['resume_text']

        # JSON resumes (misnamed as .docx)
        if 'json_content' in resume_data and resume_data['json_content']:
            json_bytes = base64.b64decode(resume_data['json_content'])
            return TextExtractor.extract_from_json(json_bytes)

        # Base64 PDFs
        if 'pdf_content' in resume_data and resume_data['pdf_content']:
            pdf_bytes = base64.b64decode(resume_data['pdf_content'])
            return TextExtractor.extract_from_pdf(pdf_bytes)

        # Base64 DOCX
        if 'docx_content' in resume_data and resume_data['docx_content']:
            docx_bytes = base64.b64decode(resume_data['docx_content'])
            return TextExtractor.extract_from_docx(docx_bytes)

        # Generic content with format
        if 'content' in resume_data and 'format' in resume_data:
            content_bytes = base64.b64decode(resume_data['content'])
            fmt = resume_data['format'].lower()
            if fmt == 'pdf':
                return TextExtractor.extract_from_pdf(content_bytes)
            elif fmt == 'docx':
                return TextExtractor.extract_from_docx(content_bytes)
            elif fmt == 'json':
                return TextExtractor.extract_from_json(content_bytes)

        # Fallback to raw text
        return resume_data.get('text', '')


class CategoryExtractor:
    """Intelligently extract semantic job category from resume data"""
    
    CATEGORY_KEYWORDS = {
        'Software Developer': ['developer', 'programmer', 'software engineer', 'coder', 'dev'],
        'Data Scientist': ['data scientist', 'machine learning', 'ml engineer', 'data engineer'],
        'DevOps': ['devops', 'sre', 'infrastructure', 'cloud engineer', 'kubernetes', 'docker'],
        'Frontend Developer': ['frontend', 'react', 'angular', 'vue', 'ui developer', 'web developer'],
        'Backend Developer': ['backend', 'server', 'api developer', 'nodejs', 'python backend'],
        'Full Stack Developer': ['full stack', 'fullstack'],
        'Data Analyst': ['data analyst', 'analytics', 'bi analyst', 'business analyst'],
        'QA Engineer': ['qa', 'quality assurance', 'test engineer', 'tester'],
        'DevOps Engineer': ['devops', 'cloud', 'infrastructure'],
        'Product Manager': ['product manager', 'pm', 'product lead'],
        'Project Manager': ['project manager', 'pmp', 'scrum master'],
        'System Administrator': ['sysadmin', 'system admin', 'linux admin'],
        'Network Engineer': ['network', 'networking', 'network admin'],
        'Security Engineer': ['security', 'cybersecurity', 'infosec'],
        'AI Engineer': ['ai engineer', 'artificial intelligence', 'deep learning'],
        'Mobile Developer': ['mobile developer', 'android', 'ios', 'react native'],
        'Database Administrator': ['dba', 'database', 'database admin'],
    }

    @staticmethod
    def extract_category(resume_data: Dict) -> str:
        """
        Extract semantic job category from resume data.
        Checks multiple sources in priority order.
        """
        raw_data = resume_data.get('raw_data', {})
        
        # Priority 1: Current job title from experience
        if 'experience' in raw_data and raw_data['experience']:
            current_job = raw_data['experience'][0]
            if current_job.get('title') and current_job['title'] != 'Unknown':
                title = current_job['title'].strip()
                return CategoryExtractor._normalize_category(title)
        
        # Priority 2: Education field
        if 'education' in raw_data and raw_data['education']:
            edu_field = raw_data['education'][0].get('degree', {}).get('field', '')
            if edu_field and edu_field != 'Unknown':
                return CategoryExtractor._normalize_category(edu_field)
        
        # Priority 3: Extract from personal summary
        summary = raw_data.get('personal_info', {}).get('summary', '')
        if summary and summary != 'Unknown':
            category = CategoryExtractor._extract_from_summary(summary)
            if category:
                return category
        
        # Priority 4: Extract from technical skills
        skills = raw_data.get('skills', {}).get('technical', {})
        category = CategoryExtractor._extract_from_skills(skills)
        if category:
            return category
        
        # Fallback
        return 'Software Developer'

    @staticmethod
    def _normalize_category(text: str) -> str:
        """
        Normalize and map job title to standard categories.
        Handles variations and returns clean category name.
        """
        if not text or text == 'Unknown':
            return 'Software Developer'
        
        text_lower = text.lower().strip()
        
        # Check against known categories
        for category, keywords in CategoryExtractor.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        # If no match, return the original title cleaned up
        # Remove extra whitespace and capitalize properly
        return ' '.join(word.capitalize() for word in text_lower.split())

    @staticmethod
    def _extract_from_summary(summary: str) -> str:
        """
        Extract category keywords from personal summary.
        """
        summary_lower = summary.lower()
        
        # Look for strong role indicators
        for category, keywords in CategoryExtractor.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in summary_lower:
                    return category
        
        return None

    @staticmethod
    def _extract_from_skills(skills: Dict) -> str:
        """
        Extract category from technical skills distribution.
        """
        if not skills:
            return None
        
        # Collect all skills
        all_skills = []
        
        # Programming languages
        prog_langs = skills.get('programming_languages', [])
        if prog_langs:
            all_skills.extend([s.get('name', '').lower() for s in prog_langs if isinstance(s, dict)])
        
        # Frameworks
        frameworks = skills.get('frameworks', [])
        if frameworks:
            all_skills.extend([s.get('name', '').lower() for s in frameworks if isinstance(s, dict)])
        
        # Databases
        databases = skills.get('databases', [])
        if databases:
            all_skills.extend([s.get('name', '').lower() for s in databases if isinstance(s, dict)])
        
        # Cloud
        cloud = skills.get('cloud', [])
        if cloud:
            all_skills.extend([s.get('name', '').lower() for s in cloud if isinstance(s, dict)])
        
        skills_str = ' '.join(all_skills).lower()
        
        # Skill-based category detection
        if any(x in skills_str for x in ['tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy']):
            if any(x in skills_str for x in ['deep learning', 'nlp', 'computer vision']):
                return 'AI Engineer'
            return 'Data Scientist'
        
        if any(x in skills_str for x in ['kubernetes', 'docker', 'ansible', 'terraform', 'jenkins']):
            return 'DevOps Engineer'
        
        if any(x in skills_str for x in ['react', 'angular', 'vue', 'html', 'css', 'javascript']):
            if any(x in skills_str for x in ['nodejs', 'python', 'java', 'go']):
                return 'Full Stack Developer'
            return 'Frontend Developer'
        
        if any(x in skills_str for x in ['nodejs', 'django', 'flask', 'spring', 'dotnet']):
            return 'Backend Developer'
        
        if any(x in skills_str for x in ['mysql', 'postgresql', 'mongodb', 'oracle']):
            return 'Database Administrator'
        
        if any(x in skills_str for x in ['android', 'ios', 'swift', 'kotlin', 'react native']):
            return 'Mobile Developer'
        
        return None


def process_resumes(resumes: List[Dict]) -> List[Dict]:
    logger.info(f"Starting text extraction for {len(resumes)} resumes")
    start_time = time.time()
    results = []
    text_extractor = TextExtractor()
    category_extractor = CategoryExtractor()

    for resume in resumes:
        try:
            extracted_text = text_extractor.extract_from_text(resume)
            category = category_extractor.extract_category(resume)
            
            results.append({
                'candidate_id': resume.get('id', None),
                'raw_text': extracted_text,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'extraction_status': 'success',
                'category': category,
                'status': 'extracted'
            })
        except Exception as e:
            logger.error(f"Extraction failed for {resume.get('id', 'unknown')}: {e}")
            results.append({
                'candidate_id': resume.get('id', None),
                'raw_text': '',
                'text_length': 0,
                'word_count': 0,
                'extraction_status': 'failed',
                'error': str(e),
                'category': 'Unknown',
                'status': 'failed'
            })

    duration = time.time() - start_time
    stats = calculate_statistics(results)
    logger.info(format_report("TEXT EXTRACTION", stats, duration))

    checkpoint_file = save_checkpoint(results, 'step2_extraction')
    logger.info(f"Checkpoint saved: {checkpoint_file}")

    return results


def run(input_data: List[Dict]) -> List[Dict]:
    return process_resumes(input_data)


if __name__ == "__main__":
    from step1_fetch_data import run as fetch_data
    resumes = fetch_data(limit=5)
    results = run(resumes)
    
    print(f"\n--- EXTRACTION RESULTS ---")
    for r in results[:2]:
        print(f"Candidate: {r['candidate_id']}")
        print(f"Status: {r['extraction_status']}")
        print(f"Category: {r['category']}")
        print(f"Word Count: {r['word_count']}")
        print(f"Preview: {r['raw_text'][:150]}...")
        print("-" * 60)