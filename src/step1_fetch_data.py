"""
Step 1: Fetch Resume Data from HuggingFace Dataset
Technique: Official HuggingFace Datasets API with error handling and logging
Reference: Phase 1 document - Section 2
"""

import os
import time
from typing import Dict, List
from dotenv import load_dotenv

# Local utilities
try:
    from .utils import get_logger, save_checkpoint, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, format_report

# Load environment variables
load_dotenv()

# Import Hugging Face Datasets library
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
logger = get_logger(__name__)

class HuggingFaceDataFetcher:
    """
    Fetch resume data from a Hugging Face dataset using the `datasets` library.
    """

    def __init__(self):
        self.dataset_name = os.getenv("HUGGINGFACE_DATASET_NAME", "datasetmaster/resumes")
        self.split = os.getenv("HUGGINGFACE_SPLIT", "train")
        self.token = os.getenv("HUGGINGFACE_API_TOKEN")

    def _validate_resume(self, resume: Dict) -> bool:
        """
        Validate if a resume has minimum required data.
        Returns True if resume is valid, False otherwise.
        """
        # Check if resume has at least some content
        has_content = (
            resume.get("personal_info", {}).get("summary") or
            resume.get("experience") or
            resume.get("skills")
        )
        return bool(has_content)

    def fetch_resumes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch resume data using Hugging Face `datasets` library.

        Args:
            limit: Number of resumes to fetch.
            offset: Starting offset for pagination.

        Returns:
            List of standardized resume dictionaries.
        """
        logger.info(f"Fetching {limit} resumes from HuggingFace dataset '{self.dataset_name}' (split: {self.split})")

        start_time = time.time()
        successful = 0
        failed = 0
        processed_resumes = []

        try:
            # ✅ FIX: use 'token=' instead of 'use_auth_token='
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                token=self.token if self.token else None
            )

            # Apply pagination manually
            end_idx = min(offset + limit, len(dataset))
            subset = dataset.select(range(offset, end_idx))

            for idx, resume in enumerate(subset):
                try:
                    # Validate resume data
                    if not self._validate_resume(resume):
                        failed += 1
                        continue

                    # Extract useful text representation
                    resume_text = self._extract_resume_text(resume)

                    processed_resume = {
                        "id": f"CAND_{offset + idx + 1:03d}",
                        "raw_data": resume,
                        "resume_text": resume_text,
                        "category": resume.get("personal_info", {}).get("summary", "Unknown")[:100],
                        "fetch_timestamp": time.time(),
                        "status": "success"
                    }
                    processed_resumes.append(processed_resume)
                    successful += 1

                except Exception as e:
                    logger.warning(f"Failed to process resume at index {offset + idx}: {e}")
                    failed += 1
                    continue

            duration = time.time() - start_time

            # Calculate statistics
            total = successful + failed
            success_rate = (successful / total * 100) if total > 0 else 0

            # Log report
            logger.info("=" * 60)
            logger.info("STEP: FETCH DATA")
            logger.info("=" * 60)
            logger.info(f"Total Items: {total}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info("=" * 60)

            # Save checkpoint
            checkpoint_file = save_checkpoint(processed_resumes, "step1_fetch")
            logger.info(f"Checkpoint saved: {checkpoint_file}")

            return processed_resumes

        except Exception as e:
            logger.error(f"❌ Failed to fetch dataset: {e}")
            raise

    def _extract_resume_text(self, resume: Dict) -> str:
        """
        Extract readable text from resume data structure.
        """
        parts = []

        # Personal info summary
        if "personal_info" in resume:
            summary = resume["personal_info"].get("summary", "")
            if summary and summary != "Unknown":
                parts.append(summary)

        # Experience
        if "experience" in resume and resume["experience"]:
            for exp in resume["experience"]:
                title = exp.get("title", "Unknown")
                company = exp.get("company", "Unknown")
                parts.append(f"{title} at {company}")

        # Skills
        if "skills" in resume:
            skills_section = []
            tech = resume["skills"].get("technical", {})
            if tech:
                for skill_type, skills_list in tech.items():
                    if isinstance(skills_list, list) and skills_list:
                        skill_names = [s.get("name", "") for s in skills_list if isinstance(s, dict)]
                        if skill_names:
                            skills_section.append(f"{skill_type}: {', '.join(skill_names)}")
            if skills_section:
                parts.append("Skills: " + "; ".join(skills_section))

        return "\n".join(parts) if parts else "No content extracted"


def run(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Execute Step 1: Fetch data"""
    fetcher = HuggingFaceDataFetcher()
    return fetcher.fetch_resumes(limit, offset)


if __name__ == "__main__":
    # Test execution
    results = run(limit=10)
    print(f"\n--- SAMPLE OUTPUT ---")
    print(f"Total fetched: {len(results)}")
    if results:
        print(f"Sample ID: {results[0]['id']}")
        print(f"Category: {results[0]['category']}")
        print(f"Text preview:\n{results[0]['resume_text'][:300]}...")