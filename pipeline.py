"""
Simplified Resume Processing Pipeline for HR Matching
Stores complete structured resume data in Weaviate for AI agent retrieval
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src import utils
from src.step1_fetch_data import run as step1_fetch
from src.step2_extract_text import run as step2_extract

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from sentence_transformers import SentenceTransformer
import torch

logger = utils.get_logger(__name__)


class WeaviateResumeStore:
    """
    Manages Weaviate connection and structured resume storage
    Optimized for AI agent retrieval with full JSON data
    """
    
    def __init__(self, url: str = None, api_key: str = None):
        self.url = url or os.getenv('WEAVIATE_URL')
        self.api_key = api_key or os.getenv('WEAVIATE_API_KEY')
        self.client = None
        self.collection_name = "Resume"
        
        if not self.url:
            raise ValueError("WEAVIATE_URL not found in environment variables")
    
    def connect(self):
        """Establish connection to Weaviate Cloud"""
        try:
            logger.info(f"Connecting to Weaviate...")
            if not self.api_key:
                raise ValueError("API key is required for Weaviate Cloud connection")
            
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key)
            )
            
            logger.info("Successfully connected to Weaviate")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_schema(self):
        """
        Create Weaviate schema for complete resume storage
        Schema designed for AI agent semantic search + structured filtering
        """
        try:
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            self.client.collections.create(
                name=self.collection_name,
                description="Complete structured resumes with embeddings for AI-powered HR matching",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    # Primary Identifiers
                    Property(
                        name="candidate_id",
                        data_type=DataType.TEXT,
                        description="Unique candidate identifier",
                        skip_vectorization=True
                    ),
                    
                    # Searchable Resume Content
                    Property(
                        name="resume_summary",
                        data_type=DataType.TEXT,
                        description="Condensed resume summary for vector search"
                    ),
                    
                    # Structured Data (JSON)
                    Property(
                        name="personal_info",
                        data_type=DataType.OBJECT,
                        description="Personal information (name, email, location, etc.)"
                    ),
                    Property(
                        name="experience",
                        data_type=DataType.OBJECT_ARRAY,
                        description="Work experience history"
                    ),
                    Property(
                        name="education",
                        data_type=DataType.OBJECT_ARRAY,
                        description="Educational background"
                    ),
                    Property(
                        name="skills",
                        data_type=DataType.OBJECT,
                        description="Technical and soft skills"
                    ),
                    Property(
                        name="projects",
                        data_type=DataType.OBJECT_ARRAY,
                        description="Projects portfolio"
                    ),
                    Property(
                        name="certifications",
                        data_type=DataType.TEXT,
                        description="Professional certifications"
                    ),
                    
                    # Metadata for Filtering
                    Property(
                        name="category",
                        data_type=DataType.TEXT,
                        description="Job category (e.g., Software Developer, Data Scientist)",
                        skip_vectorization=True
                    ),
                    Property(
                        name="experience_level",
                        data_type=DataType.TEXT,
                        description="Experience level (entry, mid, senior)",
                        skip_vectorization=True
                    ),
                    Property(
                        name="location",
                        data_type=DataType.TEXT,
                        description="Candidate location (city, country)",
                        skip_vectorization=True
                    ),
                    Property(
                        name="years_of_experience",
                        data_type=DataType.INT,
                        description="Total years of professional experience"
                    ),
                    
                    # Technical Metadata
                    Property(
                        name="embedding_model",
                        data_type=DataType.TEXT,
                        description="Embedding model used",
                        skip_vectorization=True
                    ),
                    Property(
                        name="embedding_dim",
                        data_type=DataType.INT,
                        description="Embedding dimension"
                    ),
                    Property(
                        name="indexed_at",
                        data_type=DataType.DATE,
                        description="Timestamp when resume was indexed"
                    )
                ]
            )
            
            logger.info(f"Created collection '{self.collection_name}' with structured schema")
        
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def store_resumes(self, resumes: List[Dict]) -> int:
        """
        Store complete structured resumes in Weaviate
        
        Args:
            resumes: List of resume dictionaries with embeddings
        
        Returns:
            Number of resumes successfully stored
        """
        if not resumes:
            logger.warning("No resumes to store")
            return 0
        
        try:
            collection = self.client.collections.get(self.collection_name)
            stored_count = 0
            failed_count = 0
            
            logger.info(f"Storing {len(resumes)} resumes in Weaviate...")
            
            with collection.batch.dynamic() as batch:
                for resume in resumes:
                    try:
                        if not resume.get('embedding') or resume.get('embedding_status') != 'success':
                            logger.debug(f"Skipping {resume.get('candidate_id')} - no valid embedding")
                            failed_count += 1
                            continue
                        
                        raw_data = resume.get('raw_data', {})
                        
                        # Extract key metadata
                        personal_info = raw_data.get('personal_info', {})
                        location_obj = personal_info.get('location', {})
                        location_str = f"{location_obj.get('city', 'Unknown')}, {location_obj.get('country', 'Unknown')}"
                        
                        # Determine experience level
                        experience = raw_data.get('experience', [])
                        exp_level = experience[0].get('level', 'entry') if experience else 'entry'
                        
                        # Calculate years of experience (simple estimation)
                        years_exp = len(experience) if experience else 0
                        
                        properties = {
                            "candidate_id": resume.get('candidate_id', ''),
                            "resume_summary": resume.get('resume_text', ''),
                            "personal_info": personal_info,
                            "experience": raw_data.get('experience', []),
                            "education": raw_data.get('education', []),
                            "skills": raw_data.get('skills', {}),
                            "projects": raw_data.get('projects', []),
                            "certifications": raw_data.get('certifications', ''),
                            "category": resume.get('category', 'Unknown'),
                            "experience_level": exp_level,
                            "location": location_str,
                            "years_of_experience": years_exp,
                            "embedding_model": resume.get('embedding_model', 'BAAI/bge-base-en-v1.5'),
                            "embedding_dim": resume.get('embedding_dim', 768),
                            "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
                        }
                        
                        batch.add_object(
                            properties=properties,
                            vector=resume['embedding']
                        )
                        
                        stored_count += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to store {resume.get('candidate_id', 'unknown')}: {e}")
                        failed_count += 1
            
            logger.info(f"Stored {stored_count} resumes in Weaviate")
            if failed_count > 0:
                logger.warning(f"Failed to store {failed_count} resumes")
            
            return stored_count
        
        except Exception as e:
            logger.error(f"Failed to store resumes in Weaviate: {e}", exc_info=True)
            raise
    
    def close(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")


class ResumeEmbedder:
    """Generate embeddings for complete resumes"""
    
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully (dimension: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, resumes: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for complete resumes
        
        Args:
            resumes: List of resume dictionaries with resume_text
        
        Returns:
            Resumes with embeddings added
        """
        if not resumes:
            return []
        
        logger.info(f"Generating embeddings for {len(resumes)} resumes")
        
        # Extract resume texts
        valid_resumes = []
        valid_texts = []
        
        for resume in resumes:
            text = resume.get('resume_text', '')
            if text and isinstance(text, str) and len(text.strip()) > 0:
                valid_resumes.append(resume)
                valid_texts.append(text)
        
        if not valid_texts:
            logger.error("No valid resume text found")
            return resumes
        
        logger.info(f"Processing {len(valid_texts)} valid resumes")
        
        try:
            # Add instruction prefix for better retrieval
            prefixed_texts = [
                f"Represent this resume for job matching: {text[:1000]}"  # Truncate to first 1000 chars
                for text in valid_texts
            ]
            
            # Generate embeddings
            logger.info("Encoding resumes to embeddings...")
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Attach embeddings
            for i, resume in enumerate(valid_resumes):
                resume['embedding'] = embeddings[i].tolist()
                resume['embedding_dim'] = len(embeddings[i])
                resume['embedding_status'] = 'success'
                resume['embedding_model'] = self.model_name
            
            # Mark invalid resumes
            for resume in resumes:
                if resume not in valid_resumes:
                    resume['embedding'] = None
                    resume['embedding_dim'] = 0
                    resume['embedding_status'] = 'failed'
                    resume['embedding_error'] = 'No valid text content'
            
            return resumes
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise


class SimplifiedResumePipeline:
    """
    Simplified pipeline: Fetch → Extract → Embed → Store
    No chunking, no extensive preprocessing
    """
    
    def __init__(self, store_in_weaviate: bool = True):
        self.store_in_weaviate = store_in_weaviate
        self.results = {}
        utils.ensure_directories()
    
    def run(self, num_resumes: int = 50, offset: int = 0,
            model_name: str = 'BAAI/bge-base-en-v1.5',
            batch_size: int = 32) -> dict:
        """
        Execute simplified pipeline
        
        Args:
            num_resumes: Number of resumes to process
            offset: Starting offset for dataset
            model_name: Embedding model name
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()
        
        logger.info("=" * 80)
        logger.info("SIMPLIFIED RESUME PIPELINE - START")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  - Resumes to process: {num_resumes}")
        logger.info(f"  - Store in Weaviate: {self.store_in_weaviate}")
        logger.info(f"  - Embedding model: {model_name}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Fetch structured resumes
            logger.info("\n[STEP 1/3] Fetching Resume Data")
            self.results['fetched'] = step1_fetch(num_resumes, offset)
            logger.info(f"Fetched {len(self.results['fetched'])} resumes")
            
            # Step 2: Extract text representation
            logger.info("\n[STEP 2/3] Extracting Text Representation")
            self.results['extracted'] = step2_extract(self.results['fetched'])
            
            # Merge raw_data back into extracted results
            for extracted, fetched in zip(self.results['extracted'], self.results['fetched']):
                extracted['raw_data'] = fetched.get('raw_data', {})
                extracted['resume_text'] = fetched.get('resume_text', '')
            
            logger.info(f"Extracted text from {len(self.results['extracted'])} resumes")
            
            # Step 3: Generate embeddings
            logger.info("\n[STEP 3/3] Generating Embeddings")
            embedder = ResumeEmbedder(model_name, batch_size)
            self.results['embedded'] = embedder.generate_embeddings(self.results['extracted'])
            
            successful_embeddings = sum(
                1 for r in self.results['embedded'] 
                if r.get('embedding_status') == 'success'
            )
            logger.info(f"Generated {successful_embeddings} embeddings")
            
            # Step 4: Store in Weaviate
            if self.store_in_weaviate and self.results['embedded']:
                logger.info("\n[STEP 4] Storing in Weaviate")
                self._store_in_weaviate(self.results['embedded'])
            
            pipeline_duration = time.time() - pipeline_start
            
            # Final Report
            self._print_final_report(pipeline_duration)
            
            return self.results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _store_in_weaviate(self, resumes: List[Dict]):
        """Store resumes in Weaviate"""
        store = WeaviateResumeStore()
        
        try:
            if not store.connect():
                logger.error("Failed to connect to Weaviate")
                return
            
            store.create_schema()
            stored_count = store.store_resumes(resumes)
            
            logger.info(f"Successfully stored {stored_count} resumes in Weaviate")
        
        except Exception as e:
            logger.error(f"Weaviate storage failed: {e}")
        
        finally:
            store.close()
    
    def _print_final_report(self, duration: float):
        """Print pipeline summary"""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("=" * 80)
        
        if 'fetched' in self.results:
            logger.info(f"Resumes Fetched: {len(self.results['fetched'])}")
        
        if 'extracted' in self.results:
            logger.info(f"Text Extracted: {len(self.results['extracted'])}")
        
        if 'embedded' in self.results:
            successful = sum(
                1 for r in self.results['embedded'] 
                if r.get('embedding_status') == 'success'
            )
            logger.info(f"Embeddings Generated: {successful}/{len(self.results['embedded'])}")
        
        logger.info(f"\nTotal Duration: {utils.format_duration(duration)}")
        logger.info("=" * 80)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Simplified Resume Processing Pipeline')
    parser.add_argument('--num-resumes', type=int, default=50,
                       help='Number of resumes to process')
    parser.add_argument('--offset', type=int, default=0,
                       help='Starting offset in dataset')
    parser.add_argument('--no-weaviate', action='store_true',
                       help='Disable Weaviate storage')
    parser.add_argument('--model', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Embedding model name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    
    args = parser.parse_args()
    
    pipeline = SimplifiedResumePipeline(store_in_weaviate=not args.no_weaviate)
    
    results = pipeline.run(
        num_resumes=args.num_resumes,
        offset=args.offset,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Save final output
    final_output = {
        'metadata': {
            'num_resumes_processed': args.num_resumes,
            'embedding_model': args.model,
            'stored_in_weaviate': not args.no_weaviate,
            'timestamp': datetime.now().isoformat()
        },
        'resumes': results.get('embedded', [])
    }
    
    output_file = utils.save_checkpoint(final_output, 'final_resumes')
    logger.info(f"\nFinal output saved to: {output_file}")


if __name__ == "__main__":
    main()