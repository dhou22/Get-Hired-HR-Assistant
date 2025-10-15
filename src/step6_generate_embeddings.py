"""
Step 6: Generate Embeddings (UPDATED)
Technique: Dense vector embeddings via fine-tuned transformer
Model: BAAI/bge-base-en-v1.5 (768 dimensions)
Reference: Phase 1 document - Section 3.4
"""

import time
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer

try:
    from .utils import get_logger, save_checkpoint, calculate_statistics, format_report
except ImportError:
    from utils import get_logger, save_checkpoint, calculate_statistics, format_report

logger = get_logger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', batch_size: int = 32):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for all chunks
        
        Processing:
        - Batch size: 32 (optimized for memory)
        - Normalization: L2 (cosine similarity ready)
        - Instruction prefix: Improves retrieval quality
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Extract texts
        texts = [chunk.get('chunk_text', '') for chunk in chunks]
        
        # Validate that texts exist
        valid_chunks = []
        valid_texts = []
        for chunk, text in zip(chunks, texts):
            if text and isinstance(text, str) and len(text.strip()) > 0:
                valid_chunks.append(chunk)
                valid_texts.append(text)
        
        if not valid_texts:
            logger.error("No valid text content found in chunks")
            return []
        
        logger.info(f"Processing {len(valid_texts)} valid chunks (skipped {len(chunks) - len(valid_texts)} invalid)")
        
        try:
            # Add instruction prefix (BGE best practice for retrieval)
            prefixed_texts = [
                f"Represent this resume section for retrieval: {text}"
                for text in valid_texts
            ]
            
            # Generate embeddings in batches
            logger.info("Encoding texts to embeddings...")
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Attach embeddings to chunks
            for i, chunk in enumerate(valid_chunks):
                chunk['embedding'] = embeddings[i].tolist()
                chunk['embedding_dim'] = len(embeddings[i])
                chunk['embedding_status'] = 'success'
            
            # Add failed status to invalid chunks
            for chunk in chunks:
                if chunk not in valid_chunks:
                    chunk['embedding'] = None
                    chunk['embedding_dim'] = 0
                    chunk['embedding_status'] = 'failed'
                    chunk['embedding_error'] = 'No valid text content'
            
            return chunks
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            # Mark all chunks as failed
            for chunk in chunks:
                chunk['embedding'] = None
                chunk['embedding_dim'] = 0
                chunk['embedding_status'] = 'failed'
                chunk['embedding_error'] = str(e)
            raise


def process_chunks(chunks: List[Dict], model_name: str = 'BAAI/bge-base-en-v1.5',
                   batch_size: int = 32) -> List[Dict]:
    """
    Process all chunks for embedding generation
    
    Args:
        chunks: List of chunk dictionaries from Step 5
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        
    Returns:
        List of chunks with embeddings
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation")
        return []
    
    logger.info(f"Starting embedding generation for {len(chunks)} chunks")
    start_time = time.time()
    
    try:
        generator = EmbeddingGenerator(model_name, batch_size)
        chunks_with_embeddings = generator.generate_embeddings(chunks)
        
        duration = time.time() - start_time
        
        # Calculate statistics
        total = len(chunks_with_embeddings)
        successful = sum(1 for c in chunks_with_embeddings if c.get('embedding_status') == 'success')
        failed = total - successful
        success_rate = (successful / total * 100) if total > 0 else 0
        avg_time_per_chunk = round(duration / total, 4) if total > 0 else 0
        
        stats = {
            'total_items': total,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'avg_time_per_chunk': avg_time_per_chunk
        }
        
        logger.info("=" * 60)
        logger.info("STEP: EMBEDDING GENERATION")
        logger.info("=" * 60)
        logger.info(f"Total Items: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Avg Time per Chunk: {avg_time_per_chunk}s")
        logger.info(f"Total Duration: {duration:.2f}s")
        logger.info("=" * 60)
        
        # Save checkpoint
        checkpoint_file = save_checkpoint(chunks_with_embeddings, 'step6_embeddings')
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        return chunks_with_embeddings
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise


def run(input_data: List[Dict], model_name: str = 'BAAI/bge-base-en-v1.5',
        batch_size: int = 32) -> List[Dict]:
    """Execute Step 6: Generate embeddings"""
    return process_chunks(input_data, model_name, batch_size)


if __name__ == "__main__":
    # Test with sample data
    test_chunks = [
        {
            'chunk_id': 'TEST_001_000',
            'candidate_id': 'TEST_001',
            'section_type': 'Contact Information',
            'chunk_text': 'John Doe Senior Python Developer located in San Francisco California with 5 years of experience in full-stack development microservices and cloud architecture',
            'position': 0,
            'token_count': 23,
            'pii_mapping': {},
            'encrypted_pii': '',
            'status': 'success'
        },
        {
            'chunk_id': 'TEST_001_001',
            'candidate_id': 'TEST_001',
            'section_type': 'Work Experience',
            'chunk_text': 'Led development of microservices architecture using Python and Kubernetes. Managed team of 5 developers across multiple projects. Improved system performance by 40% through optimization and caching strategies. Implemented CI/CD pipelines using Jenkins and Docker for automated testing and deployment.',
            'position': 1,
            'token_count': 51,
            'pii_mapping': {},
            'encrypted_pii': '',
            'status': 'success'
        },
        {
            'chunk_id': 'TEST_001_002',
            'candidate_id': 'TEST_001',
            'section_type': 'Skills',
            'chunk_text': 'Programming Languages: Python JavaScript Go. Frameworks: Django React FastAPI. Databases: PostgreSQL MongoDB. Cloud Platforms: AWS Google Cloud Docker Kubernetes. Tools: Git Jenkins GitHub CI/CD Linux.',
            'position': 2,
            'token_count': 35,
            'pii_mapping': {},
            'encrypted_pii': '',
            'status': 'success'
        }
    ]
    
    chunks_with_embeddings = run(test_chunks)
    
    print(f"\n--- EMBEDDING RESULTS ---")
    print(f"Total chunks processed: {len(chunks_with_embeddings)}")
    
    successful = sum(1 for c in chunks_with_embeddings if c.get('embedding_status') == 'success')
    failed = len(chunks_with_embeddings) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if chunks_with_embeddings:
        for chunk in chunks_with_embeddings:
            if chunk.get('embedding_status') == 'success':
                print(f"\n✅ Chunk: {chunk['chunk_id']}")
                print(f"   Section: {chunk['section_type']}")
                print(f"   Tokens: {chunk['token_count']}")
                print(f"   Embedding Dimension: {chunk['embedding_dim']}")
                print(f"   Embedding Preview: {chunk['embedding'][:5]}...")
                print("-" * 60)
            else:
                print(f"\n❌ Chunk: {chunk['chunk_id']}")
                print(f"   Status: {chunk.get('embedding_status')}")
                print(f"   Error: {chunk.get('embedding_error', 'Unknown error')}")
                print("-" * 60)