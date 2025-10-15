"""
Create Weaviate schema for Resume collection
Run this before running the pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure, VectorDistances

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from src import utils

logger = utils.get_logger(__name__)


def create_resume_schema():
    """Create Resume collection schema in Weaviate"""
    
    url = os.getenv('WEAVIATE_URL')
    api_key = os.getenv('WEAVIATE_API_KEY')
    
    if not url or not api_key:
        logger.error("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env")
        return False
    
    try:
        print("=" * 80)
        print("CREATING WEAVIATE SCHEMA")
        print("=" * 80)
        
        # Connect to Weaviate
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key)
        )
        logger.info("Connected successfully")
        
        collection_name = "Resume"
        
        # Check if collection exists
        if client.collections.exists(collection_name):
            logger.warning(f"Collection '{collection_name}' already exists")
            response = input("Delete and recreate? (yes/no): ").strip().lower()
            
            if response == 'yes':
                logger.info(f"Deleting existing collection '{collection_name}'...")
                client.collections.delete(collection_name)
                logger.info("Deleted successfully")
            else:
                logger.info("Keeping existing collection")
                client.close()
                return True
        
        # Create collection
        logger.info(f"Creating collection '{collection_name}'...")
        
        client.collections.create(
            name=collection_name,
            description="Complete structured resumes with embeddings for AI-powered HR matching",
            
            # Vector configuration - we provide pre-computed embeddings
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            
            # Properties
            properties=[
                # ============= PRIMARY IDENTIFIERS =============
                Property(
                    name="candidate_id",
                    data_type=DataType.TEXT,
                    description="Unique candidate identifier (e.g., CAND_001)",
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False
                ),
                
                # ============= SEARCHABLE CONTENT =============
                Property(
                    name="resume_summary",
                    data_type=DataType.TEXT,
                    description="Full resume text for semantic vector search",
                    skip_vectorization=False,  # This will be vectorized
                    index_filterable=False,
                    index_searchable=True
                ),
                
                # ============= STRUCTURED DATA (JSON) =============
                Property(
                    name="personal_info_json",
                    data_type=DataType.TEXT,
                    description="Personal info as JSON: name, email, phone, location, linkedin, github",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="experience_json",
                    data_type=DataType.TEXT,
                    description="Work experience array as JSON: company, title, dates, responsibilities, technologies",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="education_json",
                    data_type=DataType.TEXT,
                    description="Education array as JSON: degree, institution, dates, gpa, coursework",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="skills_json",
                    data_type=DataType.TEXT,
                    description="Skills object as JSON: programming languages, frameworks, databases, cloud, tools",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="projects_json",
                    data_type=DataType.TEXT,
                    description="Projects array as JSON: name, description, technologies, role, url",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="certifications",
                    data_type=DataType.TEXT,
                    description="Professional certifications (plain text)",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                # ============= METADATA FOR FILTERING =============
                Property(
                    name="category",
                    data_type=DataType.TEXT,
                    description="Job category: Software Developer, Data Scientist, DevOps Engineer, etc.",
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False
                ),
                
                Property(
                    name="experience_level",
                    data_type=DataType.TEXT,
                    description="Experience level: entry, mid, senior",
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False
                ),
                
                Property(
                    name="location",
                    data_type=DataType.TEXT,
                    description="Candidate location: City, Country",
                    skip_vectorization=True,
                    index_filterable=True,
                    index_searchable=False
                ),
                
                Property(
                    name="years_of_experience",
                    data_type=DataType.INT,
                    description="Total years of professional experience",
                    index_filterable=True,
                    index_searchable=False
                ),
                
                # ============= TECHNICAL METADATA =============
                Property(
                    name="embedding_model",
                    data_type=DataType.TEXT,
                    description="Embedding model used (e.g., BAAI/bge-base-en-v1.5)",
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="embedding_dim",
                    data_type=DataType.INT,
                    description="Embedding dimension (typically 768)",
                    index_filterable=False,
                    index_searchable=False
                ),
                
                Property(
                    name="indexed_at",
                    data_type=DataType.DATE,
                    description="Timestamp when resume was indexed in Weaviate",
                    index_filterable=True,
                    index_searchable=False
                )
            ]
        )
        
        logger.info(f"Successfully created collection '{collection_name}'")
        
        # Verify schema
        logger.info("\nVerifying schema...")
        collection = client.collections.get(collection_name)
        config = collection.config.get()
        
        print("\n" + "=" * 80)
        print("SCHEMA CREATED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nCollection: {collection_name}")
        print(f"Description: {config.description}")
        print(f"\nProperties ({len(config.properties)}):")
        
        for prop in config.properties:
            # Handle different property attributes safely
            data_type = str(prop.data_type).replace('DataType.', '')
            
            # Check vectorization settings (attribute name varies by version)
            if hasattr(prop, 'skip_vectorization'):
                skip = "Skip Vector" if prop.skip_vectorization else "Vectorize"
            elif hasattr(prop, 'vectorize_property_name'):
                skip = "Vectorize" if prop.vectorize_property_name else "Skip Vector"
            else:
                skip = "Unknown"
            
            # Check filter settings
            if hasattr(prop, 'index_filterable'):
                filterable = "Filterable" if prop.index_filterable else ""
            else:
                filterable = ""
            
            print(f"  {prop.name:25} {data_type:15} {skip:12} {filterable}")
        
        print("\n" + "=" * 80)
        print("READY FOR PIPELINE!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run: python pipeline.py --num-resumes 20")
        print("2. Run: python inspect_weaviate.py (to verify data)")
        print("3. Run: python query_resumes.py (to test queries)")
        print("=" * 80)
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to create schema: {e}", exc_info=True)
        return False


def main():
    success = create_resume_schema()
    
    if success:
        print("\n✅ Schema creation successful!")
    else:
        print("\n❌ Schema creation failed. Check logs for details.")


if __name__ == "__main__":
    main()