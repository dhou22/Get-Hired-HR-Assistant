"""
Cleanup script to delete old Weaviate collections and start fresh
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth

# Load environment
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from src import utils

logger = utils.get_logger(__name__)


def cleanup_weaviate():
    """Delete old collections from Weaviate"""
    
    url = os.getenv('WEAVIATE_URL')
    api_key = os.getenv('WEAVIATE_API_KEY')
    
    if not url or not api_key:
        logger.error("Missing Weaviate credentials in .env file")
        return False
    
    try:
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key)
        )
        
        logger.info("Connected successfully")
        
        # List all collections
        collections = client.collections.list_all()
        logger.info(f"Found {len(collections)} collection(s)")
        
        for collection_name in collections:
            logger.info(f"  - {collection_name}")
        
        # Delete old collections
        collections_to_delete = ['ResumeChunk', 'Resume']
        
        for collection_name in collections_to_delete:
            if client.collections.exists(collection_name):
                logger.info(f"Deleting collection: {collection_name}")
                client.collections.delete(collection_name)
                logger.info(f"  Deleted: {collection_name}")
            else:
                logger.info(f"  Collection '{collection_name}' does not exist")
        
        client.close()
        logger.info("\nCleanup complete! You can now run the new pipeline.")
        logger.info("Run: python pipeline.py")
        
        return True
    
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False


def verify_schema():
    """Verify the new schema after cleanup"""
    
    url = os.getenv('WEAVIATE_URL')
    api_key = os.getenv('WEAVIATE_API_KEY')
    
    try:
        logger.info("\nVerifying Weaviate state...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key)
        )
        
        collections = client.collections.list_all()
        
        if collections:
            logger.info(f"Remaining collections: {list(collections.keys())}")
        else:
            logger.info("No collections found - ready for fresh start!")
        
        client.close()
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    print("=" * 80)
    print("WEAVIATE CLEANUP UTILITY")
    print("=" * 80)
    print("\nThis script will DELETE all resume collections from Weaviate.")
    print("Collections to be deleted:")
    print("  - ResumeChunk (old chunked data)")
    print("  - Resume (if exists)")
    print("\n" + "=" * 80)
    
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\nStarting cleanup...\n")
        
        if cleanup_weaviate():
            print("\n" + "=" * 80)
            print("SUCCESS!")
            print("=" * 80)
            print("\nNext steps:")
            print("1. Run: python pipeline.py --num-resumes 20")
            print("2. Check Weaviate console to verify new structured schema")
            print("=" * 80)
        else:
            print("\nCleanup failed. Check logs for details.")
    else:
        print("\nCleanup cancelled.")


if __name__ == "__main__":
    main()