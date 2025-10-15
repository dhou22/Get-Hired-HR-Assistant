"""
Utility functions for resume processing pipeline
UTF-8 encoding support for Windows environments
"""

import os
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger with UTF-8 encoding support
    Fixes Windows PowerShell emoji encoding issues
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # File handler (UTF-8)
        file_handler = logging.FileHandler('logs/pipeline.log', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        # Console handler (UTF-8)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        # Force UTF-8 encoding for console (Windows fix)
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass  # Fallback gracefully if reconfigure fails
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'data/outputs',
        'logs'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_checkpoint(data: Any, step_name: str, format: str = 'json') -> str:
    """
    Save processing checkpoint with datetime handling
    
    Args:
        data: Data to save
        step_name: Name of processing step
        format: Output format (json only)
    
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"data/processed/{step_name}_{timestamp}.{format}"
    
    if format == 'json':
        # Custom JSON encoder to handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filename


def load_checkpoint(step_name: str) -> Optional[Any]:
    """
    Load most recent checkpoint
    
    Args:
        step_name: Name of processing step
    
    Returns:
        Loaded data or None
    """
    checkpoint_dir = Path('data/processed')
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(f"{step_name}_*.json"))
    
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_statistics(data: List[Dict], status_field: str = 'status') -> Dict:
    """
    Calculate processing statistics
    
    Args:
        data: List of items
        status_field: Field containing status
    
    Returns:
        Statistics dictionary
    """
    if not data:
        return {
            'total_items': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 0.0
        }
    
    total = len(data)
    successful = sum(
        1 for item in data
        if item.get(status_field) == 'success'
        or item.get('extraction_status') == 'success'
        or item.get('embedding_status') == 'success'
    )
    
    failed = total - successful
    
    return {
        'total_items': total,
        'successful': successful,
        'failed': failed,
        'success_rate': round((successful / total) * 100, 2) if total > 0 else 0.0
    }


def format_report(step_name: str, stats: Dict, duration: float) -> str:
    """Format processing report"""
    report = [
        "=" * 60,
        f"STEP: {step_name}",
        "=" * 60,
        f"Total Items: {stats['total_items']}",
        f"Successful: {stats['successful']}",
        f"Failed: {stats['failed']}",
        f"Success Rate: {stats['success_rate']}%",
        f"Duration: {duration:.2f}s",
        "=" * 60
    ]
    return "\n".join(report)


def validate_resume_data(resume: Dict, required_fields: List[str] = None) -> bool:
    """Validate resume data structure"""
    if not isinstance(resume, dict):
        return False
    
    if required_fields:
        return all(field in resume for field in required_fields)
    
    return bool(resume.get('candidate_id') or resume.get('id'))


def get_env_variable(name: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with validation
    
    Args:
        name: Variable name
        default: Default value
        required: Whether required
    
    Returns:
        Variable value
    """
    value = os.getenv(name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{name}' not found")
    
    return value


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def count_tokens_in_text(text: str) -> int:
    """Simple token counter (word-based)"""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def cleanup_old_checkpoints(days: int = 7):
    """Remove checkpoint files older than specified days"""
    checkpoint_dir = Path('data/processed')
    if not checkpoint_dir.exists():
        return
    
    import time
    cutoff_time = time.time() - (days * 86400)
    
    removed_count = 0
    for checkpoint in checkpoint_dir.glob('*.json'):
        if checkpoint.stat().st_mtime < cutoff_time:
            checkpoint.unlink()
            removed_count += 1
    
    if removed_count > 0:
        logger = get_logger(__name__)
        logger.info(f"Removed {removed_count} old checkpoint files")