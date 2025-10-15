"""
Flask API Server for Resume Processing Pipeline
Exposes pipeline functionality to n8n via REST API
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pipeline import ResumePipeline
import os
import json
import logging
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n cloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'resume-processing-pipeline',
        'version': '1.0.0'
    })

@app.route('/process-resumes', methods=['POST'])
def process_resumes():
    """
    Main endpoint for processing resumes
    
    Request body:
    {
        "num_resumes": 100,
        "offset": 0,
        "model": "BAAI/bge-base-en-v1.5",
        "batch_size": 32,
        "start_step": 1
    }
    
    Response:
    {
        "status": "success",
        "metadata": {...},
        "chunks_with_embeddings": [...]
    }
    """
    try:
        data = request.json or {}
        
        logger.info(f"Received processing request: {data}")
        
        # Initialize pipeline
        pipeline = ResumePipeline(
            start_step=data.get('start_step', 1),
            use_checkpoints=data.get('use_checkpoints', True)
        )
        
        # Run pipeline
        results = pipeline.run(
            num_resumes=data.get('num_resumes', 100),
            offset=data.get('offset', 0),
            model_name=data.get('model', 'BAAI/bge-base-en-v1.5'),
            batch_size=data.get('batch_size', 32)
        )
        
        # Prepare response
        response_data = {
            'status': 'success',
            'metadata': {
                'num_resumes_processed': data.get('num_resumes', 100),
                'total_chunks': len(results.get('step6', [])),
                'embedding_model': data.get('model', 'BAAI/bge-base-en-v1.5'),
                'embedding_dimension': results['step6'][0]['embedding_dim'] if results.get('step6') else 0
            },
            'chunks_with_embeddings': results.get('step6', [])
        }
        
        logger.info(f"Processing complete: {len(results.get('step6', []))} chunks generated")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/process-step/<int:step>', methods=['POST'])
def process_single_step(step):
    """
    Process a single pipeline step
    Useful for debugging or partial processing
    """
    try:
        data = request.json or {}
        input_data = data.get('input_data', [])
        
        if step == 1:
            from src.step1_fetch_data import run
            result = run(data.get('limit', 100), data.get('offset', 0))
        elif step == 2:
            from src.step2_extract_text import run
            result = run(input_data)
        elif step == 3:
            from src.step3_clean_text import run
            result = run(input_data)
        elif step == 4:
            from src.step4_anonymize_pii import run
            result = run(input_data)
        elif step == 5:
            from src.step5_semantic_chunking import run
            result = run(input_data)
        elif step == 6:
            from src.step6_generate_embeddings import run
            result = run(input_data)
        else:
            return jsonify({'status': 'error', 'error': 'Invalid step'}), 400
        
        return jsonify({
            'status': 'success',
            'step': step,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints"""
    try:
        checkpoint_dir = Path('data/processed')
        checkpoints = {}
        
        for step in ['step1_fetch', 'step2_extraction', 'step3_cleaning', 
                     'step4_anonymization', 'step5_chunking', 'step6_embeddings']:
            files = list(checkpoint_dir.glob(f"{step}_*.json"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                checkpoints[step] = {
                    'filename': latest.name,
                    'timestamp': latest.stat().st_mtime,
                    'size_mb': latest.stat().st_size / (1024 * 1024)
                }
        
        return jsonify({
            'status': 'success',
            'checkpoints': checkpoints
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/checkpoint/<step_name>', methods=['GET'])
def get_checkpoint(step_name):
    """Download specific checkpoint"""
    try:
        checkpoint_dir = Path('data/processed')
        files = list(checkpoint_dir.glob(f"{step_name}_*.json"))
        
        if not files:
            return jsonify({
                'status': 'error',
                'error': f'No checkpoint found for {step_name}'
            }), 404
        
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return send_file(latest, mimetype='application/json')
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Production: Use gunicorn
    # Development: Use Flask dev server
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Resume Processing API on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )