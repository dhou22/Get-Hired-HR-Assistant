# Get Hired - AI RH manager Assistant 
<img width="1074" height="337" alt="image" src="https://github.com/user-attachments/assets/87795166-031a-4d5e-a24d-34a50a2ea956" />


<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Datasets](https://img.shields.io/badge/Data-HuggingFace-ffc107.svg)](https://huggingface.co/)
[![Weaviate Vector DB](https://img.shields.io/badge/VectorDB-Weaviate-1f8a70.svg)](https://weaviate.io/)
[![n8n Automation](https://img.shields.io/badge/Workflow-n8n-FF6B35.svg)](https://n8n.io/)

**Intelligent resume matching system using semantic embeddings and vector search for HR recruitment workflows**

[Overview](#overview) • [Architecture](#architecture) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Configuration](#configuration) • [Contributing](#contributing)

</div>

---

## Overview

Get Hired is an end-to-end resume processing and intelligent candidate matching system designed for HR teams and recruitment platforms. It processes resumes from multiple formats, generates semantic embeddings using state-of-the-art language models, and enables vector-based similarity search to match candidates with job requirements.

### Key Capabilities

- **Multi-format Resume Processing**: Extract text from PDF, DOCX, and JSON documents with automatic format detection
- **Semantic Understanding**: Generate embeddings using BAAI/bge-base-en-v1.5 to capture resume semantics beyond keyword matching
- **Vector Search**: Query Weaviate cloud database to find candidates based on skills, experience, and job fit
- **HR Workflow Integration**: REST API endpoints for seamless n8n automation and custom integrations
- **Structured Resume Storage**: Store complete parsed resume data with rich metadata for filtering and ranking

### Use Cases

- Automated candidate screening and ranking
- Job requirement matching at scale
- Skill-based talent discovery
- Resume database indexing and retrieval
- Multi-criterion candidate filtering (years of experience, location, education, skills)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    n8n Workflow Engine                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Chat Trigger → Input Router → Data Status Check        │ │
│  │ ↓                                                      │ │
│  │ AI Agent (Mistral) ←→ Multiple Tool Calls              │ │
│  │ ├─ Check Data Status                                   │ │
│  │ ├─ Search Resumes (Vector Similarity)                  │ │
│  │ ├─ Smart Candidate Matcher (Scoring)                   │ │
│  │ ├─ Data Extraction Enforcer (Anti-hallucination)       │ │
│  │ └─ Get Contact Info                                    │ │
│  │ ↓                                                      │ │
│  │ Response Validation → Chat Output                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
            ┌───────────────────────────────────┐
            │  Flask REST API (process-resumes) │
            │   Cloudflare Tunnel (Deployment)  │
            └───────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────┐
        │    Pipeline Orchestration            │
        │ Step 1: Fetch from HuggingFace       │
        │ Step 2: Extract & Categorize         │
        │ Step 3: Generate Embeddings          │
        │ Step 4: Store in Weaviate            │
        └──────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────┐
        │  Weaviate Cloud Vector Database      │
        │  - 768-dim embeddings (BGE model)    │
        │  - Structured metadata               │
        │  - GraphQL query interface           │
        └──────────────────────────────────────┘
```

### Data Flow

1. **Input Processing**: Resume data fetched from HuggingFace Datasets API
2. **Text Extraction**: Multi-format extraction with intelligent category detection
3. **Embedding Generation**: BAAI/bge-base-en-v1.5 creates semantic vectors
4. **Vector Storage**: Resumes stored in Weaviate with rich metadata
5. **HR Query**: n8n AI agent processes job requirements and matches candidates
6. **Result Delivery**: Ranked candidate list with contact information

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Weaviate Cloud account and credentials
- HuggingFace API token (for dataset access)
- n8n instance (cloud or self-hosted)
- Cloudflare account (for tunnel access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/get-hired.git
cd get-hired
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials:
# - WEAVIATE_URL
# - WEAVIATE_API_KEY
# - HUGGINGFACE_API_TOKEN
# - HUGGINGFACE_DATASET_NAME
# - HUGGINGFACE_SPLIT
```

### Running Locally with Cloudflare Tunnel

1. Start the Flask API server:
```bash
python api_server.py
```

The server runs on `http://localhost:5000` by default.

2. In a separate terminal, start the Cloudflare tunnel:
```bash
cloudflared tunnel run get-hired-tunnel
```

3. Configure your tunnel in Cloudflare dashboard to point to `localhost:5000`

4. Import the n8n workflow JSON (`get-hired-complete.json`) into your n8n instance

5. Connect the n8n workflow to your Cloudflare tunnel URL

### Processing a Batch of Resumes

```bash
python pipeline.py --num-resumes 100 --model BAAI/bge-base-en-v1.5 --batch-size 32
```

**Options:**
- `--num-resumes`: Number of resumes to process (default: 20)
- `--offset`: Starting offset in dataset (default: 0)
- `--model`: Embedding model name (default: BAAI/bge-base-en-v1.5)
- `--batch-size`: Batch size for embedding generation (default: 32)
- `--no-weaviate`: Skip Weaviate storage

---

## API Reference

### Endpoints

#### Health Check
```http
GET /health
```

Returns service status and version information.

**Response:**
```json
{
  "status": "healthy",
  "service": "resume-processing-pipeline",
  "version": "1.0.0"
}
```

#### Process Resumes
```http
POST /process-resumes
Content-Type: application/json
```

Main endpoint for processing and storing resumes.

**Request Body:**
```json
{
  "num_resumes": 100,
  "offset": 0,
  "model": "BAAI/bge-base-en-v1.5",
  "batch_size": 32,
  "start_step": 1
}
```

**Response:**
```json
{
  "status": "success",
  "metadata": {
    "num_resumes_processed": 100,
    "total_chunks": 100,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dimension": 768
  },
  "chunks_with_embeddings": [...]
}
```

#### Process Single Step
```http
POST /process-step/<step_number>
Content-Type: application/json
```

Execute individual pipeline steps for debugging.

**Steps:**
- 1: Fetch resume data
- 2: Extract text and categorize
- 3: Clean and preprocess
- 4: Anonymize PII
- 5: Semantic chunking
- 6: Generate embeddings

#### List Checkpoints
```http
GET /checkpoints
```

Retrieve available pipeline checkpoints for resume and state.

#### Download Checkpoint
```http
GET /checkpoint/<step_name>
```

Download specific checkpoint file.

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Weaviate Cloud Configuration
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your_weaviate_api_key

# HuggingFace Configuration
HUGGINGFACE_API_TOKEN=your_huggingface_token
HUGGINGFACE_DATASET_NAME=datasetmaster/resumes
HUGGINGFACE_SPLIT=train

# Flask Configuration
FLASK_DEBUG=False
API_PORT=5000

# Pipeline Configuration
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
BATCH_SIZE=32
```

### Embedding Model

The system uses **BAAI/bge-base-en-v1.5** by default, producing 768-dimensional embeddings optimized for semantic search. To use a different model:

```python
pipeline = SimplifiedResumePipeline()
results = pipeline.run(model_name='your-model-name')
```

Recommended alternatives:
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, faster)
- `sentence-transformers/all-mpnet-base-v2` (768-dim, higher quality)

### Weaviate Schema Customization

The system automatically creates a schema with these properties:

- `candidate_id`: Unique identifier
- `resume_summary`: Vectorized resume text
- `experience_json`: Structured experience data
- `education_json`: Education credentials
- `skills_json`: Technical and soft skills
- `location`: Geographic information
- `years_of_experience`: Total experience duration

---

## Usage Examples

### Example 1: Process Resumes and Store in Weaviate

```python
from pipeline import SimplifiedResumePipeline

pipeline = SimplifiedResumePipeline(store_in_weaviate=True)
results = pipeline.run(
    num_resumes=50,
    model_name='BAAI/bge-base-en-v1.5',
    batch_size=32
)

print(f"Processed {len(results['embedded'])} resumes")
```

### Example 2: Search for Candidates via API

```bash
curl -X POST http://localhost:5000/process-resumes \
  -H "Content-Type: application/json" \
  -d '{
    "num_resumes": 100,
    "model": "BAAI/bge-base-en-v1.5",
    "batch_size": 32
  }'
```

### Example 3: Query from n8n

The n8n workflow includes an AI Agent that can be triggered with natural language queries:

- "Find 5 Python developers with 3+ years of experience"
- "Show me senior full-stack engineers in San Francisco"
- "List candidates with machine learning and cloud experience"

The agent automatically:
1. Checks if resume data exists in Weaviate
2. Translates requirements to search parameters
3. Performs semantic vector search
4. Ranks candidates by relevance
5. Returns structured results with contact information

---

## Project Structure

```
get-hired/
├── api_server.py              # Flask REST API endpoints
├── pipeline.py                # Core pipeline orchestration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
└── src/
    ├── __init__.py
    ├── utils.py               # Logging and utilities
    ├── step1_fetch_data.py     # HuggingFace data fetching
    ├── step2_extract_text.py   # Text extraction & categorization
    ├── step3_clean_text.py     # Text preprocessing
    ├── step4_anonymize_pii.py  # PII removal
    ├── step5_semantic_chunking.py  # Document segmentation
    └── step6_generate_embeddings.py # Embedding generation
```

---

## Performance & Benchmarks

### Processing Speed

- **Text Extraction**: ~50-100 resumes/minute (format dependent)
- **Embedding Generation**: ~200-400 resumes/minute (GPU) or ~50-100 resumes/minute (CPU)
- **Weaviate Storage**: ~1000 resumes/minute (batch insert)

### Model Comparison

| Model | Dimension | Speed (CPU) | Quality | Recommended For |
|-------|-----------|------------|---------|-----------------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Large-scale searches |
| all-mpnet-base-v2 | 768 | Moderate | Excellent | Balanced performance |
| **BAAI/bge-base-en-v1.5** | 768 | Moderate | Excellent | Default choice |
| bge-large-en-v1.5 | 1024 | Slow | Superior | High accuracy needed |

---

## Troubleshooting

### Issue: "Failed to connect to Weaviate"

- Verify `WEAVIATE_URL` and `WEAVIATE_API_KEY` in `.env`
- Test connectivity: `curl https://your-cluster.weaviate.network/v1/meta`
- Check firewall and network policies

### Issue: "Dataset not found on HuggingFace"

- Confirm `HUGGINGFACE_API_TOKEN` is valid
- Verify dataset exists: `huggingface_hub.list_datasets()`
- Check dataset name and split in `.env`

### Issue: "Out of Memory during embedding generation"

- Reduce `--batch-size` parameter
- Process fewer resumes: `--num-resumes 50`
- Use a smaller embedding model

### Issue: "Cloudflare tunnel connection timeout"

- Verify Flask server is running on port 5000
- Check tunnel configuration in Cloudflare dashboard
- Restart tunnel: `cloudflared tunnel run get-hired-tunnel`

---

## Security Considerations

- **API Authentication**: Implement token-based auth before production deployment
- **PII Handling**: The pipeline includes PII anonymization (step 4)
- **Data Encryption**: Store sensitive data in Weaviate with encryption at rest
- **Access Control**: Use Weaviate RBAC for multi-tenant scenarios
- **Environment Variables**: Never commit `.env` file to version control

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes and add tests
4. Submit a pull request with clear description

### Development Setup

```bash
git clone https://github.com/yourusername/get-hired.git
cd get-hired
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Support & Feedback

For issues, questions, or feature requests, please open an issue on GitHub or contact the development team.

---

**Last Updated**: October 2025  
**Maintained By**: Your Organization  
**Status**: Active Development
