# Enterprise GraphRAG System

Full-Featured, Production-Ready, Fully Offline Enterprise GraphRAG Application.

## Run
# Instance 1 of shell - 
  powershell -ExecutionPolicy Bypass -File "start_neo4j.ps1"
# Instance 2 of shell - 
  python.exe "run.py"

# Browser
  http://localhost:8000/

## Overview
Built with FastAPI, PyTorch, Gemma 4 1B Instruct, SentenceTransformers, FAISS, Neo4j Community Edition, SQLite, NetworkX, and Cytoscape.js.

## Requirements
- Windows 11
- Python 3.11
- NVIDIA RTX GPU with CUDA (or CPU fallback)
- Neo4j Community Edition (Local Server running on `bolt://localhost:7687`)

## Installation & Setup

1. **Clone & Virtual Environment**:
   ```bash
   cd "C:\Users\Loq\Documents\programs\final year project\graph"
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Copy `.env.example` to `.env` and set your `HF_TOKEN` (for first-time model download).

5. **Pre-download Models for Offline Execution**:
   ```bash
   python scripts/download_models.py
   ```

6. **Initialize Neo4j Schema**:
   Ensure Neo4j Community Edition is running locally on port 7687, then run:
   ```bash
   python scripts/setup_neo4j.py
   ```

7. **Run the Application**:
   ```bash
   python run.py
   ```
   Open your browser at `http://localhost:8000` to access the Single Page Application UI.

## Testing
Run automated unit, integration, and API tests:
```bash
pytest
```
