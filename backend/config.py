"""Configuration settings for the RAG Application."""

import os
from pathlib import Path

# Application settings
APP_NAME = "RAG Application"
VERSION = "1.0.0"
HOST = "localhost"
PORT = 8001  # Changed to avoid conflict

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
EVAL_DIR = BASE_DIR / "evaluation"

# Create directories if they don't exist
for directory in [DATA_DIR, PDF_DIR, CHUNKS_DIR, VECTOR_DB_DIR, EVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# PDF sources from assignment
PDF_SOURCES = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert_pretraining": "https://arxiv.org/pdf/1810.04805.pdf", 
    "gpt3_language_models": "https://arxiv.org/pdf/2005.14165.pdf",
    "roberta_optimization": "https://arxiv.org/pdf/1907.11692.pdf",
    "t5_text_to_text": "https://arxiv.org/pdf/1910.10683.pdf"
}

# Model configurations (optimized for local PC performance)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and efficient
LANGUAGE_MODEL = "microsoft/DialoGPT-medium"  # Good balance of quality and speed
FALLBACK_MODEL = "google/flan-t5-small"  # Lightweight fallback

# Text processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_TOKENS = 512

# Vector database settings
VECTOR_DIM = 384  # Dimension for all-MiniLM-L6-v2
TOP_K_RESULTS = 5

# Memory settings
CONVERSATION_MEMORY_SIZE = 4  # Last 4 interactions as per assignment

# Evaluation settings
TEST_QUESTIONS_FILE = EVAL_DIR / "test_questions.json"
EVALUATION_RESULTS_FILE = EVAL_DIR / "evaluation_results.json"

# File paths
FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.index"
CHUNKS_METADATA_PATH = VECTOR_DB_DIR / "chunks_metadata.pkl" 