import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application settings
    app_name: str = "RAG Application"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # PDF URLs for the research papers
    pdf_urls: List[str] = [
        "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need
        "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
        "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
        "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
        "https://arxiv.org/pdf/1910.10683.pdf",  # T5
    ]
    
    # PDF metadata
    pdf_names: List[str] = [
        "attention_is_all_you_need.pdf",
        "bert_pretraining.pdf", 
        "gpt3_language_models.pdf",
        "roberta_optimization.pdf",
        "t5_text_to_text.pdf"
    ]
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    language_model: str = "microsoft/DialoGPT-medium"
    
    # Vector database settings
    vector_db_path: str = "data/vector_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
    
    # Conversation memory settings
    memory_size: int = 4  # Last 4 interactions
    
    # Directories
    data_dir: str = "data"
    pdf_dir: str = "data/pdfs"
    chunks_dir: str = "data/chunks"
    reports_dir: str = "reports"
    
    # Evaluation settings
    test_questions_file: str = "evaluation/test_questions.json"
    evaluation_output_dir: str = "evaluation/results"
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.pdf_dir, exist_ok=True)
os.makedirs(settings.chunks_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs(settings.evaluation_output_dir, exist_ok=True) 