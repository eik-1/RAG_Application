import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
import json

from config import settings


class VectorDatabase:
    """FAISS-based vector database for storing and retrieving document embeddings."""
    
    def __init__(self, embedding_model_name: str = None):
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.embedding_model = None
        self.index = None
        self.chunks_metadata = []
        self.dimension = None
        
        # Paths for saving/loading
        self.db_path = Path(settings.vector_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.db_path / "faiss_index.index"
        self.metadata_path = self.db_path / "chunks_metadata.pkl"
        self.embeddings_path = self.db_path / "embeddings.npy"
        
    def load_embedding_model(self):
        """Load the sentence transformer model for generating embeddings."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.success(f"Loaded embedding model with dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches to manage memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=True if i == 0 else False
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        logger.success(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_index(self, chunks_data: Dict[str, List[Dict[str, Any]]]):
        """Build FAISS index from document chunks."""
        
        # Flatten all chunks from all documents
        all_chunks = []
        for doc_name, chunks in chunks_data.items():
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in all_chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Initialize FAISS index
        if self.dimension is None:
            self.dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunks_metadata = all_chunks
        
        logger.success(f"Built FAISS index with {self.index.ntotal} vectors")
        
        # Save to disk
        self.save_index()
        
        return {
            'total_chunks': len(all_chunks),
            'embedding_dimension': self.dimension,
            'index_size': self.index.ntotal
        }
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            logger.success(f"Saved vector database to {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector database: {str(e)}")
            raise
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                logger.warning("Vector database files not found")
                return False
            
            # Load embedding model first
            if self.embedding_model is None:
                self.load_embedding_model()
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            logger.success(f"Loaded vector database with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks given a query."""
        
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("Vector database not available. Please build the index first.")
        
        if self.embedding_model is None:
            self.load_embedding_model()
        
        top_k = top_k or settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = len(results) + 1
                results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} results for query: '{query[:50]}...'")
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        
        if self.index is None:
            if not self.load_index():
                return {"status": "not_built"}
        
        # Document distribution
        doc_distribution = {}
        for chunk in self.chunks_metadata:
            doc_name = chunk.get('document', 'unknown')
            doc_distribution[doc_name] = doc_distribution.get(doc_name, 0) + 1
        
        stats = {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.dimension,
            'total_chunks': len(self.chunks_metadata),
            'document_distribution': doc_distribution,
            'embedding_model': self.embedding_model_name,
            'index_type': type(self.index).__name__
        }
        
        return stats
    
    def rebuild_if_needed(self, chunks_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Rebuild index if it doesn't exist or is outdated."""
        
        if not self.index_path.exists():
            logger.info("Index not found, building new index")
            self.build_index(chunks_data)
            return True
        
        # Check if we have the same number of chunks
        if self.load_index():
            current_total = sum(len(chunks) for chunks in chunks_data.values())
            if len(self.chunks_metadata) != current_total:
                logger.info("Chunk count mismatch, rebuilding index")
                self.build_index(chunks_data)
                return True
        else:
            logger.info("Failed to load existing index, rebuilding")
            self.build_index(chunks_data)
            return True
        
        return False


class EmbeddingManager:
    """Manager for handling different embedding strategies and caching."""
    
    def __init__(self):
        self.cache_dir = Path(settings.data_dir) / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, model_name: str, text_hash: str) -> Path:
        """Get cache file path for a specific model and text hash."""
        safe_model_name = model_name.replace('/', '_')
        return self.cache_dir / f"{safe_model_name}_{text_hash}.npy"
    
    def compute_text_hash(self, texts: List[str]) -> str:
        """Compute hash for a list of texts."""
        import hashlib
        combined_text = ''.join(texts)
        return hashlib.md5(combined_text.encode()).hexdigest()
    
    def cache_embeddings(self, model_name: str, texts: List[str], embeddings: np.ndarray):
        """Cache embeddings for future use."""
        text_hash = self.compute_text_hash(texts)
        cache_path = self.get_cache_path(model_name, text_hash)
        
        try:
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {str(e)}")
    
    def load_cached_embeddings(self, model_name: str, texts: List[str]) -> Optional[np.ndarray]:
        """Load cached embeddings if available."""
        text_hash = self.compute_text_hash(texts)
        cache_path = self.get_cache_path(model_name, text_hash)
        
        if cache_path.exists():
            try:
                embeddings = np.load(cache_path)
                logger.info(f"Loaded cached embeddings from {cache_path}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {str(e)}")
        
        return None 