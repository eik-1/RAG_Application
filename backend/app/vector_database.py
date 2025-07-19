"""Vector database using FAISS for semantic search."""

import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from config import (
    EMBEDDING_MODEL, VECTOR_DIM, TOP_K_RESULTS,
    FAISS_INDEX_PATH, CHUNKS_METADATA_PATH
)

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not available")
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Sentence transformers not available, using simple text search")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class VectorDatabase:
    """FAISS-based vector database for semantic search."""
    
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the embedding model and load existing index if available."""
        print("Initializing vector database...")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            print("Using simple text search (no embeddings)")
            self.chunks = []
            self.is_initialized = True
            return
        
        # Load embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Try to load existing index and metadata
        if FAISS_INDEX_PATH.exists() and CHUNKS_METADATA_PATH.exists():
            self.load_index()
        else:
            # Create empty index
            self.index = faiss.IndexFlatIP(VECTOR_DIM)  # Inner product for cosine similarity
            self.chunks = []
        
        self.is_initialized = True
        print("Vector database initialized")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.is_initialized:
            self.initialize()
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings.astype('float32')
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index from chunks or store for simple search."""
        if not self.is_initialized:
            self.initialize()
        
        # Store chunks for simple search if embeddings not available
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            print(f"Storing {len(chunks)} chunks for simple text search...")
            self.chunks = chunks
            print("Simple text search ready")
            return
        
        print(f"Building FAISS index for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create new index
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunk metadata
        self.chunks = chunks
        
        # Save index and metadata
        self.save_index()
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and chunk metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        
        # Save chunk metadata
        with open(CHUNKS_METADATA_PATH, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved to {FAISS_INDEX_PATH}")
        print(f"Metadata saved to {CHUNKS_METADATA_PATH}")
    
    def load_index(self):
        """Load FAISS index and chunk metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            
            # Load chunk metadata
            with open(CHUNKS_METADATA_PATH, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"Loaded index with {self.index.ntotal} vectors")
            print(f"Loaded {len(self.chunks)} chunk metadata")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            # Create empty index if loading fails
            self.index = faiss.IndexFlatIP(VECTOR_DIM)
            self.chunks = []
    
    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity or simple text search."""
        if not self.is_initialized:
            self.initialize()
        
        # Use simple text search if embeddings not available
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            return self._simple_text_search(query, k)
        
        if self.index.ntotal == 0:
            print("Warning: Vector database is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search for similar vectors
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def _simple_text_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Simple keyword-based text search fallback."""
        if not self.chunks:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for chunk in self.chunks:
            text = chunk.get('text', '').lower()
            chunk_words = set(text.split())
            
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = overlap / len(query_words)
                result = chunk.copy()
                result['similarity_score'] = score
                results.append(result)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Add rank
        for i, result in enumerate(results[:k]):
            result['rank'] = i + 1
        
        return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.is_initialized:
            self.initialize()
        
        source_counts = {}
        for chunk in self.chunks:
            source = chunk.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal if self.index else 0,
            "vector_dimension": VECTOR_DIM,
            "embedding_model": EMBEDDING_MODEL,
            "sources": source_counts,
            "index_exists": FAISS_INDEX_PATH.exists(),
            "metadata_exists": CHUNKS_METADATA_PATH.exists()
        }
    
    def rebuild_index(self, chunks: List[Dict[str, Any]]):
        """Rebuild the entire index with new chunks."""
        print("Rebuilding vector database index...")
        
        # Remove existing files
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if CHUNKS_METADATA_PATH.exists():
            CHUNKS_METADATA_PATH.unlink()
        
        # Build new index
        self.build_index(chunks)
        
        print("Index rebuild completed") 