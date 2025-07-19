"""Main RAG system that integrates all components."""

from typing import List, Dict, Any, Optional
from collections import deque
import time
import json

from .pdf_ingestion import PDFIngestion
from .vector_database import VectorDatabase
from .language_model import LanguageModel
from config import CONVERSATION_MEMORY_SIZE


class RAGSystem:
    """Complete RAG system with conversational memory."""
    
    def __init__(self):
        self.pdf_ingestion = PDFIngestion()
        self.vector_db = VectorDatabase()
        self.language_model = LanguageModel()
        
        # Conversation memory (last 4 interactions)
        self.conversation_memory = deque(maxlen=CONVERSATION_MEMORY_SIZE)
        
        self.is_initialized = False
        self.initialization_status = {
            "pdf_processing": False,
            "vector_db": False,
            "language_model": False
        }
    
    def initialize(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the complete RAG system."""
        print("🚀 Initializing RAG System...")
        start_time = time.time()
        
        try:
            # Step 1: Process PDFs
            print("📄 Processing PDFs...")
            if force_rebuild or not self._check_existing_data():
                chunks = self.pdf_ingestion.process_all_pdfs()
                if not chunks:
                    raise Exception("No chunks were processed from PDFs")
            else:
                print("Using existing PDF chunks...")
                chunks = self._load_existing_chunks()
            
            self.initialization_status["pdf_processing"] = True
            
            # Step 2: Initialize vector database
            print("🔍 Initializing vector database...")
            self.vector_db.initialize()
            
            # Build index if needed
            if force_rebuild or self.vector_db.index.ntotal == 0:
                self.vector_db.build_index(chunks)
            
            self.initialization_status["vector_db"] = True
            
            # Step 3: Initialize language model
            print("🤖 Initializing language model...")
            self.language_model.initialize()
            self.initialization_status["language_model"] = True
            
            self.is_initialized = True
            initialization_time = time.time() - start_time
            
            print(f"✅ RAG System initialized successfully in {initialization_time:.2f} seconds")
            
            return {
                "success": True,
                "initialization_time": initialization_time,
                "status": self.initialization_status,
                "stats": self.get_system_stats()
            }
            
        except Exception as e:
            print(f"❌ RAG System initialization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": self.initialization_status
            }
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and return a response with memory."""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "System not initialized. Please initialize first.",
                "response": "I'm not ready yet. Please wait for system initialization.",
                "sources": []
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Search for relevant chunks
            print(f"🔍 Searching for: {user_message}")
            retrieved_chunks = self.vector_db.search(user_message)
            
            if not retrieved_chunks:
                response = "I couldn't find relevant information to answer your question. Could you try rephrasing or asking about the research papers I have access to?"
                sources = []
            else:
                # Step 2: Generate response with context and memory
                print("🤖 Generating response...")
                context = self._prepare_context(retrieved_chunks)
                
                response = self.language_model.generate_response(
                    context=context,
                    query=user_message,
                    retrieved_chunks=retrieved_chunks,
                    conversation_history=list(self.conversation_memory)
                )
                
                sources = self._prepare_sources(retrieved_chunks)
            
            # Step 3: Update conversation memory
            self._update_memory(user_message, response)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "processing_time": processing_time,
                "retrieved_chunks": len(retrieved_chunks),
                "memory_size": len(self.conversation_memory)
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your request. Please try again.",
                "sources": []
            }
    
    def _check_existing_data(self) -> bool:
        """Check if processed data already exists."""
        return (
            self.pdf_ingestion.chunks_dir.exists() and
            any(self.pdf_ingestion.chunks_dir.glob("*_chunks.json"))
        )
    
    def _load_existing_chunks(self) -> List[Dict[str, Any]]:
        """Load existing chunks from disk."""
        all_chunks = []
        
        for chunks_file in self.pdf_ingestion.chunks_dir.glob("*_chunks.json"):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error loading {chunks_file}: {e}")
        
        return all_chunks
    
    def _prepare_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks."""
        context_parts = []
        
        for chunk in retrieved_chunks[:3]:  # Top 3 most relevant
            source = chunk.get('source', 'unknown')
            text = chunk.get('text', '')[:300]  # Limit text length
            context_parts.append(f"From {source}: {text}")
        
        return "\n\n".join(context_parts)
    
    def _prepare_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for response."""
        sources = []
        
        for chunk in retrieved_chunks:
            sources.append({
                "source": chunk.get('source', 'unknown'),
                "chunk_id": chunk.get('id', ''),
                "similarity_score": chunk.get('similarity_score', 0.0),
                "text_preview": chunk.get('text', '')[:200] + "..."
            })
        
        return sources
    
    def _update_memory(self, user_message: str, assistant_response: str):
        """Update conversation memory with new interaction."""
        interaction = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": time.time()
        }
        
        self.conversation_memory.append(interaction)
        print(f"💭 Memory updated. Current size: {len(self.conversation_memory)}")
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.conversation_memory.clear()
        print("💭 Conversation memory cleared")
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get current conversation memory."""
        return list(self.conversation_memory)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "is_initialized": self.is_initialized,
            "initialization_status": self.initialization_status,
            "memory_size": len(self.conversation_memory),
            "max_memory_size": CONVERSATION_MEMORY_SIZE
        }
        
        # Add vector database stats
        if self.vector_db.is_initialized:
            stats.update(self.vector_db.get_stats())
        
        # Add language model info
        if self.language_model.is_initialized:
            stats["language_model"] = self.language_model.get_model_info()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        health = {
            "status": "healthy" if self.is_initialized else "initializing",
            "components": {
                "pdf_ingestion": "ready" if self.initialization_status["pdf_processing"] else "not_ready",
                "vector_database": "ready" if self.initialization_status["vector_db"] else "not_ready",
                "language_model": "ready" if self.initialization_status["language_model"] else "not_ready"
            },
            "memory": {
                "current_size": len(self.conversation_memory),
                "max_size": CONVERSATION_MEMORY_SIZE
            }
        }
        
        return health
    
    def rebuild_system(self) -> Dict[str, Any]:
        """Rebuild the entire system from scratch."""
        print("🔄 Rebuilding RAG system...")
        
        # Clear memory
        self.clear_memory()
        
        # Reset initialization status
        self.is_initialized = False
        self.initialization_status = {
            "pdf_processing": False,
            "vector_db": False,
            "language_model": False
        }
        
        # Reinitialize with force rebuild
        return self.initialize(force_rebuild=True) 