import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
import json
from pathlib import Path
import time

from config import settings
from app.pdf_ingestion import PDFIngestionPipeline
from app.vector_database import VectorDatabase
from app.language_model import RAGResponseGenerator, ConversationMemoryManager


class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(self):
        self.pdf_pipeline = PDFIngestionPipeline()
        self.vector_db = VectorDatabase()
        self.response_generator = RAGResponseGenerator()
        self.memory_manager = ConversationMemoryManager()
        
        self.is_initialized = False
        self.initialization_stats = {}
    
    async def initialize(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the RAG system by processing PDFs and building the vector database."""
        
        if self.is_initialized and not force_rebuild:
            logger.info("RAG system already initialized")
            return self.initialization_stats
        
        logger.info("Initializing RAG system...")
        
        try:
            # Step 1: Process PDFs
            logger.info("Step 1: Processing PDF documents")
            document_chunks = await self.pdf_pipeline.process_all_pdfs()
            
            if not document_chunks:
                raise RuntimeError("No documents were successfully processed")
            
            # Step 2: Build/rebuild vector database
            logger.info("Step 2: Building vector database")
            if force_rebuild or self.vector_db.rebuild_if_needed(document_chunks):
                vector_stats = self.vector_db.build_index(document_chunks)
            else:
                vector_stats = self.vector_db.get_database_stats()
            
            # Step 3: Load language model
            logger.info("Step 3: Loading language model")
            start_time = time.time()
            self.response_generator.load_model()
            load_time = time.time() - start_time
            logger.success(f"Language model loaded in {load_time:.2f} seconds")
            
            # Generate initialization statistics
            logger.info("Step 4: Generating initialization statistics")
            pdf_stats = self.pdf_pipeline.get_processing_stats(document_chunks)
            
            self.initialization_stats = {
                'status': 'initialized',
                'pdf_processing': pdf_stats,
                'vector_database': vector_stats,
                'language_model': {
                    'model_name': self.response_generator.llm.model_name,
                    'device': self.response_generator.llm.device,
                    'loaded': self.response_generator.is_loaded()
                },
                'memory_manager': self.memory_manager.get_memory_summary()
            }
            
            self.is_initialized = True
            logger.success("RAG system initialization completed successfully")
            
            # Log final stats
            logger.info("=== INITIALIZATION COMPLETE ===")
            logger.info(f"✅ Total documents processed: {self.initialization_stats['pdf_processing']['total_documents']}")
            logger.info(f"✅ Total chunks created: {self.initialization_stats['pdf_processing']['total_chunks']}")
            logger.info(f"✅ Vector database size: {self.initialization_stats['vector_database']['index_size']}")
            logger.info(f"✅ Language model: {self.initialization_stats['language_model']['model_name']}")
            logger.info(f"✅ Model loaded: {self.initialization_stats['language_model']['loaded']}")
            logger.info("🚀 RAG Application is ready for queries!")
            
            return self.initialization_stats
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            self.initialization_stats = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    async def query(
        self, 
        user_question: str, 
        top_k: int = None, 
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Process a user query and return a RAG response."""
        
        if not self.is_initialized:
            await self.initialize()
        
        try:
            logger.info(f"Processing query: '{user_question[:100]}...'")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.vector_db.search(
                query=user_question, 
                top_k=top_k or settings.top_k_results
            )
            
            if not retrieved_chunks:
                return {
                    'response': "I apologize, but I couldn't find any relevant information in the documents to answer your question.",
                    'sources': [],
                    'query': user_question,
                    'retrieval_stats': {
                        'chunks_found': 0,
                        'top_score': 0
                    }
                }
            
            # Step 2: Get conversation context
            conversation_history = self.memory_manager.get_conversation_context()
            
            # Step 3: Generate response
            rag_response = self.response_generator.generate_rag_response(
                query=user_question,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history
            )
            
            # Step 4: Update conversation memory
            self.memory_manager.add_interaction(
                user_message=user_question,
                assistant_response=rag_response['response']
            )
            
            # Enhance response with additional metadata
            enhanced_response = {
                **rag_response,
                'retrieval_stats': {
                    'chunks_found': len(retrieved_chunks),
                    'top_score': retrieved_chunks[0]['similarity_score'] if retrieved_chunks else 0,
                    'documents_referenced': list(set(chunk['document'] for chunk in retrieved_chunks))
                },
                'memory_context': len(conversation_history) > 0
            }
            
            if not include_sources:
                enhanced_response.pop('sources', None)
            
            logger.success(f"Generated response for query (score: {enhanced_response['retrieval_stats']['top_score']:.3f})")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            return {
                'response': "I encountered an error while processing your question. Please try again.",
                'sources': [],
                'query': user_question,
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of all system components."""
        
        status = {
            'system_initialized': self.is_initialized,
            'components': {
                'pdf_pipeline': {
                    'status': 'ready'
                },
                'vector_database': self.vector_db.get_database_stats(),
                'language_model': {
                    'loaded': self.response_generator.is_loaded(),
                    'model_name': self.response_generator.llm.model_name if hasattr(self.response_generator.llm, 'model_name') else 'not_loaded'
                },
                'memory_manager': self.memory_manager.get_memory_summary()
            }
        }
        
        if self.is_initialized:
            status['initialization_stats'] = self.initialization_stats
        
        return status
    
    def clear_conversation_memory(self):
        """Clear the conversation memory."""
        self.memory_manager.clear_memory()
        logger.info("Conversation memory cleared")
    
    async def get_similar_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get similar document chunks without generating a response."""
        
        if not self.is_initialized:
            await self.initialize()
        
        return self.vector_db.search(query=query, top_k=top_k)
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed documents."""
        
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        # Load chunks from saved files to get detailed stats
        stats = {
            'documents': {},
            'total_chunks': 0,
            'total_characters': 0
        }
        
        chunks_dir = Path(settings.chunks_dir)
        for chunk_file in chunks_dir.glob("*_chunks.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                doc_name = chunk_file.stem.replace('_chunks', '')
                doc_stats = {
                    'chunks': len(chunks),
                    'total_characters': sum(chunk['size'] for chunk in chunks),
                    'avg_chunk_size': sum(chunk['size'] for chunk in chunks) / len(chunks) if chunks else 0
                }
                
                stats['documents'][doc_name] = doc_stats
                stats['total_chunks'] += doc_stats['chunks']
                stats['total_characters'] += doc_stats['total_characters']
                
            except Exception as e:
                logger.warning(f"Failed to load stats for {chunk_file}: {str(e)}")
        
        if stats['total_chunks'] > 0:
            stats['avg_chunk_size'] = stats['total_characters'] / stats['total_chunks']
        
        return stats


class RAGSystemManager:
    """Singleton manager for the RAG system."""
    
    _instance = None
    _rag_system = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGSystemManager, cls).__new__(cls)
        return cls._instance
    
    def get_rag_system(self) -> RAGSystem:
        """Get the RAG system instance."""
        if self._rag_system is None:
            self._rag_system = RAGSystem()
        return self._rag_system
    
    async def initialize_system(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the RAG system."""
        rag_system = self.get_rag_system()
        return await rag_system.initialize(force_rebuild=force_rebuild)
    
    def reset_system(self):
        """Reset the RAG system (for testing or reinitialization)."""
        self._rag_system = None


# Global instance
rag_manager = RAGSystemManager() 