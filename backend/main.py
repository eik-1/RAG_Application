"""Main FastAPI application for the RAG system."""

import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from app.rag_system import RAGSystem
from config import APP_NAME, VERSION, HOST, PORT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting RAG Application...")
    
    # Initialize system in background
    def initialize_system():
        try:
            result = rag_system.initialize()
            if result["success"]:
                logger.info("✅ RAG system initialized successfully")
            else:
                logger.error(f"❌ RAG system initialization failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"❌ Startup initialization failed: {e}")
    
    # Run initialization in background
    asyncio.create_task(asyncio.to_thread(initialize_system))
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down RAG Application...")

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="A comprehensive RAG application with conversational memory",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: str
    sources: List[Dict[str, Any]]
    processing_time: Optional[float] = None
    error: Optional[str] = None

class InitializeRequest(BaseModel):
    force_rebuild: Optional[bool] = False

class SystemResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None





@app.get("/")
async def root():
    """Root endpoint with basic application information."""
    return {
        "application": APP_NAME,
        "version": VERSION,
        "status": "running",
        "description": "RAG Application with conversational memory",
        "endpoints": {
            "chat": "/chat",
            "status": "/status",
            "health": "/health",
            "initialize": "/initialize",
            "memory": "/memory",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = rag_system.get_health_status()
    
    return {
        "status": "healthy",
        "timestamp": health_status,
        "components": health_status.get("components", {}),
        "memory": health_status.get("memory", {}),
        "system_ready": rag_system.is_initialized
    }


@app.get("/status")
async def get_system_status():
    """Get detailed system status and statistics."""
    try:
        stats = rag_system.get_system_stats()
        health = rag_system.get_health_status()
        
        # Structure the response to match frontend expectations
        return {
            "system_initialized": rag_system.is_initialized,
            "components": {
                "pdf_pipeline": {
                    "status": "ready" if stats.get("initialization_status", {}).get("pdf_processing") else "not_ready"
                },
                "vector_database": {
                    "status": "ready" if stats.get("initialization_status", {}).get("vector_db") else "not_ready",
                    "total_vectors": stats.get("total_vectors", 0),
                    "total_chunks": stats.get("total_chunks", 0),
                    "embedding_dimension": stats.get("vector_dimension", 0),
                    "document_distribution": stats.get("sources", {})
                },
                "language_model": {
                    "loaded": stats.get("initialization_status", {}).get("language_model", False),
                    "model_name": stats.get("language_model", {}).get("current_model_type", "unknown")
                },
                "memory_manager": {
                    "total_interactions": stats.get("memory_size", 0),
                    "memory_size_limit": stats.get("max_memory_size", 4)
                }
            },
            "health": health,
            "raw_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_rag(request: ChatRequest) -> ChatResponse:
    """Chat with the RAG system."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process the chat request
        result = rag_system.chat(request.message)
        
        return ChatResponse(
            success=result["success"],
            response=result["response"],
            sources=result.get("sources", []),
            processing_time=result.get("processing_time"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            success=False,
            response="I encountered an error while processing your request.",
            sources=[],
            error=str(e)
        )


@app.post("/initialize")
async def initialize_system(request: InitializeRequest, background_tasks: BackgroundTasks):
    """Initialize or reinitialize the RAG system."""
    try:
        def run_initialization():
            return rag_system.initialize(force_rebuild=request.force_rebuild)
        
        # Run initialization
        result = await asyncio.to_thread(run_initialization)
        
        return SystemResponse(
            status="success" if result["success"] else "error",
            message="System initialization completed" if result["success"] else f"Initialization failed: {result.get('error')}",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory")
async def get_conversation_memory():
    """Get current conversation memory."""
    try:
        memory = rag_system.get_memory()
        return {
            "status": "success",
            "memory": memory,
            "memory_size": len(memory),
            "max_memory_size": rag_system.conversation_memory.maxlen
        }
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
async def clear_conversation_memory():
    """Clear conversation memory."""
    try:
        rag_system.clear_memory()
        return SystemResponse(
            status="success",
            message="Conversation memory cleared successfully"
        )
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats")
async def get_document_stats():
    """Get document processing statistics."""
    try:
        stats = rag_system.get_system_stats()
        
        return {
            "status": "success",
            "statistics": {
                "total_chunks": stats.get("total_chunks", 0),
                "total_vectors": stats.get("total_vectors", 0),
                "sources": stats.get("sources", {}),
                "embedding_model": stats.get("embedding_model", "Unknown"),
                "vector_dimension": stats.get("vector_dimension", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks."""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Use the vector database search functionality
        results = rag_system.vector_db.search(request.query, k=request.top_k)
        
        return {
            "status": "success",
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild_system(background_tasks: BackgroundTasks):
    """Rebuild the entire RAG system from scratch."""
    try:
        def run_rebuild():
            return rag_system.rebuild_system()
        
        # Run rebuild in background
        result = await asyncio.to_thread(run_rebuild)
        
        return SystemResponse(
            status="success" if result["success"] else "error",
            message="System rebuild completed" if result["success"] else f"Rebuild failed: {result.get('error')}",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error in rebuild endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"🚀 Starting {APP_NAME} v{VERSION}")
    print(f"📡 Server will run at http://{HOST}:{PORT}")
    print(f"📚 API documentation at http://{HOST}:{PORT}/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    ) 