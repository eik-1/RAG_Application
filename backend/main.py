import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from loguru import logger
import sys

from config import settings
from app.rag_system import rag_manager


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    include_sources: bool = True
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    query: str
    retrieval_stats: Dict[str, Any]
    memory_context: bool = False
    model_used: Optional[str] = None


class InitializationRequest(BaseModel):
    force_rebuild: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    
    # Startup
    logger.info("Starting RAG Application backend...")
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    try:
        # Initialize the RAG system
        logger.info("Initializing RAG system...")
        await rag_manager.initialize_system()
        logger.success("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        logger.warning("Application will start but RAG system may not be available")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Application backend...")


# Create FastAPI app
app = FastAPI(
    title="RAG Application API",
    description="A Retrieval-Augmented Generation application with conversational memory",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "RAG Application API",
        "version": "1.0.0",
        "description": "A Retrieval-Augmented Generation application with conversational memory",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    rag_system = rag_manager.get_rag_system()
    status = rag_system.get_system_status()
    
    return {
        "status": "healthy" if status["system_initialized"] else "initializing",
        "timestamp": "2024-01-01T00:00:00",
        "components": status["components"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a RAG response."""
    
    try:
        rag_system = rag_manager.get_rag_system()
        
        response = await rag_system.query(
            user_question=request.message,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")


@app.post("/initialize")
async def initialize_system(request: InitializationRequest = InitializationRequest()):
    """Initialize or reinitialize the RAG system."""
    
    try:
        logger.info(f"Initializing system (force_rebuild: {request.force_rebuild})")
        
        stats = await rag_manager.initialize_system(force_rebuild=request.force_rebuild)
        
        return {
            "status": "success",
            "message": "RAG system initialized successfully",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize system: {str(e)}")


@app.get("/status")
async def get_system_status():
    """Get detailed system status."""
    
    rag_system = rag_manager.get_rag_system()
    return rag_system.get_system_status()


@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search for similar document chunks."""
    
    try:
        rag_system = rag_manager.get_rag_system()
        
        results = await rag_system.get_similar_documents(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")


@app.get("/documents/stats")
async def get_document_statistics():
    """Get statistics about processed documents."""
    
    rag_system = rag_manager.get_rag_system()
    return rag_system.get_document_statistics()


@app.post("/memory/clear")
async def clear_conversation_memory():
    """Clear the conversation memory."""
    
    try:
        rag_system = rag_manager.get_rag_system()
        rag_system.clear_conversation_memory()
        
        return {
            "status": "success",
            "message": "Conversation memory cleared"
        }
        
    except Exception as e:
        logger.error(f"Memory clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@app.get("/memory/status")
async def get_memory_status():
    """Get conversation memory status."""
    
    rag_system = rag_manager.get_rag_system()
    return rag_system.memory_manager.get_memory_summary()


# Background task endpoints

@app.post("/rebuild")
async def rebuild_system(background_tasks: BackgroundTasks):
    """Rebuild the entire system in the background."""
    
    def rebuild_task():
        try:
            asyncio.run(rag_manager.initialize_system(force_rebuild=True))
            logger.success("Background rebuild completed")
        except Exception as e:
            logger.error(f"Background rebuild failed: {str(e)}")
    
    background_tasks.add_task(rebuild_task)
    
    return {
        "status": "accepted",
        "message": "System rebuild started in background"
    }


# Error handlers

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    logger.info(f"Starting RAG Application on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    ) 