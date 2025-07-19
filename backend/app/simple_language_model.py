"""Simple language model fallback that doesn't require transformers."""

import re
from typing import List, Dict, Any


class SimpleLM:
    """Simple rule-based language model for testing without transformers."""
    
    def __init__(self):
        self.is_initialized = True
        self.model_type = "simple_rule_based"
    
    def initialize(self):
        """Initialize the simple model."""
        print("🤖 Initializing simple rule-based language model...")
        self.is_initialized = True
    
    def generate_response(self, context: str, query: str, 
                         retrieved_chunks: List[Dict[str, Any]],
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate a simple response based on retrieved context."""
        
        # Extract key concepts from query
        query_lower = query.lower()
        
        # Topic detection
        topics = {
            'transformer': ['transformer', 'attention', 'self-attention'],
            'bert': ['bert', 'bidirectional', 'masked', 'pretraining'],
            'gpt': ['gpt', 'generative', 'autoregressive', 'few-shot'],
            'roberta': ['roberta', 'optimization', 'robust'],
            't5': ['t5', 'text-to-text', 'transfer', 'unified']
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # Build response from retrieved chunks
        response_parts = []
        
        if retrieved_chunks:
            response_parts.append("Based on the research papers:")
            
            for i, chunk in enumerate(retrieved_chunks[:2]):  # Top 2 chunks
                source = chunk.get('source', 'unknown')
                text = chunk.get('text', '')[:200]  # First 200 chars
                
                # Clean text
                text = re.sub(r'\s+', ' ', text).strip()
                
                response_parts.append(f"\nFrom {source}: {text}")
        
        # Add topic-specific information
        if 'transformer' in detected_topics:
            response_parts.append("\nThe Transformer architecture introduced the attention mechanism that allows models to focus on relevant parts of the input sequence.")
        
        if 'bert' in detected_topics:
            response_parts.append("\nBERT uses bidirectional training, which allows it to understand context from both directions in a sentence.")
        
        if 'gpt' in detected_topics:
            response_parts.append("\nGPT models are autoregressive and can perform few-shot learning by providing examples in the prompt.")
        
        # Fallback response
        if not response_parts:
            response_parts = [
                "I can provide information about the research papers covering Transformer, BERT, GPT-3, RoBERTa, and T5.",
                "Please ask about specific aspects like architecture, training methods, or performance comparisons."
            ]
        
        return " ".join(response_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "primary_model": "simple_rule_based",
            "fallback_model": None,
            "current_model_type": self.model_type,
            "is_initialized": self.is_initialized,
            "cuda_available": False,
            "max_tokens": 500
        } 