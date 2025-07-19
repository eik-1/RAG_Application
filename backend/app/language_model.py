"""Language model integration for response generation."""

from typing import List, Dict, Any, Optional
from config import LANGUAGE_MODEL, FALLBACK_MODEL, MAX_TOKENS

# Try to import transformers, fall back to simple model if not available
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        T5Tokenizer, T5ForConditionalGeneration,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available, using simple rule-based model")
    TRANSFORMERS_AVAILABLE = False
    try:
        from .simple_language_model import SimpleLM
    except ImportError:
        # Define SimpleLM inline if import fails
        SimpleLM = None


class LanguageModel:
    """Handles language model integration and response generation."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.fallback_model = None
        self.fallback_tokenizer = None
        self.simple_model = None
        self.is_initialized = False
        self.model_type = None
    
    def initialize(self):
        """Initialize the language model and fallback model."""
        print("Initializing language model...")
        
        if not TRANSFORMERS_AVAILABLE:
            # Use simple rule-based model when transformers not available
            print("Using simple rule-based language model...")
            self.simple_model = self._create_simple_model()
            self.model_type = "simple_rule_based"
            self.is_initialized = True
            return
        
        try:
            # Try to load primary model
            self._load_primary_model()
        except Exception as e:
            print(f"Failed to load primary model: {e}")
            print("Falling back to smaller model...")
            try:
                self._load_fallback_model()
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                print("Using simple rule-based model...")
                self.simple_model = self._create_simple_model()
                self.model_type = "simple_rule_based"
        
        self.is_initialized = True
        print(f"Language model initialized: {self.model_type}")
    
    def _load_primary_model(self):
        """Load the primary language model."""
        print(f"Loading primary model: {LANGUAGE_MODEL}")
        
        if "dialo" in LANGUAGE_MODEL.lower():
            # DialoGPT model
            self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                LANGUAGE_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_type = "dialogpt"
        else:
            # Generic causal LM
            self.tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                LANGUAGE_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_type = "causal_lm"
    
    def _load_fallback_model(self):
        """Load the fallback model (T5-small)."""
        print(f"Loading fallback model: {FALLBACK_MODEL}")
        
        self.fallback_tokenizer = T5Tokenizer.from_pretrained(FALLBACK_MODEL)
        self.fallback_model = T5ForConditionalGeneration.from_pretrained(
            FALLBACK_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model_type = "t5_fallback"
    
    def generate_response(self, context: str, query: str, 
                         retrieved_chunks: List[Dict[str, Any]],
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate a response based on context, query, and retrieved chunks."""
        if not self.is_initialized:
            self.initialize()
        
        # Use simple model if transformers not available
        if self.model_type == "simple_rule_based":
            return self.simple_model.generate_response(context, query, retrieved_chunks, conversation_history)
        
        # Prepare the prompt
        prompt = self._create_prompt(context, query, retrieved_chunks, conversation_history)
        
        try:
            if self.model_type == "t5_fallback":
                return self._generate_t5_response(prompt)
            else:
                return self._generate_causal_response(prompt)
        except Exception as e:
            print(f"Error generating response: {e}")
            # Try fallback if primary model fails
            if self.model_type != "t5_fallback" and hasattr(self, 'fallback_model') and self.fallback_model is not None:
                try:
                    return self._generate_t5_response(prompt)
                except Exception as e2:
                    print(f"Fallback model also failed: {e2}")
            
            # Final fallback to simple model
            if not hasattr(self, 'simple_model') or self.simple_model is None:
                self.simple_model = self._create_simple_model()
            
            return self.simple_model.generate_response(context, query, retrieved_chunks, conversation_history)
    
    def _create_prompt(self, context: str, query: str, 
                      retrieved_chunks: List[Dict[str, Any]],
                      conversation_history: List[Dict[str, str]] = None) -> str:
        """Create a prompt for the language model."""
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for entry in conversation_history[-4:]:  # Last 4 interactions
                prompt_parts.append(f"Human: {entry.get('user', '')}")
                prompt_parts.append(f"Assistant: {entry.get('assistant', '')}")
            prompt_parts.append("")
        
        # Add retrieved context
        if retrieved_chunks:
            prompt_parts.append("Relevant information:")
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Top 3 chunks
                source = chunk.get('source', 'unknown')
                text = chunk.get('text', '')[:400]  # Limit chunk length
                prompt_parts.append(f"[{source}] {text}")
            prompt_parts.append("")
        
        # Add the current query
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def _generate_causal_response(self, prompt: str) -> str:
        """Generate response using causal language model."""
        # Tokenize input
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncate=True, 
            max_length=512
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            self.model = self.model.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (after the prompt)
        response = generated_text[len(prompt):].strip()
        
        # Clean up response
        response = self._clean_response(response)
        
        return response if response else "I need more information to provide a complete answer."
    
    def _generate_t5_response(self, prompt: str) -> str:
        """Generate response using T5 model."""
        # For T5, format as a question-answering task
        t5_prompt = f"Answer the following question based on the context: {prompt}"
        
        # Tokenize
        inputs = self.fallback_tokenizer.encode(
            t5_prompt,
            return_tensors="pt",
            truncate=True,
            max_length=512
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            self.fallback_model = self.fallback_model.cuda()
        
        # Generate
        with torch.no_grad():
            outputs = self.fallback_model.generate(
                inputs,
                max_length=MAX_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
        
        # Decode
        response = self.fallback_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self._clean_response(response)
        
        return response if response else "I need more information to provide a complete answer."
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive phrases
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines[-3:]:  # Avoid immediate repetition
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Limit length
        if len(response) > 1000:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def _create_simple_model(self):
        """Create a simple inline language model."""
        class SimpleLanguageModel:
            def __init__(self):
                self.is_initialized = True
                self.model_type = "simple_rule_based"
            
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
                        import re
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
                    "current_model_type": "simple_rule_based",
                    "is_initialized": True,
                    "cuda_available": False,
                    "max_tokens": 500
                }
        
        return SimpleLanguageModel()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model_type == "simple_rule_based":
            return self.simple_model.get_model_info()
        
        return {
            "primary_model": LANGUAGE_MODEL,
            "fallback_model": FALLBACK_MODEL,
            "current_model_type": self.model_type,
            "is_initialized": self.is_initialized,
            "cuda_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
            "max_tokens": MAX_TOKENS
        } 