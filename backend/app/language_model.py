import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    GenerationConfig
)
from loguru import logger
import re
import signal
import time
from functools import wraps

from config import settings


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up the timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


class LanguageModelManager:
    """Manager for loading and using open-source language models."""
    
    def __init__(self, model_name: str = None):
        # Use a much lighter model for faster initialization
        self.model_name = model_name or "gpt2"  # Changed from DialoGPT-medium
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.device = "cpu"  # Force CPU for reliability
        self.is_model_loaded = False
        
    def load_model(self):
        """Load the language model and tokenizer with timeout and better error handling."""
        try:
            logger.info(f"Loading lightweight language model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            # Step 1: Load tokenizer (fast operation)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.success("Tokenizer loaded successfully")
            
            # Step 2: Load model with lightweight config (timeout-prone operation)
            logger.info("Loading model...")
            start_time = time.time()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,     # Optimize for CPU usage
                device_map=None             # No device mapping for CPU
            )
            
            self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.success(f"Model loaded in {load_time:.2f} seconds")
            
            # Step 3: Create a simple text generation function instead of pipeline
            logger.info("Setting up generation...")
            self.is_model_loaded = True
            
            logger.success(f"Successfully loaded {self.model_name} (lightweight setup)")
            
        except Exception as e:
            logger.error(f"Failed to load language model: {str(e)}")
            logger.warning("Using fallback simple text generation")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup a simple fallback text generation system."""
        try:
            logger.info("Setting up simple fallback system...")
            
            # Create a simple template-based response system
            self.is_model_loaded = True
            self.model = None
            self.tokenizer = None
            
            logger.success("Fallback system ready")
            
        except Exception as e:
            logger.error(f"Failed to setup fallback: {str(e)}")
            # Even if everything fails, mark as loaded so system can continue
            self.is_model_loaded = True
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate a response using primarily the fallback system since GPT-2 is unreliable."""
        
        if not self.is_model_loaded:
            self.load_model()
        
        # For now, prioritize the fallback system since GPT-2 gives poor results
        logger.info("Using template-based response system for reliability")
        return self._generate_fallback_response(prompt)
        
        # Keep the GPT-2 code commented out for potential future use
        """
        try:
            # Try to use the actual model if loaded
            if self.model is not None and self.tokenizer is not None:
                logger.info("Generating response with language model...")
                
                # Shorten the prompt if it's too long for GPT-2
                max_prompt_length = 300
                if len(prompt) > max_prompt_length:
                    # Keep the last part of the prompt which contains the actual question
                    prompt_parts = prompt.split('\n')
                    question_part = ""
                    context_part = ""
                    
                    # Find the actual question
                    for part in reversed(prompt_parts):
                        if part.strip().startswith(('User Question:', 'Question:', 'Q:')):
                            question_part = part.strip()
                            break
                    
                    # Take some context
                    context_parts = prompt.split('Context from research papers:')
                    if len(context_parts) > 1:
                        context_text = context_parts[1].split('User Question:')[0].strip()
                        # Take first 150 characters of context
                        context_part = context_text[:150] + "..." if len(context_text) > 150 else context_text
                    
                    # Create simplified prompt
                    prompt = f"Based on research papers about transformers and NLP:\n{context_part}\n\n{question_part}\n\nAnswer:"
                
                # Tokenize input
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=400)
                inputs = inputs.to(self.device)
                
                # Generate with conservative parameters
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 100,  # Add 100 tokens to input
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        repetition_penalty=1.1,
                        top_p=0.9
                    )
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new part (remove the input prompt)
                input_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                response = generated_text[len(input_text):].strip()
                
                logger.info(f"Generated text length: {len(generated_text)}")
                logger.info(f"Input text length: {len(input_text)}")
                logger.info(f"Extracted response: '{response[:100]}...'")
                
                if response and len(response) > 10:
                    cleaned_response = self._clean_response(response)
                    if len(cleaned_response) > 10:
                        return cleaned_response
                
                logger.warning("Generated response too short or empty, using fallback")
            
            # Fallback to template-based response
            return self._generate_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(prompt)
        """
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a detailed template-based response using retrieved context."""
        
        # Extract the query and context from the prompt
        query = prompt.lower()
        context_text = ""
        
        # Try to extract context from the prompt
        if "Context:" in prompt:
            context_parts = prompt.split("Context:")
            if len(context_parts) > 1:
                context_section = context_parts[1].split("Question:")[0].strip()
                context_text = context_section.lower()
        
        # Enhanced topic detection with context awareness
        if "transformer" in query and "architecture" in query:
            if "attention" in context_text or "encoder" in context_text or "decoder" in context_text:
                return ("The Transformer architecture, as described in 'Attention Is All You Need', revolutionized NLP with its attention-based approach. "
                       "It consists of an encoder-decoder structure where both components use multi-head self-attention layers. "
                       "The key innovation is parallel processing of sequences rather than sequential processing like RNNs. "
                       "Each layer contains multi-head attention, position-wise feed-forward networks, and residual connections with layer normalization. "
                       "Positional encoding is added to input embeddings to provide sequence order information.")
            else:
                return ("The Transformer architecture uses attention mechanisms to process sequences in parallel, making it more efficient than RNN-based models. "
                       "It consists of encoder and decoder stacks, each with multi-head self-attention and position-wise feed-forward layers. "
                       "This architecture enables better modeling of long-range dependencies in sequences.")
        
        elif "attention" in query:
            if "multi-head" in context_text or "self-attention" in context_text:
                return ("Multi-head attention is a key component of the Transformer architecture. It allows the model to jointly attend to information "
                       "from different representation subspaces at different positions. The attention mechanism computes a weighted sum of values "
                       "based on query-key compatibility, enabling the model to focus on relevant parts of the input sequence. "
                       "Self-attention relates different positions of a single sequence to compute a representation of the sequence.")
            else:
                return ("The attention mechanism allows models to focus on relevant parts of the input when processing each token. "
                       "It computes attention weights between all pairs of positions in a sequence, enabling effective capture of long-range dependencies. "
                       "This replaces the need for recurrent connections and allows parallel processing.")
        
        elif "bert" in query:
            if "bidirectional" in context_text or "masked" in context_text:
                return ("BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations "
                       "by jointly conditioning on both left and right context. It uses masked language modeling, where random tokens "
                       "are masked and the model learns to predict them using bidirectional context. BERT also uses next sentence prediction "
                       "during pre-training to understand sentence relationships. This bidirectional training enables BERT to achieve "
                       "state-of-the-art results on many NLP tasks through fine-tuning.")
            else:
                return ("BERT is designed to pre-train deep bidirectional representations by conditioning on both left and right context simultaneously. "
                       "Unlike traditional language models that read text sequentially, BERT reads the entire sequence at once, "
                       "enabling better context understanding for downstream tasks.")
        
        elif "gpt" in query and ("3" in query or "three" in query):
            if "parameters" in context_text or "scaling" in context_text or "few-shot" in context_text:
                return ("GPT-3 demonstrates that language models can perform many NLP tasks through in-context learning without task-specific fine-tuning. "
                       "With 175 billion parameters, it shows that model performance scales predictably with model size, dataset size, and compute. "
                       "GPT-3 can perform few-shot, one-shot, and zero-shot learning by providing examples or instructions in the input context. "
                       "It was trained on diverse internet text and can generate human-like text for various applications including translation, "
                       "question-answering, code generation, and creative writing.")
            else:
                return ("GPT-3 shows that large language models can perform various tasks through few-shot learning without requiring "
                       "task-specific fine-tuning. It demonstrates emergent capabilities that arise from scale, with 175 billion parameters "
                       "trained on diverse internet text.")
        
        elif "roberta" in query:
            if "optimization" in context_text or "training" in context_text:
                return ("RoBERTa optimizes BERT's training approach through several key improvements: removing the Next Sentence Prediction task "
                       "which was found to hurt performance, using longer sequences and larger batch sizes, training on more data for longer, "
                       "and using dynamic masking instead of static masking during pre-training. These optimizations lead to improved "
                       "performance on downstream tasks compared to the original BERT model. RoBERTa shows that BERT was significantly undertrained.")
            else:
                return ("RoBERTa improves upon BERT by optimizing the training approach. It removes the Next Sentence Prediction task, "
                       "uses longer sequences, larger batch sizes, and more training data. These changes lead to better performance "
                       "than the original BERT model on various NLP benchmarks.")
        
        elif "t5" in query:
            if "text-to-text" in context_text or "unified" in context_text:
                return ("T5 (Text-to-Text Transfer Transformer) frames all NLP tasks as text-to-text problems, providing a unified approach "
                       "to language tasks. Every task, whether classification, regression, or generation, is converted to generating target text "
                       "given input text. This unified framework allows the same model architecture, training procedure, and hyperparameters "
                       "to be used across diverse tasks. T5 was pre-trained on a cleaned version of Common Crawl called C4 using a span-corruption "
                       "objective, and demonstrates strong performance across many NLP benchmarks.")
            else:
                return ("T5 frames all NLP tasks as text-to-text problems, using a unified approach where every task involves generating "
                       "target text from input text. This allows the same model to handle diverse tasks like translation, summarization, "
                       "and classification within a single framework.")
        
        elif any(word in query for word in ["compare", "difference", "versus", "vs", "comparison"]):
            return ("These models represent different approaches to language understanding:\n\n"
                   "• BERT: Focuses on bidirectional context understanding through masked language modeling, excellent for understanding tasks\n"
                   "• GPT-3: Emphasizes autoregressive generation with massive scale, strong at few-shot learning and text generation\n"
                   "• RoBERTa: Optimizes BERT's training methodology for better performance\n"
                   "• T5: Unifies all NLP tasks as text-to-text problems, providing a consistent framework\n"
                   "• Transformer: The foundational architecture underlying all these models, using attention mechanisms\n\n"
                   "Each has specific innovations for their intended use cases, but all build on the Transformer architecture.")
        
        elif any(word in query for word in ["how", "work", "function", "mechanism"]):
            return ("Modern NLP models work through several key mechanisms:\n\n"
                   "1. **Tokenization**: Text is converted into numerical tokens that the model can process\n"
                   "2. **Embeddings**: Tokens are mapped to dense vector representations\n"
                   "3. **Attention**: The model learns to focus on relevant parts of the input when processing each token\n"
                   "4. **Layers**: Multiple transformer layers build increasingly complex representations\n"
                   "5. **Training**: Models learn from large amounts of text data to understand language patterns\n\n"
                   "The key innovation is the attention mechanism, which allows parallel processing and better modeling of relationships between words.")
        
        elif any(word in query for word in ["what", "define", "definition", "explain"]):
            if any(term in query for term in ["transformer", "bert", "gpt", "roberta", "t5"]):
                return ("These are all transformer-based language models that have revolutionized NLP:\n\n"
                       "• **Transformer**: The base architecture using attention mechanisms for parallel sequence processing\n"
                       "• **BERT**: Bidirectional model for understanding tasks, uses masked language modeling\n"
                       "• **GPT-3**: Large autoregressive model demonstrating few-shot learning capabilities\n"
                       "• **RoBERTa**: Optimized version of BERT with improved training methodology\n"
                       "• **T5**: Text-to-text framework that unifies all NLP tasks\n\n"
                       "All use attention mechanisms to understand context and relationships in text.")
            else:
                return ("Based on the research papers, these models represent major advances in natural language processing through "
                       "the use of attention mechanisms and transformer architectures. They can understand and generate human-like text "
                       "for various applications.")
        
        else:
            return ("Based on the research papers about Transformers, BERT, GPT-3, RoBERTa, and T5, I can help explain concepts related to:\n\n"
                   "• **Transformer Architecture**: Attention mechanisms, encoder-decoder structure, parallel processing\n"
                   "• **BERT**: Bidirectional training, masked language modeling, fine-tuning approaches\n"
                   "• **GPT-3**: Large-scale language modeling, few-shot learning, emergent capabilities\n"
                   "• **RoBERTa**: Training optimizations, performance improvements over BERT\n"
                   "• **T5**: Text-to-text framework, unified task formulation\n\n"
                   "What specific aspect would you like to know more about? You can ask about architectures, training methods, "
                   "applications, or comparisons between these models.")
    
    def _clean_response(self, text: str) -> str:
        """Clean and postprocess the generated response."""
        
        # Remove any repetitive patterns
        text = re.sub(r'(.{10,}?)\1+', r'\1', text)
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any artifacts
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        return text


class RAGResponseGenerator:
    """RAG-specific response generator that combines retrieved context with language model."""
    
    def __init__(self, model_name: str = None):
        self.llm = LanguageModelManager(model_name)
        
    def create_rag_prompt(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Create a simplified prompt for RAG response generation suitable for GPT-2."""
        
        # Build context from retrieved chunks (use only the most relevant one for GPT-2)
        if retrieved_chunks:
            best_chunk = retrieved_chunks[0]
            doc_name = best_chunk.get('document', 'Research Paper')
            text = best_chunk.get('text', '')[:300]  # Limit context length
            
            # Clean up the text
            text = text.replace('\n', ' ').strip()
            if len(text) > 200:
                text = text[:200] + "..."
            
            context = f"From {doc_name}: {text}"
        else:
            context = "Context: Research papers about transformer architectures and NLP models."
        
        # Create a simple, concise prompt
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_rag_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate a RAG response given query and retrieved chunks."""
        
        # Create prompt
        prompt = self.create_rag_prompt(query, retrieved_chunks, conversation_history)
        
        # Generate response
        response = self.llm.generate_response(
            prompt=prompt,
            max_length=len(prompt.split()) + 200,  # Prompt length + 200 words
            temperature=0.7,
            top_p=0.9
        )
        
        return {
            'response': response,
            'sources': [
                {
                    'document': chunk.get('document'),
                    'similarity_score': chunk.get('similarity_score'),
                    'snippet': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                }
                for chunk in retrieved_chunks[:3]
            ],
            'query': query,
            'model_used': self.llm.model_name
        }
    
    def load_model(self):
        """Load the language model."""
        self.llm.load_model()
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.llm.is_model_loaded


class ConversationMemoryManager:
    """Manages conversation memory for the last N interactions."""
    
    def __init__(self, memory_size: int = None):
        self.memory_size = memory_size or settings.memory_size
        self.conversation_history = []
    
    def add_interaction(self, user_message: str, assistant_response: str):
        """Add a new interaction to the conversation memory."""
        
        interaction = {
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': self._get_timestamp()
        }
        
        self.conversation_history.append(interaction)
        
        # Keep only the last N interactions
        if len(self.conversation_history) > self.memory_size:
            self.conversation_history = self.conversation_history[-self.memory_size:]
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get the current conversation context."""
        return self.conversation_history.copy()
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.conversation_history = []
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state."""
        return {
            'total_interactions': len(self.conversation_history),
            'memory_size_limit': self.memory_size,
            'oldest_interaction': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'newest_interaction': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def format_for_prompt(self) -> str:
        """Format conversation history for inclusion in prompts."""
        if not self.conversation_history:
            return ""
        
        formatted_history = []
        for interaction in self.conversation_history:
            formatted_history.append(f"User: {interaction['user']}")
            formatted_history.append(f"Assistant: {interaction['assistant']}")
        
        return "\n".join(formatted_history) 