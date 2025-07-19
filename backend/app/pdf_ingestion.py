"""PDF ingestion and text extraction module."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from config import PDF_SOURCES, PDF_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("⚠️  Requests not available")
    REQUESTS_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    print("⚠️  PyPDF2 not available")
    PYPDF2_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    print("⚠️  pypdf not available")
    PYPDF_AVAILABLE = False


class PDFIngestion:
    """Handles PDF download, text extraction, and chunking."""
    
    def __init__(self):
        self.pdf_dir = PDF_DIR
        self.chunks_dir = CHUNKS_DIR
    
    def download_pdf(self, name: str, url: str) -> Path:
        """Download a PDF from URL if not already exists."""
        pdf_path = self.pdf_dir / f"{name}.pdf"
        
        if pdf_path.exists():
            print(f"PDF already exists: {pdf_path}")
            return pdf_path
        
        if not REQUESTS_AVAILABLE:
            print(f"⚠️  Cannot download {name}: requests not available")
            # Create a placeholder file to prevent repeated attempts
            with open(pdf_path, 'w') as f:
                f.write("Placeholder - requests not available")
            return pdf_path
        
        print(f"Downloading {name} from {url}")
        try:
            import requests
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF with fallback methods."""
        text = ""
        
        # Check if this is a placeholder file
        if pdf_path.stat().st_size < 1000:  # Very small file, likely placeholder
            with open(pdf_path, 'r') as f:
                content = f.read()
                if "Placeholder" in content:
                    print(f"Using sample text for {pdf_path.stem}")
                    return self._get_sample_text(pdf_path.stem)
        
        if not PYPDF2_AVAILABLE and not PYPDF_AVAILABLE:
            print(f"⚠️  PDF extraction libraries not available, using sample text for {pdf_path.stem}")
            return self._get_sample_text(pdf_path.stem)
        
        # Try PyPDF2 first
        if PYPDF2_AVAILABLE:
            try:
                from PyPDF2 import PdfReader
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                if len(text.strip()) > 100:  # Good extraction
                    return self._clean_text(text)
            except Exception as e:
                print(f"PyPDF2 failed for {pdf_path}: {e}")
        
        # Fallback to pypdf
        if PYPDF_AVAILABLE:
            try:
                import pypdf
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                return self._clean_text(text)
            except Exception as e:
                print(f"pypdf also failed for {pdf_path}: {e}")
        
        # Final fallback to sample text
        print(f"Using sample text for {pdf_path.stem}")
        return self._get_sample_text(pdf_path.stem)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        # Remove very short lines (likely artifacts)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
        return '\n'.join(lines)
    
    def _get_sample_text(self, paper_name: str) -> str:
        """Get sample text for papers when PDF extraction is not available."""
        sample_texts = {
            "attention_is_all_you_need": """
            Attention Is All You Need
            
            The Transformer architecture is based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely. 
            The model consists of an encoder and decoder, each composed of a stack of identical layers. Each layer has two sub-layers: 
            a multi-head self-attention mechanism, and a position-wise fully connected feed-forward network. 
            
            The attention function can be described as mapping a query and a set of key-value pairs to an output. 
            Multi-head attention allows the model to jointly attend to information from different representation subspaces 
            at different positions. Self-attention, sometimes called intra-attention, is an attention mechanism relating 
            different positions of a single sequence in order to compute a representation of the sequence.
            
            The Transformer uses multi-head attention in three different ways: encoder-decoder attention, 
            encoder self-attention, and decoder self-attention. The model achieves superior performance on 
            machine translation tasks while being more parallelizable and requiring significantly less time to train.
            """,
            
            "bert_pretraining": """
            BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
            
            BERT introduces a new language representation model that is designed to pre-train deep bidirectional 
            representations from unlabeled text by jointly conditioning on both left and right context in all layers. 
            Unlike previous language models, BERT uses a bidirectional approach that allows it to understand context 
            from both directions in a sentence.
            
            The pre-training procedure uses two unsupervised tasks: Masked Language Model (MLM) and Next Sentence 
            Prediction (NSP). In the MLM task, some percentage of input tokens are masked and the model attempts to 
            predict the masked tokens. The NSP task trains the model to understand relationships between sentences.
            
            BERT's bidirectional nature allows it to capture deeper understanding of language context compared to 
            left-to-right or right-to-left language models. The model can be fine-tuned with just one additional 
            output layer to create state-of-the-art models for a wide range of NLP tasks without substantial 
            task-specific architecture modifications.
            """,
            
            "gpt3_language_models": """
            Language Models are Few-Shot Learners (GPT-3)
            
            GPT-3 demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance. 
            The model has 175 billion parameters and shows strong performance on many NLP tasks without task-specific 
            fine-tuning. Instead, it uses few-shot learning where the model is given a few examples of the task 
            at inference time as conditioning.
            
            The key finding is that larger models are significantly more sample-efficient, requiring fewer examples 
            to achieve good performance on downstream tasks. GPT-3 shows qualitatively different behavior compared 
            to smaller models - it can perform tasks it was never explicitly trained to do.
            
            The model demonstrates emergent capabilities in few-shot settings, including arithmetic, word unscrambling, 
            using novel words in sentences, and performing simple reasoning tasks. The scaling behavior suggests that 
            larger models may continue to improve performance across a wide range of tasks.
            """,
            
            "roberta_optimization": """
            RoBERTa: A Robustly Optimized BERT Pretraining Approach
            
            RoBERTa builds on BERT's language masking strategy and modifies key hyperparameters, removing the 
            Next Sentence Prediction task, training with larger mini-batches and learning rates, and training 
            on more data for longer. The study shows that BERT was significantly undertrained and proposes an 
            improved recipe for training BERT models.
            
            Key modifications include: removing the Next Sentence Prediction objective, training with larger 
            batches, using longer training sequences, dynamically changing the masking pattern applied to 
            training data, and training on a larger dataset. These changes lead to substantial improvements 
            over the original BERT model.
            
            RoBERTa demonstrates that with proper optimization, the BERT pretraining approach can achieve 
            significantly better performance. The model matches or exceeds the performance of more complex 
            architectures on several benchmarks, showing the importance of optimization and training procedures.
            """,
            
            "t5_text_to_text": """
            Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
            
            T5 introduces a unified framework that treats every NLP task as a text-to-text problem. The model 
            is trained on a diverse mixture of unsupervised and supervised tasks, where each task is cast as 
            feeding the model text as input and training it to generate target text.
            
            The text-to-text framework allows the same model, loss function, and hyperparameters to be used 
            across diverse tasks including translation, question answering, classification, and summarization. 
            This unified approach simplifies the comparison of different tasks and enables transfer learning 
            across different domains.
            
            T5 systematically studies transfer learning by exploring various factors including model architectures, 
            pre-training objectives, unlabeled datasets, transfer approaches, and scaling. The study provides 
            insights into when and how transfer learning is most effective, demonstrating that larger models 
            and longer training generally improve performance across tasks.
            """
        }
        
        return sample_texts.get(paper_name, f"Sample text for {paper_name} research paper.")
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_pdf(self, name: str, url: str) -> List[Dict[str, Any]]:
        """Download, extract, and chunk a PDF."""
        print(f"Processing {name}...")
        
        # Check if chunks already exist
        chunks_file = self.chunks_dir / f"{name}_chunks.json"
        if chunks_file.exists():
            print(f"Chunks already exist for {name}")
            with open(chunks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Download and extract
        pdf_path = self.download_pdf(name, url)
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        
        # Create chunk metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "id": f"{name}_chunk_{i}",
                "text": chunk,
                "source": name,
                "chunk_index": i,
                "url": url
            })
        
        # Save chunks
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(chunks)} chunks for {name}")
        return chunk_data
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Process all PDFs and return combined chunks."""
        all_chunks = []
        
        if not REQUESTS_AVAILABLE and not PYPDF2_AVAILABLE and not PYPDF_AVAILABLE:
            print("⚠️  PDF processing libraries not available, using sample data")
        
        for name, url in PDF_SOURCES.items():
            try:
                chunks = self.process_pdf(name, url)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to process {name}: {e}")
                # Create minimal sample chunks if processing fails
                sample_chunks = self._create_sample_chunks(name, url)
                all_chunks.extend(sample_chunks)
                continue
        
        print(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def _create_sample_chunks(self, name: str, url: str) -> List[Dict[str, Any]]:
        """Create sample chunks when PDF processing fails."""
        sample_text = self._get_sample_text(name)
        chunks = self.chunk_text(sample_text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "id": f"{name}_chunk_{i}",
                "text": chunk,
                "source": name,
                "chunk_index": i,
                "url": url
            })
        
        print(f"Created {len(chunks)} sample chunks for {name}")
        return chunk_data 