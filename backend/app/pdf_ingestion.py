import os
import re
import requests
from typing import List, Dict, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber
from loguru import logger
from tqdm import tqdm
import json

from config import settings


class PDFIngestionPipeline:
    """Pipeline for downloading, extracting, and preprocessing PDF documents."""
    
    def __init__(self):
        self.pdf_dir = Path(settings.pdf_dir)
        self.chunks_dir = Path(settings.chunks_dir)
        
    async def download_pdfs(self) -> List[str]:
        """Download all PDF documents from the configured URLs."""
        downloaded_files = []
        
        for url, filename in zip(settings.pdf_urls, settings.pdf_names):
            filepath = self.pdf_dir / filename
            
            if filepath.exists():
                logger.info(f"PDF already exists: {filename}")
                downloaded_files.append(str(filepath))
                continue
                
            try:
                logger.info(f"Downloading {filename} from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as file:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                
                downloaded_files.append(str(filepath))
                logger.success(f"Successfully downloaded: {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                
        return downloaded_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using multiple methods for robustness."""
        
        def extract_with_pypdf2(pdf_path: str) -> str:
            """Extract text using PyPDF2."""
            text = ""
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
            return text
        
        def extract_with_pdfplumber(pdf_path: str) -> str:
            """Extract text using pdfplumber for better layout handling."""
            text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed for {pdf_path}: {str(e)}")
            return text
        
        # Try pdfplumber first (better for layout), fallback to PyPDF2
        text = extract_with_pdfplumber(pdf_path)
        if not text or len(text.strip()) < 100:
            logger.info(f"Fallback to PyPDF2 for {pdf_path}")
            text = extract_with_pypdf2(pdf_path)
        
        if not text or len(text.strip()) < 100:
            logger.error(f"Failed to extract meaningful text from {pdf_path}")
            return ""
        
        logger.info(f"Extracted {len(text)} characters from {Path(pdf_path).name}")
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'cid:\d+', '', text)  # Remove CID references
        text = re.sub(r'\x00', '', text)      # Remove null bytes
        
        # Clean up common formatting issues
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenated line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
        # Remove page headers/footers patterns (common in academic papers)
        lines = text.split('\n')
        cleaned_lines = []
        total_lines = len(lines)
        skipped_lines = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip only very obvious headers/footers (much less aggressive)
            skip_line = False
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                skip_line = True
            # Skip pure page numbers
            elif re.match(r'^\d+$', line) and len(line) <= 3:
                skip_line = True
            # Skip very long all-caps headers (but allow shorter ones)
            elif re.match(r'^[A-Z\s]{20,}$', line):
                skip_line = True
            # Skip pure arxiv/preprint lines
            elif re.match(r'^arxiv:\d+\.\d+', line.lower()):
                skip_line = True
            elif line.lower().strip() == 'preprint':
                skip_line = True
                
            if skip_line:
                skipped_lines += 1
            else:
                cleaned_lines.append(line)
        
        logger.info(f"Preprocessing: kept {len(cleaned_lines)}/{total_lines} lines (skipped {skipped_lines})")
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Max 2 consecutive newlines
        cleaned_text = cleaned_text.strip()
        
        logger.info(f"Preprocessed text length: {len(cleaned_text)} (original: {len(text)})")
        
        return cleaned_text
    
    def create_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, any]]:
        """Split text into overlapping chunks for embedding."""
        
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        logger.info(f"Creating chunks from text of length: {len(text)}")
        
        # Split by sentences first for better semantic chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Fallback: if no sentences found or only one sentence, split by paragraphs
        if len(sentences) <= 1:
            logger.warning("Sentence splitting failed, using paragraph splitting")
            sentences = text.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]
            logger.info(f"Split into {len(sentences)} paragraphs instead")
        
        # Final fallback: if still no good split, use word-based chunking
        if len(sentences) <= 1 and len(text) > chunk_size:
            logger.warning("Paragraph splitting failed, using word-based chunking")
            words = text.split()
            words_per_chunk = chunk_size // 10  # Rough estimate
            sentences = []
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                sentences.append(' '.join(chunk_words))
            logger.info(f"Created {len(sentences)} word-based chunks")
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'size': current_size,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'size': current_size,
                'chunk_id': len(chunks)
            })
        
        if chunks:
            avg_size = sum(c['size'] for c in chunks) / len(chunks)
            logger.info(f"Created {len(chunks)} chunks with average size {avg_size:.0f} characters")
        else:
            logger.warning("No chunks created - creating single chunk from entire text")
            # Create a single chunk from the entire text as last resort
            if text.strip():
                chunks = [{
                    'text': text.strip(),
                    'size': len(text.strip()),
                    'chunk_id': 0
                }]
                logger.info(f"Created 1 fallback chunk with {len(text.strip())} characters")
        
        return chunks
    
    async def process_all_pdfs(self) -> Dict[str, List[Dict[str, any]]]:
        """Download and process all PDFs, returning chunks for each document."""
        
        # Download PDFs
        pdf_files = await self.download_pdfs()
        
        all_document_chunks = {}
        
        for pdf_path in pdf_files:
            document_name = Path(pdf_path).stem
            
            logger.info(f"Processing {document_name}")
            
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                logger.error(f"No text extracted from {document_name}")
                continue
            
            # Preprocess text
            cleaned_text = self.preprocess_text(raw_text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text)
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    'document': document_name,
                    'source_file': pdf_path,
                    'document_length': len(cleaned_text)
                })
            
            all_document_chunks[document_name] = chunks
            
            # Save chunks to file for inspection
            chunks_file = self.chunks_dir / f"{document_name}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            logger.success(f"Processed {document_name}: {len(chunks)} chunks")
        
        return all_document_chunks
    
    def get_processing_stats(self, all_chunks: Dict[str, List[Dict[str, any]]]) -> Dict[str, any]:
        """Generate statistics about the processed documents."""
        
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        total_chars = sum(
            sum(chunk['size'] for chunk in chunks) 
            for chunks in all_chunks.values()
        )
        
        stats = {
            'total_documents': len(all_chunks),
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': total_chars / total_chunks if total_chunks > 0 else 0,
            'documents': {}
        }
        
        for doc_name, chunks in all_chunks.items():
            stats['documents'][doc_name] = {
                'chunks': len(chunks),
                'characters': sum(chunk['size'] for chunk in chunks),
                'average_chunk_size': sum(chunk['size'] for chunk in chunks) / len(chunks) if chunks else 0
            }
        
        return stats 