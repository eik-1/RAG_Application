# RAG Application

A Retrieval-Augmented Generation system that analyzes research papers and provides conversational Q&A with memory support.

## Overview

This application processes five key NLP research papers and allows users to ask questions about their content through a chat interface. The system maintains conversation history and provides source attribution for all responses.

### Research Papers Included

- Attention Is All You Need (Transformer architecture)
- BERT: Pre-training of Deep Bidirectional Transformers
- Language Models are Few-Shot Learners (GPT-3)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Exploring the Limits of Transfer Learning with T5

## Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for frontend)
- Git

## Installation

### Quick Start

The easiest way to get started is using the provided startup script:

```bash
# Clone and navigate to the project
git clone <repository-url>
cd RAG_Application

# Install all dependencies
python start_application.py install
```

### Manual Setup

If you prefer to set up components individually:

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Option 1: Using Startup Scripts

```bash
# Start backend server
python start_application.py backend

# In a new terminal, start frontend (optional)
python start_application.py frontend
```

### Option 2: Manual Startup

#### Start Backend

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py
```

The backend will start on `http://localhost:8001`

#### Start Frontend (Optional)

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## First Run

On the first startup, the system will:

1. Download the required research papers automatically
2. Process the PDFs and extract text content
3. Build a vector database for semantic search
4. Initialize the language model

This process may take a few minutes depending on your internet connection and system performance.

## Usage

### Web Interface

Navigate to `http://localhost:3000` to access the chat interface where you can:

- Ask questions about the research papers
- View source attribution for responses
- Monitor system status and statistics
- Clear conversation memory

### API Access

The backend provides a REST API at `http://localhost:8001` with endpoints for:

- `/chat` - Send messages to the RAG system
- `/status` - Check system health and statistics
- `/docs` - View API documentation

## Example Questions

Try asking questions like:

- "What is the main innovation in the Transformer architecture?"
- "How does BERT's training differ from GPT?"
- "What are the key improvements in RoBERTa?"
- "Explain T5's text-to-text approach"

## System Requirements

- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and data
- **Network**: Required for initial model downloads

## Troubleshooting

### Common Issues

**Port conflicts**: The default ports are 8001 (backend) and 3000 (frontend). If these are in use, modify the configuration in `backend/config.py`.

**Memory errors**: If you encounter memory issues, the system will automatically fall back to simpler models.

**Network timeouts**: For package installation issues, try:
```bash
cd backend
python install_packages.py
```

**Missing dependencies**: Install packages individually if the batch installation fails:
```bash
pip install fastapi uvicorn pydantic
pip install requests PyPDF2 nltk
```

### Logs and Debugging

- Backend logs appear in the terminal where you started the server
- Frontend logs are available in the browser developer console
- System status can be checked via the web interface or `/health` endpoint

## Development

### Project Structure

```
RAG_Application/
├── backend/                 # FastAPI backend
│   ├── app/                # Core application modules
│   ├── data/               # Downloaded papers and processed data
│   ├── evaluation/         # Test questions and evaluation framework
│   └── main.py            # Application entry point
├── frontend/               # Next.js frontend
│   ├── components/        # React components
│   ├── lib/              # API client and utilities
│   └── app/              # Next.js app router pages
└── start_application.py   # Convenience startup script
```

### Running Tests

```bash
cd backend
python evaluate.py quick     # Quick test with 3 questions
python evaluate.py full      # Full evaluation with 10 questions
```

## Features

- **Automatic PDF processing** from research paper URLs
- **Vector-based semantic search** using FAISS
- **Conversational memory** maintaining last 4 interactions
- **Source attribution** for all responses
- **Real-time system monitoring**
- **Comprehensive evaluation framework**

## Technical Details

- **Backend**: FastAPI with Python
- **Frontend**: Next.js with React
- **Vector Database**: FAISS for similarity search
- **Language Models**: Hugging Face transformers with fallbacks
- **Text Processing**: PyPDF2 for extraction, custom chunking

## License

This project is developed for educational purposes.
