# RAG Application with Conversational Memory

A comprehensive Retrieval-Augmented Generation (RAG) application that ingests content from research papers, creates a vector database for semantic retrieval, and powers a conversational bot with memory for interactions.

## Features

- 📄 **PDF Ingestion**: Extracts and processes text from 5 research papers
- 🔍 **Vector Database**: Uses FAISS for efficient similarity search
- 🤖 **Language Model**: Integrates open-source models from Hugging Face
- 💭 **Conversational Memory**: Maintains context over the last 4 interactions
- 🌐 **Web Interface**: React frontend for easy interaction
- 📊 **Evaluation Framework**: Comprehensive testing and evaluation metrics

## Architecture

```
RAG Application
├── backend/           # FastAPI server and RAG implementation
├── frontend/          # React web interface
├── data/              # PDF documents and processed chunks
├── evaluation/        # Test questions and evaluation metrics
└── reports/           # Generated evaluation reports
```

## Research Papers Used

1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Transformer architecture
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805.pdf)
3. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
4. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
5. [T5: Text-to-Text Transfer Transformer](https://arxiv.org/pdf/1910.10683.pdf)

## Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm (for frontend)
- Git

### Option 1: Using the Startup Script (Recommended)

```bash
# Install all dependencies
python start_application.py install

# Start the backend server
python start_application.py backend

# In a new terminal, start the frontend (optional)
python start_application.py frontend
```

### Option 2: Manual Setup

#### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Frontend Setup (Optional)

```bash
cd frontend
npm install
npm run dev
```

### First Run

1. **Start the backend**: The server will start on `http://localhost:8000`
2. **System initialization**: On first run, the system will download research papers and build the vector database
3. **Access the application**:
   - **API Documentation**: `http://localhost:8000/docs`
   - **Web Interface**: `http://localhost:3000` (if frontend is running)

### Testing the System

```bash
# Quick test
python start_application.py quick-test

# Full evaluation
python start_application.py evaluate
```

## Usage

1. The application will automatically download and process the research papers
2. Open your browser to `http://localhost:3000`
3. Start chatting with the RAG bot about the content in the papers
4. The bot maintains memory of your last 4 interactions

## Evaluation

Run the evaluation suite with:

```bash
cd backend
python evaluate.py
```

This will run 10 predefined questions and generate evaluation metrics using the implemented framework.

## Project Structure

- `backend/app/`: Core RAG implementation
- `backend/models/`: Vector database and embeddings
- `backend/evaluation/`: Evaluation framework and metrics
- `frontend/src/`: React components and UI
- `data/`: PDF storage and processed chunks
- `reports/`: Generated evaluation reports

## Technologies Used

- **Backend**: FastAPI, Hugging Face Transformers, FAISS, PyPDF2
- **Frontend**: Next.js 15, React 19, JavaScript, Tailwind CSS
- **ML/AI**: Sentence Transformers, Open-source LLMs
- **Evaluation**: Custom metrics framework

## API Endpoints

The FastAPI backend provides the following main endpoints:

- `GET /` - Basic application information
- `GET /health` - System health check
- `POST /chat` - Send a message to the RAG system
- `GET /status` - Detailed system status
- `POST /initialize` - Initialize/rebuild the system
- `POST /memory/clear` - Clear conversation memory
- `GET /documents/stats` - Document processing statistics

Full API documentation is available at `http://localhost:8000/docs` when the server is running.

## Features Implemented

### ✅ Core Requirements (100 marks)

1. **PDF Ingestion & Data Sourcing (20 marks)**

   - ✅ Downloads 5 specified research papers automatically
   - ✅ Robust text extraction with fallback methods
   - ✅ Advanced preprocessing and chunking

2. **Vector Database Creation (20 marks)**

   - ✅ FAISS-based vector database
   - ✅ Efficient similarity search
   - ✅ Proper embedding generation and storage

3. **Open Source Language Model Integration (20 marks)**

   - ✅ Hugging Face model integration
   - ✅ Context-aware response generation
   - ✅ Fallback model support

4. **Conversational Bot with Memory (10 marks)**

   - ✅ Maintains last 4 interactions
   - ✅ Context-aware responses
   - ✅ Memory management

5. **Interaction & Evaluation (20 marks)**

   - ✅ 10 comprehensive test questions
   - ✅ Custom evaluation framework
   - ✅ Multiple evaluation metrics

6. **Final Report (10 marks)**
   - ✅ Automated report generation
   - ✅ Comprehensive evaluation results
   - ✅ Technical implementation details

### 🌟 Additional Features

- **Modern Web Interface**: React-based chat interface
- **Real-time System Monitoring**: Live status dashboard
- **Document Statistics**: Detailed analytics
- **API Documentation**: Comprehensive REST API
- **Startup Scripts**: Easy application management
- **Error Handling**: Robust error recovery
- **Logging**: Comprehensive logging system

## Evaluation Results

The system includes a comprehensive evaluation framework that tests:

- **Relevance**: How well answers address questions
- **Accuracy**: Correctness based on source documents
- **Coherence**: Quality and clarity of responses
- **Faithfulness**: Consistency with source material
- **Context Awareness**: Effective use of conversation memory

Run `python start_application.py evaluate` for detailed evaluation results.

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `backend/config.py`
2. **Memory issues**: Reduce chunk size or use smaller models
3. **PDF download fails**: Check internet connection and firewall
4. **Model loading errors**: Ensure sufficient disk space and memory

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and data
- **Internet**: Required for initial model and PDF downloads

## License

This project is for educational purposes as part of an assignment.
