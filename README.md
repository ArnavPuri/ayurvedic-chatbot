# Ayurvedic RAG Chatbot

An AI-powered chatbot for Ayurvedic knowledge using Retrieval Augmented Generation (RAG). Built with FastAPI, this application processes Ayurvedic documents and provides intelligent responses based on uploaded content.

## Features

### 🎉 Phase 3 Complete! 🎉

- **Semantic Search with Vector Embeddings**
  - Vector embeddings using sentence-transformers
  - Semantic similarity matching (not just keywords!)
  - ChromaDB for persistent vector storage
  - True contextual understanding of queries

- **LLM Integration**
  - OpenAI (GPT-4, GPT-3.5) support
  - Anthropic (Claude) support
  - Context-aware response generation
  - Intelligent answers based on retrieved documents

- **Document Processing Pipeline**
  - Upload PDF and TXT files
  - Intelligent text chunking with configurable overlap
  - Ayurvedic-specific text preprocessing
  - Sanskrit character preservation
  - Keyword extraction and metadata generation
  - Section detection and structure preservation
  - **NEW:** Automatic embedding generation
  - **NEW:** Vector storage in ChromaDB

- **Advanced Search & Retrieval**
  - **NEW:** Semantic similarity search
  - **NEW:** Vector-based ranking
  - Multi-document search support
  - Fallback to keyword search if needed

- **API Endpoints**
  - Document upload and processing with embeddings
  - Chat interface with LLM-powered responses
  - Semantic search endpoint
  - Document management (list, view, delete)
  - Health checks and system statistics

### Phase 2 Features (Still Available)

All Phase 2 features remain available as fallbacks:
- Keyword-based search
- Basic text processing
- Document management

## Project Structure

```
ayurvedic-chatbot/
├── app/
│   ├── __init__.py
│   ├── api/              # API route handlers (future)
│   └── rag/              # RAG pipeline modules
│       ├── document_loader.py      # PDF/TXT file loading
│       ├── text_preprocessor.py    # Text cleaning & normalization
│       ├── text_chunker.py         # Intelligent text chunking
│       ├── embeddings.py           # Vector embeddings (Phase 3)
│       ├── vector_store.py         # ChromaDB integration (Phase 3)
│       ├── llm_integration.py      # LLM APIs (Phase 3)
│       └── document_processor.py   # Complete pipeline orchestration
├── data/
│   ├── documents/        # Uploaded documents storage
│   └── chroma_db/        # Vector database (Phase 3)
├── main.py               # FastAPI application entry point
├── requirements.txt      # Python dependencies
├── .env.example          # Environment configuration template
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
```bash
cd ayurvedic-chatbot
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv ayurenv
```

3. **Activate the virtual environment**:

   On macOS/Linux:
```bash
source ayurenv/bin/activate
```

   On Windows:
```bash
ayurenv\Scripts\activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# ============================================================
# LLM API Keys (Phase 3 - Optional but recommended)
# ============================================================
# Get OpenAI key: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Get Anthropic key: https://console.anthropic.com/account/keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# ============================================================
# Vector Database Configuration (Phase 3)
# ============================================================
CHROMA_DB_PATH=./data/chroma_db

# ============================================================
# Embedding Model (Phase 3)
# ============================================================
# Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ============================================================
# LLM Configuration (Phase 3)
# ============================================================
LLM_PROVIDER=openai  # or 'anthropic'
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# ============================================================
# Document Processing
# ============================================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
PRESERVE_SANSKRIT=true

# ============================================================
# Server Configuration
# ============================================================
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# ============================================================
# Feature Flags
# ============================================================
USE_VECTOR_STORE=true
USE_LLM=true
```

**Important Notes:**
- **Without API keys:** The system runs with semantic search but shows retrieved content instead of LLM-generated responses
- **With API keys:** Full Phase 3 experience with intelligent, contextual AI responses
- The app works in Phase 3 mode (semantic search) even without LLM keys!
- ChromaDB data persists between runs in `./data/chroma_db`

## Quick Start (Phase 3)

### Option 1: Full Phase 3 with LLM

1. Set up your API key:
```bash
export OPENAI_API_KEY='your-key-here'
# OR
export ANTHROPIC_API_KEY='your-key-here'
```

2. Start the server:
```bash
python main.py
```

3. Upload a document and chat!

### Option 2: Phase 3 without LLM (Semantic Search Only)

1. Just start the server (no API keys needed):
```bash
python main.py
```

2. You'll get:
   - ✅ Semantic search with vector embeddings
   - ✅ ChromaDB persistent storage
   - ✅ Much better search than keyword matching
   - ⚠️ Responses show relevant content (not AI-generated)

3. Add API keys later to enable LLM responses!

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example API Calls

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

#### 2. Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

#### 3. List All Documents

```bash
curl http://localhost:8000/documents
```

#### 4. Semantic Search (Phase 3)

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "what are doshas", "top_k": 5}'
```

Response includes semantic similarity scores and search type (semantic/keyword).

#### 5. Chat with LLM (Phase 3)

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the three doshas", "max_results": 3}'
```

Returns:
- AI-generated answer (if LLM enabled)
- Source documents with similarity scores
- Conversation ID for tracking

## Architecture

### Document Processing Pipeline (Phase 3)

1. **Document Loading** (`document_loader.py`)
   - Extracts text from PDF files (page by page)
   - Loads plain text files
   - Handles file uploads from bytes

2. **Text Preprocessing** (`text_preprocessor.py`)
   - Unicode normalization (preserves Sanskrit diacritics)
   - Whitespace cleaning
   - OCR error correction
   - Ayurvedic term detection
   - Keyword extraction
   - Section detection

3. **Text Chunking** (`text_chunker.py`)
   - Recursive character splitting
   - Intelligent separator hierarchy (paragraphs → sentences → words)
   - Configurable chunk size and overlap
   - Token estimation
   - Metadata preservation

4. **Embeddings Generation** (`embeddings.py`) **NEW in Phase 3**
   - Generates vector embeddings using sentence-transformers
   - Default: all-MiniLM-L6-v2 (384 dimensions)
   - Batch processing for efficiency
   - Normalized vectors for cosine similarity

5. **Vector Storage** (`vector_store.py`) **NEW in Phase 3**
   - ChromaDB integration for persistent storage
   - Semantic similarity search
   - Metadata filtering and management
   - Automatic document indexing

6. **LLM Integration** (`llm_integration.py`) **NEW in Phase 3**
   - OpenAI and Anthropic support
   - Context-aware response generation
   - Custom prompts for Ayurvedic content
   - Graceful fallbacks

7. **Document Processing** (`document_processor.py`)
   - Orchestrates the complete pipeline
   - Generates document IDs and hashes
   - **NEW:** Generates and stores embeddings
   - **NEW:** Semantic search capability
   - Stores document metadata in memory
   - Provides both semantic and keyword search

### API Endpoints

- `GET /` - API information and feature list
- `GET /health` - Health check with system stats
- `POST /upload` - Upload and process documents (with embeddings in Phase 3)
- `GET /documents` - List all processed documents
- `GET /documents/{doc_id}` - Get document details
- `DELETE /documents/{doc_id}` - Delete a document and its embeddings
- `POST /search` - **Semantic search** across documents (Phase 3)
- `POST /chat` - Chat interface with **LLM-powered responses** (Phase 3)
- `GET /stats` - Overall system statistics including vector store info

## Development

### Running Tests

Individual modules can be tested by running them directly:

```bash
# Test document loader
python app/rag/document_loader.py

# Test preprocessor
python app/rag/text_preprocessor.py

# Test chunker
python app/rag/text_chunker.py

# Test document processor
python app/rag/document_processor.py
```

### Code Style

This project follows Python best practices:
- Type hints for function parameters and returns
- Comprehensive docstrings
- Structured logging
- Error handling and validation
- Modular, testable architecture

## Dependencies

### Core Framework
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### RAG Stack (Phase 3 Active)
- `langchain` - LLM framework and utilities
- `chromadb` - Vector database for embeddings
- `sentence-transformers` - Generate semantic embeddings

### Document Processing
- `pypdf` - PDF extraction
- `python-docx` - Word document support (prepared)

### LLM APIs (Phase 3 Active)
- `openai` - OpenAI API client (GPT-4, GPT-3.5)
- `anthropic` - Anthropic API client (Claude)

## Ayurvedic Content Support

The preprocessing pipeline is specifically optimized for Ayurvedic content:

- **Sanskrit Preservation**: Diacritics and special characters are preserved
- **Term Detection**: Recognizes common Ayurvedic terms (doshas, prakriti, panchakarma, etc.)
- **Medical Terminology**: Handles medical condition patterns
- **Structure Awareness**: Maintains document sections and headings

## Roadmap

### ✅ Phase 1 (Complete)
- Document loading and basic processing
- Text extraction from PDFs

### ✅ Phase 2 (Complete)
- Text preprocessing and chunking
- Keyword search
- API endpoints
- Document management

### ✅ Phase 3 (Complete - Current!)
- ✅ Vector embeddings generation
- ✅ Semantic similarity search
- ✅ LLM integration for response generation
- ✅ ChromaDB integration for persistent storage
- ✅ Enhanced search with fallbacks
- ✅ Comprehensive API updates

### Phase 4 (Future Enhancements)
- [ ] Conversation history management
- [ ] Multi-turn conversations with context
- [ ] Enhanced metadata extraction
- [ ] Multi-language support
- [ ] Advanced section detection
- [ ] Image extraction from PDFs
- [ ] Citation and source tracking
- [ ] User authentication
- [ ] Rate limiting and caching
- [ ] Streaming responses
- [ ] Fine-tuned embeddings for Ayurvedic content

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Maintain code style and documentation
2. Add tests for new features
3. Update this README for significant changes
4. Follow semantic versioning

## License

[Specify your license here]

## Support

For issues, questions, or contributions, please [open an issue](link-to-issues) or contact the maintainers.

## Acknowledgments

Built with modern Python web technologies and designed specifically for Ayurvedic knowledge management.

