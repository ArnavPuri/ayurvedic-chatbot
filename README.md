# Ayurvedic RAG Chatbot

An AI-powered chatbot for Ayurvedic knowledge using Retrieval Augmented Generation (RAG). Built with FastAPI, this application processes Ayurvedic documents and provides intelligent responses based on uploaded content.

## Features

### Phase 2 Complete ✨

- **Document Processing Pipeline**
  - Upload PDF and TXT files
  - Intelligent text chunking with configurable overlap
  - Ayurvedic-specific text preprocessing
  - Sanskrit character preservation
  - Keyword extraction and metadata generation
  - Section detection and structure preservation

- **Search & Retrieval**
  - Keyword-based search across all documents
  - Relevance scoring and ranking
  - Multi-document search support

- **API Endpoints**
  - Document upload and processing
  - Chat interface for querying
  - Document management (list, view, delete)
  - Health checks and system statistics

### Coming in Phase 3

- Vector embeddings for semantic search
- LLM integration for natural language responses
- Vector database integration (ChromaDB ready)
- Enhanced conversation management

## Project Structure

```
ayurvedic-chatbot/
├── app/
│   ├── __init__.py
│   ├── api/              # API route handlers (future)
│   └── rag/              # RAG pipeline modules
│       ├── document_loader.py      # PDF/TXT file loading
│       ├── text_preprocessor.py    # Text cleaning & normalization
│       ├── text_chunker.py          # Intelligent text chunking
│       └── document_processor.py   # Complete pipeline orchestration
├── data/
│   └── documents/        # Uploaded documents storage
├── main.py               # FastAPI application entry point
├── requirements.txt      # Python dependencies
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

Create a `.env` file in the project root (optional for Phase 2):

```env
# API Keys (for Phase 3)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector Database (for Phase 3)
CHROMA_DB_PATH=./data/chroma_db

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

Note: Phase 2 works without API keys. They'll be needed for Phase 3 (LLM integration).

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

#### 4. Search Documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "what are doshas", "top_k": 5}'
```

#### 5. Chat Query

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the three doshas", "max_results": 3}'
```

## Architecture

### Document Processing Pipeline

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

4. **Document Processing** (`document_processor.py`)
   - Orchestrates the complete pipeline
   - Generates document IDs and hashes
   - Stores processed documents in memory
   - Provides search functionality

### API Endpoints

- `GET /` - API information and feature list
- `GET /health` - Health check with system stats
- `POST /upload` - Upload and process documents
- `GET /documents` - List all processed documents
- `GET /documents/{doc_id}` - Get document details
- `DELETE /documents/{doc_id}` - Delete a document
- `POST /search` - Search across documents
- `POST /chat` - Chat interface (keyword search in Phase 2)
- `GET /stats` - Overall system statistics

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

### RAG Stack
- `langchain` - LLM framework (prepared for Phase 3)
- `chromadb` - Vector database (ready for Phase 3)
- `sentence-transformers` - Embeddings (for Phase 3)

### Document Processing
- `pypdf` - PDF extraction
- `python-docx` - Word document support (prepared)

### LLM APIs (for Phase 3)
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client

## Ayurvedic Content Support

The preprocessing pipeline is specifically optimized for Ayurvedic content:

- **Sanskrit Preservation**: Diacritics and special characters are preserved
- **Term Detection**: Recognizes common Ayurvedic terms (doshas, prakriti, panchakarma, etc.)
- **Medical Terminology**: Handles medical condition patterns
- **Structure Awareness**: Maintains document sections and headings

## Roadmap

### Phase 2 (Current)
- ✅ Document loading and processing
- ✅ Text preprocessing and chunking
- ✅ Keyword search
- ✅ Basic API endpoints

### Phase 3 (Next)
- [ ] Vector embeddings generation
- [ ] Semantic similarity search
- [ ] LLM integration for response generation
- [ ] ChromaDB integration for persistent storage
- [ ] Conversation history management
- [ ] Enhanced metadata extraction

### Future Enhancements
- Multi-language support
- Advanced section detection
- Image extraction from PDFs
- Citation and source tracking
- User authentication
- Rate limiting and caching

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

