"""
Ayurvedic RAG Chatbot - FastAPI Server (Phase 3 Complete!)
Complete RAG pipeline with semantic search and LLM integration
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our components
try:
    from app.rag.document_processor import DocumentProcessor
    from app.rag.llm_integration import LLMGenerator
except ImportError:
    DocumentProcessor = None
    LLMGenerator = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ayurvedic RAG Chatbot",
    description="AI-powered chatbot for Ayurvedic knowledge using RAG with semantic search",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor (Phase 3 - with embeddings!)
document_processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    preserve_sanskrit=True,
    use_vector_store=True,
    embedding_model='all-MiniLM-L6-v2',
    vector_store_path='./data/chroma_db'
) if DocumentProcessor else None

# Initialize LLM generator (Phase 3)
# Try to initialize, but don't fail if API keys aren't set
llm_generator = None
if LLMGenerator:
    try:
        # Try OpenAI first, fall back to Anthropic
        llm_provider = 'openai' if os.getenv('OPENAI_API_KEY') else 'anthropic'
        
        if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'):
            llm_generator = LLMGenerator(
                provider=llm_provider,
                temperature=0.7,
                max_tokens=1000
            )
            logger.info(f"LLM initialized: {llm_provider}")
        else:
            logger.warning("No LLM API keys found. Running without LLM generation.")
            
    except Exception as e:
        logger.warning(f"Could not initialize LLM: {str(e)}")
        logger.warning("Running without LLM generation. Search will still work!")


# ========== Pydantic Models ==========

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_results: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: str

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str
    processing_stats: dict
    keywords: List[str]

class DocumentSummary(BaseModel):
    id: str
    filename: str
    chunk_count: int
    has_ayurvedic_content: bool
    keywords: List[str]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    results: List[dict]
    query: str
    total_found: int

class HealthResponse(BaseModel):
    status: str
    version: str
    documents_loaded: int
    total_chunks: int


# ========== API Endpoints ==========

@app.get("/", response_model=dict)
async def root():
    """Root endpoint - API info"""
    stats = document_processor.get_stats() if document_processor else {}
    phase = stats.get('phase', 2)
    
    return {
        "message": "Ayurvedic RAG Chatbot API v3.0",
        "status": f"Phase {phase} Active - {'Semantic Search Enabled!' if phase == 3 else 'Keyword Search'}",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "PDF & TXT upload",
            "Smart text chunking",
            "Ayurvedic content preprocessing",
            "Keyword extraction",
            "Semantic similarity search (Phase 3)",
            "Vector embeddings (Phase 3)",
            "ChromaDB storage (Phase 3)",
            "LLM-powered responses (Phase 3)" if llm_generator else "Search-based responses"
        ],
        "llm_enabled": llm_generator is not None,
        "embeddings_enabled": phase == 3
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system stats"""
    if not document_processor:
        return HealthResponse(
            status="error",
            version="3.0.0",
            documents_loaded=0,
            total_chunks=0
        )
    
    stats = document_processor.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        documents_loaded=stats.get('total_documents', 0),
        total_chunks=stats.get('total_chunks', 0)
    )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process Ayurvedic documents (Phase 3)
    
    Complete pipeline:
    - Extracts text from PDF/TXT
    - Preprocesses and cleans
    - Chunks intelligently
    - Extracts keywords and metadata
    - Generates vector embeddings (Phase 3)
    - Stores in ChromaDB (Phase 3)
    """
    if not document_processor:
        raise HTTPException(
            status_code=500,
            detail="Document processor not initialized"
        )
    
    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: PDF, TXT. Got: {file.content_type}"
        )
    
    try:
        # Read file content
        logger.info(f"Uploading file: {file.filename}")
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process through complete pipeline
        processed_doc = document_processor.process_file(content, file.filename)
        
        return DocumentUploadResponse(
            doc_id=processed_doc.id,
            filename=processed_doc.filename,
            chunk_count=len(processed_doc.chunks),
            status="processed",
            processing_stats=processed_doc.processing_stats,
            keywords=processed_doc.metadata.get('keywords', [])[:10]  # Top 10
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/documents", response_model=List[DocumentSummary])
async def list_documents():
    """List all uploaded documents with summaries"""
    if not document_processor:
        return []
    
    docs = document_processor.list_documents()
    
    return [
        DocumentSummary(
            id=doc['id'],
            filename=doc['filename'],
            chunk_count=doc['chunk_count'],
            has_ayurvedic_content=doc['has_ayurvedic_content'],
            keywords=doc['keywords']
        )
        for doc in docs
    ]

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get detailed information about a specific document"""
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    doc = document_processor.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": doc.id,
        "filename": doc.filename,
        "chunk_count": len(doc.chunks),
        "metadata": doc.metadata,
        "processing_stats": doc.processing_stats,
        "chunks_preview": [
            {
                "index": chunk['chunk_index'],
                "preview": chunk['content'][:200] + "...",
                "token_count": chunk['token_count']
            }
            for chunk in doc.chunks[:5]  # First 5 chunks
        ]
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    success = document_processor.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": f"Document {doc_id} deleted successfully"}

@app.post("/search", response_model=SearchResult)
async def search_documents(request: SearchRequest):
    """
    Search across all documents using semantic similarity (Phase 3!)
    
    - Uses vector embeddings for semantic understanding
    - Finds contextually similar content
    - Much better than keyword matching
    """
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Phase 3: Semantic search (falls back to keyword if needed)
    results = document_processor.search_chunks(
        request.query,
        top_k=request.top_k,
        use_semantic=True
    )
    
    return SearchResult(
        results=results,
        query=request.query,
        total_found=len(results)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - Complete RAG with LLM (Phase 3!)
    
    Flow:
    1. Semantic search for relevant chunks
    2. LLM generates contextual answer
    3. Returns answer with sources
    """
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Step 1: Semantic search for relevant chunks (Phase 3)
    search_results = document_processor.search_chunks(
        request.query,
        top_k=request.max_results,
        use_semantic=True
    )
    
    # Format sources
    sources = [
        {
            "filename": result['filename'],
            "chunk_index": result['chunk_index'],
            "score": result.get('score', 0),
            "search_type": result.get('search_type', 'unknown'),
            "preview": result['content'][:200] + "..."
        }
        for result in search_results
    ]
    
    # Generate conversation ID
    import uuid
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    
    # Step 2: Generate answer
    if not search_results:
        answer = (
            "I couldn't find any relevant information in the uploaded documents. "
            "Please try rephrasing your question or upload more Ayurvedic content."
        )
    elif llm_generator:
        # Phase 3: LLM-powered response generation
        try:
            logger.info(f"Generating LLM response for: {request.query}")
            
            response = llm_generator.generate_response(
                query=request.query,
                context_chunks=search_results,
                conversation_history=None  # TODO: Add conversation history in future
            )
            
            answer = response['answer']
            logger.info(f"LLM response generated using {response['model']}")
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Fallback to showing search results
            answer = (
                f"Found {len(search_results)} relevant passages. "
                f"(LLM unavailable - showing top result)\n\n"
                f"{search_results[0]['content'][:400]}..."
            )
    else:
        # Phase 2 fallback: Just show search results
        answer = (
            f"Found {len(search_results)} relevant passages. "
            f"Here's the most relevant content:\n\n"
            f"{search_results[0]['content'][:400]}...\n\n"
            f"(Note: Enable LLM for full AI-generated responses)"
        )
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id
    )

@app.get("/stats")
async def get_stats():
    """Get overall system statistics"""
    if not document_processor:
        return {"error": "Processor not initialized"}
    
    return document_processor.get_stats()


# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )