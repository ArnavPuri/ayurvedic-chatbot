"""
Ayurvedic RAG Chatbot - FastAPI Server (Phase 2 Complete)
Integrated document processing pipeline
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Import our document processor
# In production: from app.rag.document_processor import DocumentProcessor
# For now, assuming it's importable
try:
    from app.rag.document_processor import DocumentProcessor
except ImportError:
    # Fallback if running standalone
    DocumentProcessor = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ayurvedic RAG Chatbot",
    description="AI-powered chatbot for Ayurvedic knowledge using RAG",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor (Phase 2 complete!)
document_processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    preserve_sanskrit=True
) if DocumentProcessor else None


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
    return {
        "message": "Ayurvedic RAG Chatbot API v2.0",
        "status": "Phase 2 Complete - Document processing ready!",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "PDF & TXT upload",
            "Smart text chunking",
            "Ayurvedic content preprocessing",
            "Keyword extraction",
            "Basic search"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system stats"""
    if not document_processor:
        return HealthResponse(
            status="error",
            version="2.0.0",
            documents_loaded=0,
            total_chunks=0
        )
    
    stats = document_processor.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        documents_loaded=stats.get('total_documents', 0),
        total_chunks=stats.get('total_chunks', 0)
    )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process Ayurvedic documents
    
    Now with complete pipeline:
    - Extracts text from PDF/TXT
    - Preprocesses and cleans
    - Chunks intelligently
    - Extracts keywords and metadata
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
    Search across all documents using keyword matching
    (Phase 3 will upgrade this to vector similarity search)
    """
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = document_processor.search_chunks(
        request.query,
        top_k=request.top_k
    )
    
    return SearchResult(
        results=results,
        query=request.query,
        total_found=len(results)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - RAG query processing
    
    Phase 2: Basic search implemented
    Phase 3: Will add vector embeddings + LLM
    """
    if not document_processor:
        raise HTTPException(status_code=500, detail="Processor not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Search for relevant chunks
    search_results = document_processor.search_chunks(
        request.query,
        top_k=request.max_results
    )
    
    # Format sources
    sources = [
        {
            "filename": result['filename'],
            "chunk_index": result['chunk_index'],
            "score": result['score'],
            "preview": result['content'][:200] + "..."
        }
        for result in search_results
    ]
    
    # Generate conversation ID
    conversation_id = request.conversation_id or f"conv_{len(search_results)}"
    
    # Phase 2: Return search results as "answer"
    # Phase 3: Will pass to LLM for actual answer generation
    if search_results:
        answer = f"Found {len(search_results)} relevant passages. Here's the top result:\n\n{search_results[0]['content'][:300]}..."
    else:
        answer = "I couldn't find any relevant information in the uploaded documents. Please try rephrasing your question or upload more Ayurvedic content."
    
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