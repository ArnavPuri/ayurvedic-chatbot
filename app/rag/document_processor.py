"""
Document Processor - Complete Pipeline
Orchestrates: Loading -> Preprocessing -> Chunking
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import hashlib
import json

# Import our previous modules
# In real project: from app.rag.document_loader import DocumentLoader, Document
# For now, we'll assume they're in the same package

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Complete processed document with all metadata"""
    id: str
    filename: str
    content_hash: str
    original_text: str
    chunks: List[Dict]  # List of chunk dictionaries
    metadata: Dict
    processing_stats: Dict


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline
    
    Flow: Upload -> Load -> Preprocess -> Chunk -> Store
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_sanskrit: bool = True
    ):
        """
        Initialize document processor with all components
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            preserve_sanskrit: Whether to preserve Sanskrit diacritics
        """
        # Initialize our components
        # Note: In real code, import these from their modules
        from app.rag.text_preprocessor import TextPreprocessor
        from app.rag.text_chunker import TextChunker
        from app.rag.document_loader import DocumentLoader
        
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor(preserve_sanskrit=preserve_sanskrit)
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # In-memory storage for processed documents
        # Later we'll replace this with vector DB
        self.documents: Dict[str, ProcessedDocument] = {}
        
        logger.info("DocumentProcessor initialized")
    
    def process_file(
        self,
        file_bytes: bytes,
        filename: str
    ) -> ProcessedDocument:
        """
        Process a file through the complete pipeline
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            
        Returns:
            ProcessedDocument with all chunks and metadata
        """
        logger.info(f"Starting processing for: {filename}")
        
        try:
            # Step 1: Load document
            logger.info("Step 1: Loading document...")
            documents = self.loader.load_from_bytes(file_bytes, filename)
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Combine all pages/sections into one text
            # For PDFs, each page is a separate Document
            combined_text = "\n\n".join(doc.content for doc in documents)
            
            # Step 2: Preprocess
            logger.info("Step 2: Preprocessing text...")
            preprocessed = self.preprocessor.preprocess(
                combined_text,
                extract_metadata=True
            )
            
            # Step 3: Chunk the cleaned text
            logger.info("Step 3: Chunking text...")
            
            # Merge metadata from all sources
            combined_metadata = {
                'filename': filename,
                'source_type': documents[0].metadata.get('source_type'),
                'total_pages': documents[0].metadata.get('total_pages', 1),
                **preprocessed.metadata,
                'keywords': preprocessed.keywords,
                'sections': preprocessed.sections
            }
            
            text_chunks = self.chunker.chunk_text(
                preprocessed.cleaned,
                metadata=combined_metadata
            )
            
            # Step 4: Convert chunks to dictionaries for storage
            chunks_dict = []
            for chunk in text_chunks:
                chunk_dict = {
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'token_count': chunk.token_count
                }
                chunks_dict.append(chunk_dict)
            
            # Step 5: Generate document ID and hash
            doc_id = self._generate_doc_id(filename)
            content_hash = self._hash_content(combined_text)
            
            # Step 6: Collect processing stats
            preprocessing_stats = self.preprocessor.get_preprocessing_stats(preprocessed)
            chunking_stats = self.chunker.get_chunk_stats(text_chunks)
            
            processing_stats = {
                **preprocessing_stats,
                **chunking_stats,
                'pages_processed': len(documents)
            }
            
            # Step 7: Create final processed document
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename=filename,
                content_hash=content_hash,
                original_text=combined_text,
                chunks=chunks_dict,
                metadata=combined_metadata,
                processing_stats=processing_stats
            )
            
            # Store in memory (later: store in vector DB)
            self.documents[doc_id] = processed_doc
            
            logger.info(f"Processing complete: {len(chunks_dict)} chunks created")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            raise
    
    def _generate_doc_id(self, filename: str) -> str:
        """
        Generate unique document ID
        Uses filename + timestamp hash
        """
        import time
        unique_string = f"{filename}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def _hash_content(self, content: str) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get_document(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Retrieve a processed document by ID"""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Dict]:
        """List all processed documents with summary info"""
        summaries = []
        for doc_id, doc in self.documents.items():
            summaries.append({
                'id': doc.id,
                'filename': doc.filename,
                'chunk_count': len(doc.chunks),
                'has_ayurvedic_content': doc.metadata.get('has_ayurvedic_content', False),
                'keywords': doc.metadata.get('keywords', [])[:5],  # Top 5
                'stats': doc.processing_stats
            })
        return summaries
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.info(f"Deleted document: {doc_id}")
            return True
        return False
    
    def search_chunks(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Simple keyword search across all chunks
        (Later: replace with vector similarity search)
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        # Normalize query
        normalized_query = self.preprocessor.normalize_query(query)
        query_terms = set(normalized_query.split())
        
        results = []
        
        # Search through all documents and chunks
        for doc_id, doc in self.documents.items():
            for chunk in doc.chunks:
                # Simple keyword matching (BOW - Bag of Words)
                chunk_text = chunk['content'].lower()
                chunk_terms = set(chunk_text.split())
                
                # Calculate overlap score
                overlap = len(query_terms.intersection(chunk_terms))
                
                if overlap > 0:
                    # Calculate relevance score (Jaccard similarity)
                    union = len(query_terms.union(chunk_terms))
                    score = overlap / union if union > 0 else 0
                    
                    results.append({
                        'doc_id': doc_id,
                        'filename': doc.filename,
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'score': score,
                        'metadata': chunk['metadata']
                    })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        if not self.documents:
            return {'total_documents': 0}
        
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        total_chars = sum(len(doc.original_text) for doc in self.documents.values())
        
        # Count Ayurvedic documents
        ayurvedic_docs = sum(
            1 for doc in self.documents.values()
            if doc.metadata.get('has_ayurvedic_content', False)
        )
        
        return {
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'ayurvedic_documents': ayurvedic_docs,
            'avg_chunks_per_doc': total_chunks // len(self.documents)
        }
    
    def export_document(self, doc_id: str, format: str = 'json') -> str:
        """
        Export processed document
        
        Args:
            doc_id: Document ID
            format: Export format ('json' or 'text')
            
        Returns:
            Exported content as string
        """
        doc = self.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        
        if format == 'json':
            # Convert dataclass to dict
            doc_dict = asdict(doc)
            return json.dumps(doc_dict, indent=2)
        
        elif format == 'text':
            # Export as readable text
            output = [
                f"Document: {doc.filename}",
                f"ID: {doc.id}",
                f"Chunks: {len(doc.chunks)}",
                f"\n{'='*50}\n",
                doc.original_text
            ]
            return '\n'.join(output)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Testing
if __name__ == "__main__":
    # Sample test
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Create sample text file
    sample_content = """
    INTRODUCTION TO AYURVEDA
    
    Ayurveda is an ancient system of medicine from India. It focuses on balance
    between mind, body, and spirit.
    
    The Three Doshas
    
    There are three primary doshas in Ayurveda:
    - Vata (air and space)
    - Pitta (fire and water)  
    - Kapha (earth and water)
    
    Each person has a unique constitution based on these doshas.
    """
    
    # Process the sample
    try:
        result = processor.process_file(
            sample_content.encode(),
            "ayurveda_intro.txt"
        )
        
        print("\n=== Processing Results ===")
        print(f"Document ID: {result.id}")
        print(f"Chunks created: {len(result.chunks)}")
        print(f"Keywords: {result.metadata.get('keywords', [])}")
        print(f"\nFirst chunk:")
        print(result.chunks[0]['content'][:200] + "...")
        
        # Test search
        print("\n=== Search Test ===")
        results = processor.search_chunks("what are doshas")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   {result['content'][:150]}...")
        
        # Overall stats
        print("\n=== Overall Stats ===")
        stats = processor.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()