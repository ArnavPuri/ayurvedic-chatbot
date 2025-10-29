"""
Document Processor - Complete Pipeline - Phase 3
Orchestrates: Loading -> Preprocessing -> Chunking -> Embeddings -> Vector Storage
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import hashlib
import json
import numpy as np

# Import our modules
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
    embeddings_generated: bool = False  # Phase 3: Track if embeddings are generated


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline - Phase 3
    
    Flow: Upload -> Load -> Preprocess -> Chunk -> Embed -> Vector Store
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_sanskrit: bool = True,
        use_vector_store: bool = True,
        embedding_model: str = 'all-MiniLM-L6-v2',
        vector_store_path: str = './data/chroma_db'
    ):
        """
        Initialize document processor with all components (Phase 3)
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            preserve_sanskrit: Whether to preserve Sanskrit diacritics
            use_vector_store: Enable Phase 3 vector features
            embedding_model: Model for embeddings
            vector_store_path: Path for ChromaDB storage
        """
        # Import components
        from app.rag.text_preprocessor import TextPreprocessor
        from app.rag.text_chunker import TextChunker
        from app.rag.document_loader import DocumentLoader
        
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor(preserve_sanskrit=preserve_sanskrit)
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Phase 3: Initialize embeddings and vector store
        self.use_vector_store = use_vector_store
        
        if use_vector_store:
            try:
                from app.rag.embeddings import EmbeddingGenerator
                from app.rag.vector_store import VectorStore
                
                self.embedding_generator = EmbeddingGenerator(
                    model_name=embedding_model
                )
                
                self.vector_store = VectorStore(
                    collection_name='ayurvedic_documents',
                    persist_directory=vector_store_path,
                    embedding_dimension=self.embedding_generator.dimension
                )
                
                logger.info("Phase 3 features enabled: Embeddings + Vector Store")
                
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Phase 3 features: {str(e)}"
                    "\nFalling back to Phase 2 (keyword search)"
                )
                self.use_vector_store = False
                self.embedding_generator = None
                self.vector_store = None
        else:
            self.embedding_generator = None
            self.vector_store = None
            logger.info("Phase 2 mode: Keyword search only")
        
        # In-memory storage for document metadata
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
            
            # Step 7: Phase 3 - Generate embeddings and store in vector DB
            embeddings_generated = False
            
            if self.use_vector_store and self.embedding_generator:
                logger.info("Step 7: Generating embeddings...")
                
                try:
                    # Extract chunk texts for embedding
                    chunk_texts = [chunk['content'] for chunk in chunks_dict]
                    
                    # Generate embeddings for all chunks
                    embeddings = self.embedding_generator.encode_documents(
                        chunk_texts,
                        batch_size=32,
                        show_progress=False
                    )
                    
                    # Prepare metadata for vector store
                    vector_metadatas = []
                    vector_ids = []
                    
                    for i, chunk in enumerate(chunks_dict):
                        # Create unique ID for each chunk
                        chunk_id = f"{doc_id}_chunk_{i}"
                        vector_ids.append(chunk_id)
                        
                        # Prepare metadata (flatten nested structures)
                        meta = {
                            'doc_id': doc_id,
                            'filename': filename,
                            'chunk_index': chunk['chunk_index'],
                            'token_count': chunk['token_count'],
                            'start_char': chunk['start_char'],
                            'end_char': chunk['end_char']
                        }
                        vector_metadatas.append(meta)
                    
                    # Add to vector store
                    self.vector_store.add_documents(
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=vector_metadatas,
                        ids=vector_ids
                    )
                    
                    embeddings_generated = True
                    logger.info(
                        f"Successfully generated and stored {len(embeddings)} embeddings"
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    logger.warning("Document processed but embeddings failed")
            
            # Step 8: Create final processed document
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename=filename,
                content_hash=content_hash,
                original_text=combined_text,
                chunks=chunks_dict,
                metadata=combined_metadata,
                processing_stats=processing_stats,
                embeddings_generated=embeddings_generated
            )
            
            # Store metadata in memory
            self.documents[doc_id] = processed_doc
            
            logger.info(
                f"Processing complete: {len(chunks_dict)} chunks created"
                f"{' with embeddings' if embeddings_generated else ''}"
            )
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
        """Delete a document and its embeddings"""
        if doc_id in self.documents:
            # Delete from vector store if Phase 3 enabled
            if self.use_vector_store and self.vector_store:
                try:
                    self.vector_store.delete_by_metadata({'doc_id': doc_id})
                    logger.info(f"Deleted embeddings for: {doc_id}")
                except Exception as e:
                    logger.warning(f"Error deleting embeddings: {str(e)}")
            
            # Delete from memory
            del self.documents[doc_id]
            logger.info(f"Deleted document: {doc_id}")
            return True
        return False
    
    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        use_semantic: bool = True
    ) -> List[Dict]:
        """
        Search across all chunks - Phase 3 uses semantic search!
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_semantic: Use semantic search (Phase 3) vs keyword (Phase 2)
            
        Returns:
            List of matching chunks with scores
        """
        # Phase 3: Semantic search with embeddings
        if use_semantic and self.use_vector_store and self.embedding_generator:
            return self._semantic_search(query, top_k)
        
        # Phase 2: Fallback to keyword search
        return self._keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Semantic search using embeddings (Phase 3)
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                metadata = result['metadata']
                doc_id = metadata.get('doc_id', 'unknown')
                
                formatted_results.append({
                    'doc_id': doc_id,
                    'filename': metadata.get('filename', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'content': result['content'],
                    'score': result['similarity'],
                    'distance': result['distance'],
                    'metadata': metadata,
                    'search_type': 'semantic'
                })
            
            logger.info(f"Semantic search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            logger.warning("Falling back to keyword search")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Keyword-based search (Phase 2 fallback)
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of results with relevance scores
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
                        'metadata': chunk['metadata'],
                        'search_type': 'keyword'
                    })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get overall statistics (Phase 3 enhanced)"""
        if not self.documents:
            base_stats = {
                'total_documents': 0,
                'phase': 3 if self.use_vector_store else 2
            }
            
            # Add vector store stats if available
            if self.use_vector_store and self.vector_store:
                base_stats.update(self.vector_store.get_stats())
            
            return base_stats
        
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        total_chars = sum(len(doc.original_text) for doc in self.documents.values())
        
        # Count Ayurvedic documents
        ayurvedic_docs = sum(
            1 for doc in self.documents.values()
            if doc.metadata.get('has_ayurvedic_content', False)
        )
        
        # Count documents with embeddings
        docs_with_embeddings = sum(
            1 for doc in self.documents.values()
            if doc.embeddings_generated
        )
        
        stats = {
            'phase': 3 if self.use_vector_store else 2,
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'ayurvedic_documents': ayurvedic_docs,
            'avg_chunks_per_doc': total_chunks // len(self.documents),
            'documents_with_embeddings': docs_with_embeddings,
            'embeddings_enabled': self.use_vector_store
        }
        
        # Add vector store stats if available
        if self.use_vector_store and self.vector_store:
            stats['vector_store'] = self.vector_store.get_stats()
        
        # Add embedding model info if available
        if self.embedding_generator:
            stats['embedding_model'] = self.embedding_generator.get_info()
        
        return stats
    
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