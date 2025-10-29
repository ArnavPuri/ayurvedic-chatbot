"""
Document Loader Module
Handles PDF and text file processing
"""
from typing import List, Dict, Optional
from pathlib import Path
import pypdf
from dataclasses import dataclass
import logging

# Setup logging (Python's built-in logging is like console.log but better)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# dataclass is like TypeScript interface + class combined
@dataclass
class Document:
    """Represents a processed document"""
    content: str
    metadata: Dict[str, any]
    source: str
    page_number: Optional[int] = None


class DocumentLoader:
    """
    Loads and extracts text from various document formats
    Currently supports: PDF, TXT
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Extract text from PDF file
        Returns a list of Document objects (one per page)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Open PDF file (with statement auto-closes)
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = pypdf.PdfReader(file)
                
                # Get total pages
                num_pages = len(pdf_reader.pages)
                logger.info(f"Processing PDF: {file_path} ({num_pages} pages)")
                
                # Iterate through pages (enumerate is like .map with index)
                for page_num, page in enumerate(pdf_reader.pages):
                    # Extract text from page
                    text = page.extract_text()
                    
                    # Skip empty pages
                    if not text.strip():
                        logger.warning(f"Page {page_num + 1} is empty, skipping")
                        continue
                    
                    # Create Document object
                    doc = Document(
                        content=text,
                        metadata={
                            'source_type': 'pdf',
                            'total_pages': num_pages,
                            'filename': Path(file_path).name
                        },
                        source=file_path,
                        page_number=page_num + 1
                    )
                    
                    documents.append(doc)
                    
                logger.info(f"Successfully extracted {len(documents)} pages")
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        return documents
    
    def load_text(self, file_path: str) -> List[Document]:
        """
        Load plain text file
        
        Args:
            file_path: Path to text file
            
        Returns:
            List containing single Document object
        """
        try:
            # Read text file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if not content.strip():
                raise ValueError("Text file is empty")
            
            doc = Document(
                content=content,
                metadata={
                    'source_type': 'text',
                    'filename': Path(file_path).name,
                    'char_count': len(content)
                },
                source=file_path
            )
            
            logger.info(f"Successfully loaded text file: {file_path}")
            return [doc]
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}")
            raise
    
    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load document from bytes (for file uploads)
        This is what we'll use in our API endpoint
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            
        Returns:
            List of Document objects
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Save temporarily to process
        # In production, you might use tempfile module
        temp_path = f"/tmp/{filename}"
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            
            # Route to appropriate loader
            if file_ext == '.pdf':
                return self.load_pdf(temp_path)
            elif file_ext == '.txt':
                return self.load_text(temp_path)
            
        finally:
            # Clean up temp file (finally always runs, like try-finally in Java)
            try:
                Path(temp_path).unlink()
            except:
                pass
    
    def get_document_stats(self, documents: List[Document]) -> Dict:
        """
        Get statistics about loaded documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {'total_docs': 0}
        
        total_chars = sum(len(doc.content) for doc in documents)
        total_words = sum(len(doc.content.split()) for doc in documents)
        
        return {
            'total_docs': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_words_per_doc': total_words // len(documents) if documents else 0,
            'source_types': list(set(doc.metadata.get('source_type') for doc in documents))
        }


# Example usage (for testing)
if __name__ == "__main__":
    # This block only runs if you execute this file directly
    # Like: python document_loader.py
    
    loader = DocumentLoader()
    
    # Test with a sample file
    try:
        docs = loader.load_text("sample.txt")
        stats = loader.get_document_stats(docs)
        print(f"Loaded documents: {stats}")
    except Exception as e:
        print(f"Test failed: {e}")