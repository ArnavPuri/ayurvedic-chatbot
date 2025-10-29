"""
Text Chunking Module
Splits documents into optimal chunks for embedding and retrieval
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    chunk_index: int
    metadata: Dict
    start_char: int
    end_char: int
    token_count: Optional[int] = None


class TextChunker:
    """
    Smart text chunking with configurable strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize chunker with configuration
        
        Args:
            chunk_size: Target size in characters (not tokens!)
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Validation
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks using recursive character splitting
        
        This is the main method you'll use!
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Use recursive splitting for better results
        chunks = self._recursive_split(text)
        
        # Convert to TextChunk objects with metadata
        text_chunks = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            # Find position in original text
            start_pos = text.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            
            chunk = TextChunk(
                content=chunk_text.strip(),
                chunk_index=idx,
                metadata={
                    **metadata,
                    'chunk_method': 'recursive',
                    'total_chunks': len(chunks)
                },
                start_char=start_pos,
                end_char=end_pos,
                token_count=self._estimate_tokens(chunk_text)
            )
            
            text_chunks.append(chunk)
            current_pos = start_pos + 1
        
        logger.info(f"Created {len(text_chunks)} chunks from {len(text)} characters")
        return text_chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using multiple separators
        This is the secret sauce for good chunking!
        
        Strategy:
        1. Try splitting by paragraphs (\n\n)
        2. If chunks too large, split by sentences (. or ! or ?)
        3. If still too large, split by characters
        """
        # List of separators in order of preference
        separators = [
            "\n\n",      # Paragraphs (best for preserving context)
            "\n",        # Lines
            ". ",        # Sentences
            "! ",        # Exclamations
            "? ",        # Questions
            "; ",        # Semi-colons
            ", ",        # Commas
            " ",         # Words
            ""           # Characters (last resort)
        ]
        
        return self._split_text_recursive(text, separators)
    
    def _split_text_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """
        Recursive helper for splitting text
        """
        # Base case: no separators left, return as is
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator == "":
            # Character-level split (last resort)
            splits = list(text)
        else:
            splits = text.split(separator)
        
        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_len = len(split)
            
            # If single split is too large, recurse with next separator
            if split_len > self.chunk_size:
                # First, add accumulated chunks
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Recurse on the large split
                if remaining_separators:
                    sub_chunks = self._split_text_recursive(split, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    # Force split by characters
                    chunks.extend(self._split_by_characters(split))
                continue
            
            # Add split to current chunk
            if current_size + split_len + len(separator) <= self.chunk_size:
                current_chunk.append(split)
                current_size += split_len + len(separator)
            else:
                # Current chunk is full, start new one
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_splits(chunks, separator)
                current_chunk.append(split)
                current_size = sum(len(s) for s in current_chunk) + len(separator) * len(current_chunk)
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by characters when all else fails"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _get_overlap_splits(self, chunks: List[str], separator: str) -> List[str]:
        """
        Get splits from previous chunk for overlap
        This maintains context between chunks!
        """
        if not chunks:
            return []
        
        last_chunk = chunks[-1]
        splits = last_chunk.split(separator)
        
        # Take splits from end that fit in overlap size
        overlap_splits = []
        overlap_size = 0
        
        for split in reversed(splits):
            if overlap_size + len(split) + len(separator) <= self.chunk_overlap:
                overlap_splits.insert(0, split)
                overlap_size += len(split) + len(separator)
            else:
                break
        
        return overlap_splits
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters for English)
        For more accuracy, use tiktoken library
        """
        # Simple heuristic: split by whitespace and punctuation
        words = re.findall(r'\w+', text)
        return len(words)
    
    def chunk_by_sentences(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Alternative chunking: by sentence count
        Useful for maintaining semantic boundaries
        
        Args:
            text: Text to chunk
            metadata: Optional metadata
            
        Returns:
            List of TextChunk objects
        """
        # Split into sentences using regex
        sentence_endings = r'[.!?]+[\s]+'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            if current_size + sentence_len <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_len
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_len
        
        # Add remaining
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        # Convert to TextChunk objects
        return [
            TextChunk(
                content=chunk,
                chunk_index=idx,
                metadata={**(metadata or {}), 'chunk_method': 'sentence'},
                start_char=0,
                end_char=len(chunk),
                token_count=self._estimate_tokens(chunk)
            )
            for idx, chunk in enumerate(chunks)
        ]
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {'total_chunks': 0}
        
        sizes = [len(chunk.content) for chunk in chunks]
        tokens = [chunk.token_count or 0 for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) // len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'avg_tokens': sum(tokens) // len(tokens) if tokens else 0,
            'total_tokens': sum(tokens)
        }


# Example usage and testing
if __name__ == "__main__":
    # Sample Ayurvedic text for testing
    sample_text = """
    Ayurveda is one of the world's oldest holistic healing systems. It was developed more than 3,000 years ago in India.
    
    It's based on the belief that health and wellness depend on a delicate balance between the mind, body, and spirit. Its main goal is to promote good health, not fight disease. But treatments may be geared toward specific health problems.
    
    According to Ayurveda, everything in the universe – dead or alive – is connected. If your mind, body, and spirit are in harmony with the universe, you have good health. When something disrupts this balance, you get sick.
    
    Among the things that can upset this balance are genetic or birth defects, injuries, climate and seasonal change, age, and your emotions. Those who practice Ayurveda believe every person is made of five basic elements found in the universe: space, air, fire, water, and earth.
    
    These combine in the human body to form three life forces or energies, called doshas. They control how your body works. The three doshas are Vata dosha (space and air), Pitta dosha (fire and water), and Kapha dosha (water and earth).
    """
    
    # Test with default settings
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text, metadata={'source': 'test'})
    
    print("\n=== Chunking Results ===")
    for chunk in chunks:
        print(f"\nChunk {chunk.chunk_index}:")
        print(f"Length: {len(chunk.content)} chars, ~{chunk.token_count} tokens")
        print(f"Content: {chunk.content[:100]}...")
    
    stats = chunker.get_chunk_stats(chunks)
    print(f"\n=== Stats ===")
    print(stats)