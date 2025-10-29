"""
Text Preprocessing Module
Cleans and normalizes text, especially Ayurvedic content
"""
from typing import Dict, List, Optional, Tuple
import re
import unicodedata
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedText:
    """Container for preprocessed text with metadata"""
    original: str
    cleaned: str
    metadata: Dict
    keywords: List[str]
    sections: List[Dict]


class TextPreprocessor:
    """
    Preprocesses text for optimal RAG performance
    Handles Ayurvedic-specific content (Sanskrit terms, medical terminology)
    """
    
    def __init__(self, preserve_sanskrit: bool = True):
        """
        Initialize preprocessor
        
        Args:
            preserve_sanskrit: Whether to preserve Sanskrit diacritics
        """
        self.preserve_sanskrit = preserve_sanskrit
        
        # Common Ayurvedic terms to preserve (case-insensitive)
        self.ayurvedic_terms = {
            'dosha', 'doshas', 'vata', 'pitta', 'kapha',
            'agni', 'ama', 'ojas', 'tejas', 'prana',
            'dhatu', 'dhatus', 'srotas', 'malas',
            'vyadhi', 'prakriti', 'vikriti',
            'rasayana', 'panchakarma', 'abhyanga',
            'shirodhara', 'nasya', 'basti',
            'ayurveda', 'ayurvedic', 'vaidya'
        }
        
        # Medical terms patterns
        self.medical_patterns = [
            r'\b\w+itis\b',      # inflammation: arthritis, gastritis
            r'\b\w+osis\b',      # condition: osteoporosis, thrombosis
            r'\b\w+emia\b',      # blood condition: anemia, hypoglycemia
        ]
        
        logger.info("TextPreprocessor initialized")
    
    def preprocess(
        self,
        text: str,
        extract_metadata: bool = True
    ) -> PreprocessedText:
        """
        Main preprocessing pipeline
        
        Steps:
        1. Normalize Unicode
        2. Clean whitespace
        3. Remove unwanted characters
        4. Extract metadata (titles, sections)
        5. Extract keywords
        
        Args:
            text: Raw text to preprocess
            extract_metadata: Whether to extract sections and keywords
            
        Returns:
            PreprocessedText object
        """
        original = text
        
        # Step 1: Unicode normalization
        cleaned = self._normalize_unicode(text)
        
        # Step 2: Clean whitespace
        cleaned = self._clean_whitespace(cleaned)
        
        # Step 3: Remove unwanted characters (but preserve important ones)
        cleaned = self._remove_unwanted_chars(cleaned)
        
        # Step 4: Fix common OCR errors (from scanned PDFs)
        cleaned = self._fix_ocr_errors(cleaned)
        
        # Extract metadata if requested
        metadata = {}
        keywords = []
        sections = []
        
        if extract_metadata:
            sections = self._extract_sections(cleaned)
            keywords = self._extract_keywords(cleaned)
            metadata = self._extract_metadata(cleaned)
        
        result = PreprocessedText(
            original=original,
            cleaned=cleaned,
            metadata=metadata,
            keywords=keywords,
            sections=sections
        )
        
        logger.info(f"Preprocessed {len(original)} -> {len(cleaned)} chars")
        return result
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters
        Handles special characters, accents, etc.
        """
        if self.preserve_sanskrit:
            # Use NFKC normalization (preserves diacritics)
            # NFKC = Normalization Form KC (Compatibility Composition)
            return unicodedata.normalize('NFKC', text)
        else:
            # Use NFKD + remove diacritics
            normalized = unicodedata.normalize('NFKD', text)
            # Remove combining characters (accents, diacritics)
            return ''.join(c for c in normalized if not unicodedata.combining(c))
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Clean and normalize whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing/leading whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Final trim
        return text.strip()
    
    def _remove_unwanted_chars(self, text: str) -> str:
        """
        Remove unwanted characters but preserve structure
        """
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Remove weird unicode characters that slip through
        # But preserve Sanskrit diacritics if enabled
        if not self.preserve_sanskrit:
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove excessive punctuation (e.g., "......" -> "...")
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'-{3,}', '--', text)
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors from scanned documents
        """
        # Common OCR mistakes
        replacements = {
            r'\bl\b': 'I',           # lowercase L mistaken for I
            r'\bO\b': '0',           # O mistaken for zero (in numbers)
            r'rn': 'm',              # rn mistaken for m
            r'\|': 'I',              # pipe mistaken for I
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """
        Extract document sections (chapters, headings)
        Useful for maintaining document structure
        """
        sections = []
        
        # Pattern for common heading formats:
        # "Chapter 1: Title", "1. Title", "TITLE IN CAPS"
        heading_patterns = [
            r'^#{1,3}\s+(.+)$',                    # Markdown headers
            r'^([A-Z][A-Z\s]+)$',                  # ALL CAPS lines
            r'^(Chapter|Section)\s+\d+:?\s*(.+)$', # Chapter X: Title
            r'^\d+\.\s+([A-Z].+)$',                # 1. Title
        ]
        
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in heading_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    # Extract title (last group in match)
                    title = match.groups()[-1] if match.groups() else line
                    
                    sections.append({
                        'title': title.strip(),
                        'line_number': i,
                        'char_position': current_pos,
                        'level': self._determine_heading_level(line)
                    })
                    break
            
            current_pos += len(line) + 1
        
        return sections
    
    def _determine_heading_level(self, line: str) -> int:
        """Determine heading level (1=highest, 3=lowest)"""
        if line.startswith('###'):
            return 3
        elif line.startswith('##'):
            return 2
        elif line.startswith('#'):
            return 1
        elif line.isupper() and len(line) > 3:
            return 1  # ALL CAPS likely main heading
        elif re.match(r'^Chapter', line):
            return 1
        else:
            return 2
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords (Ayurvedic terms, medical terms)
        These can be used for metadata and search boosting
        """
        keywords = set()
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Extract Ayurvedic terms
        for term in self.ayurvedic_terms:
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{term}\b', text_lower):
                keywords.add(term)
        
        # Extract medical terminology
        for pattern in self.medical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.update(matches)
        
        # Extract capitalized terms (likely proper nouns or important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Filter to multi-word terms or longer terms
        important_caps = [term for term in capitalized if len(term) > 5 or ' ' in term]
        keywords.update(term.lower() for term in important_caps[:20])  # Limit to top 20
        
        return sorted(list(keywords))
    
    def _extract_metadata(self, text: str) -> Dict:
        """
        Extract general metadata about the text
        """
        lines = text.split('\n')
        words = text.split()
        
        # Try to find title (usually first non-empty line or largest heading)
        title = None
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (line.isupper() or len(line) > 10):
                title = line
                break
        
        # Count sentences (rough estimate)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Detect language hints (very basic)
        has_sanskrit = any(term in text.lower() for term in list(self.ayurvedic_terms)[:5])
        
        return {
            'title': title,
            'word_count': len(words),
            'char_count': len(text),
            'line_count': len(lines),
            'sentence_count': sentence_count,
            'has_ayurvedic_content': has_sanskrit,
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize user queries for better matching
        Simpler than document preprocessing
        
        Args:
            query: User's search query
            
        Returns:
            Normalized query string
        """
        # Lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove punctuation at start/end
        query = query.strip('.,!?;:')
        
        return query
    
    def get_preprocessing_stats(self, processed: PreprocessedText) -> Dict:
        """Get statistics about preprocessing results"""
        return {
            'original_length': len(processed.original),
            'cleaned_length': len(processed.cleaned),
            'reduction_percent': round(
                (1 - len(processed.cleaned) / len(processed.original)) * 100, 2
            ) if len(processed.original) > 0 else 0,
            'keywords_found': len(processed.keywords),
            'sections_found': len(processed.sections),
            'has_ayurvedic_content': processed.metadata.get('has_ayurvedic_content', False)
        }


# Example usage and testing
if __name__ == "__main__":
    # Sample text with various issues
    sample_text = """
    AYURVEDA: THE SCIENCE OF LIFE
    
    Chapter 1: Introduction to Ayurveda
    
    Ayurveda    is an ancient    system of medicine that originated in India over 3,000 years ago.
    
    The  word  "Ayurveda"  comes from the Sanskrit words "ayur" (life) and "veda" (knowledge).
    
    The Three Doshas
    
    According to Ayurveda, there are three primary doshas:
    1. Vata dosha - governs movement and communication
    2. Pitta dosha - controls digestion and metabolism  
    3. Kapha dosha - provides structure and lubrication
    
    Common conditions treated include arthritis, gastritis, and various other inflammatory conditions.
    
    Panchakarma is a powerful detoxification therapy used in Ayurveda.
    """
    
    # Test preprocessing
    preprocessor = TextPreprocessor(preserve_sanskrit=True)
    result = preprocessor.preprocess(sample_text)
    
    print("\n=== Preprocessing Results ===")
    print(f"\nCleaned text preview:")
    print(result.cleaned[:200] + "...")
    
    print(f"\n\nMetadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n\nKeywords found: {result.keywords}")
    
    print(f"\n\nSections found:")
    for section in result.sections:
        print(f"  Level {section['level']}: {section['title']}")
    
    print(f"\n\nStats:")
    stats = preprocessor.get_preprocessing_stats(result)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query normalization
    query = "  What are the THREE doshas??  "
    normalized = preprocessor.normalize_query(query)
    print(f"\n\nQuery normalization:")
    print(f"  Original: '{query}'")
    print(f"  Normalized: '{normalized}'")