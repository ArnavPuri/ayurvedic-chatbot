"""
Embeddings Module - Phase 3
Generate vector embeddings for semantic search using sentence-transformers
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
	"""
	Generates vector embeddings for text chunks using sentence-transformers
	
	Features:
	- Uses all-MiniLM-L6-v2 model (fast and efficient)
	- Batch processing for performance
	- Normalized embeddings for cosine similarity
	- Supports both single and batch text encoding
	"""
	
	def __init__(
		self,
		model_name: str = 'all-MiniLM-L6-v2',
		device: str = None
	):
		"""
		Initialize the embedding generator
		
		Args:
			model_name: HuggingFace model name for embeddings
				- all-MiniLM-L6-v2: Fast, 384 dimensions (default)
				- all-mpnet-base-v2: Higher quality, 768 dimensions
				- multi-qa-mpnet-base-dot-v1: Optimized for Q&A
			device: 'cuda', 'cpu', or None (auto-detect)
		"""
		logger.info(f"Initializing embedding model: {model_name}")
		
		try:
			self.model = SentenceTransformer(model_name, device=device)
			self.model_name = model_name
			self.dimension = self.model.get_sentence_embedding_dimension()
			
			logger.info(
				f"Embedding model loaded successfully. "
				f"Dimension: {self.dimension}"
			)
			
		except Exception as e:
			logger.error(f"Failed to load embedding model: {str(e)}")
			raise
	
	def encode(
		self,
		texts: Union[str, List[str]],
		batch_size: int = 32,
		show_progress: bool = False,
		normalize: bool = True
	) -> np.ndarray:
		"""
		Encode text(s) into embeddings
		
		Args:
			texts: Single text or list of texts to encode
			batch_size: Number of texts to process at once
			show_progress: Show progress bar for large batches
			normalize: Normalize embeddings for cosine similarity
			
		Returns:
			numpy array of embeddings (n_texts, embedding_dim)
		"""
		try:
			# Handle single text input
			if isinstance(texts, str):
				texts = [texts]
			
			if len(texts) == 0:
				logger.warning("Empty text list provided for encoding")
				return np.array([])
			
			# Generate embeddings
			embeddings = self.model.encode(
				texts,
				batch_size=batch_size,
				show_progress_bar=show_progress,
				normalize_embeddings=normalize,
				convert_to_numpy=True
			)
			
			logger.debug(
				f"Encoded {len(texts)} texts into embeddings "
				f"of shape {embeddings.shape}"
			)
			
			return embeddings
			
		except Exception as e:
			logger.error(f"Error encoding texts: {str(e)}")
			raise
	
	def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
		"""
		Encode a single query for search
		
		Args:
			query: Query text
			normalize: Normalize embedding for cosine similarity
			
		Returns:
			numpy array embedding (embedding_dim,)
		"""
		embedding = self.encode(
			query,
			batch_size=1,
			normalize=normalize
		)
		
		return embedding[0]  # Return single vector
	
	def encode_documents(
		self,
		documents: List[str],
		batch_size: int = 32,
		show_progress: bool = True
	) -> np.ndarray:
		"""
		Encode multiple documents for indexing
		
		Args:
			documents: List of document texts
			batch_size: Batch size for processing
			show_progress: Show progress bar
			
		Returns:
			numpy array of embeddings (n_docs, embedding_dim)
		"""
		logger.info(f"Encoding {len(documents)} documents...")
		
		embeddings = self.encode(
			documents,
			batch_size=batch_size,
			show_progress=show_progress,
			normalize=True
		)
		
		logger.info(f"Successfully encoded {len(documents)} documents")
		
		return embeddings
	
	def compute_similarity(
		self,
		query_embedding: np.ndarray,
		doc_embeddings: np.ndarray
	) -> np.ndarray:
		"""
		Compute cosine similarity between query and documents
		
		Args:
			query_embedding: Query embedding vector (embedding_dim,)
			doc_embeddings: Document embeddings (n_docs, embedding_dim)
			
		Returns:
			Similarity scores (n_docs,)
		"""
		# Ensure query is 2D for matrix multiplication
		if query_embedding.ndim == 1:
			query_embedding = query_embedding.reshape(1, -1)
		
		# Compute cosine similarity (embeddings are already normalized)
		similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
		
		return similarities
	
	def get_info(self) -> dict:
		"""
		Get information about the embedding model
		
		Returns:
			Dictionary with model info
		"""
		return {
			'model_name': self.model_name,
			'embedding_dimension': self.dimension,
			'max_sequence_length': self.model.max_seq_length,
			'device': str(self.model.device)
		}


# Test the module
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	
	print("\n=== Testing Embedding Generator ===\n")
	
	# Initialize
	generator = EmbeddingGenerator()
	print(f"Model Info: {generator.get_info()}\n")
	
	# Test single encoding
	text = "Ayurveda is an ancient system of medicine from India"
	embedding = generator.encode_query(text)
	print(f"Single text embedding shape: {embedding.shape}")
	print(f"First 5 values: {embedding[:5]}\n")
	
	# Test batch encoding
	texts = [
		"The three doshas are Vata, Pitta, and Kapha",
		"Panchakarma is a detoxification treatment",
		"Turmeric has anti-inflammatory properties",
		"Ayurvedic diet is based on individual constitution"
	]
	
	embeddings = generator.encode_documents(texts, show_progress=False)
	print(f"Batch embeddings shape: {embeddings.shape}\n")
	
	# Test similarity
	query = "What are the doshas in Ayurveda?"
	query_emb = generator.encode_query(query)
	
	similarities = generator.compute_similarity(query_emb, embeddings)
	print(f"Query: {query}")
	print(f"Similarities: {similarities}")
	print(f"Most similar: {texts[np.argmax(similarities)]}")
	
	print("\n✅ Embedding generator working perfectly!")

