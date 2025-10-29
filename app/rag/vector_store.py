"""
Vector Store Module - Phase 3
ChromaDB integration for persistent vector storage and semantic search
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
	"""
	ChromaDB-based vector store for semantic search
	
	Features:
	- Persistent storage of embeddings
	- Semantic similarity search
	- Metadata filtering
	- Document management (add, delete, update)
	- Automatic ID generation
	"""
	
	def __init__(
		self,
		collection_name: str = 'ayurvedic_documents',
		persist_directory: str = './data/chroma_db',
		embedding_dimension: int = 384
	):
		"""
		Initialize ChromaDB vector store
		
		Args:
			collection_name: Name of the collection to use
			persist_directory: Directory for persistent storage
			embedding_dimension: Dimension of embeddings (must match model)
		"""
		self.collection_name = collection_name
		self.persist_directory = persist_directory
		self.embedding_dimension = embedding_dimension
		
		# Create persist directory if it doesn't exist
		Path(persist_directory).mkdir(parents=True, exist_ok=True)
		
		logger.info(f"Initializing ChromaDB at {persist_directory}")
		
		try:
			# Initialize ChromaDB client with persistent storage
			self.client = chromadb.PersistentClient(
				path=persist_directory,
				settings=Settings(
					anonymized_telemetry=False,
					allow_reset=True
				)
			)
			
			# Get or create collection
			self.collection = self.client.get_or_create_collection(
				name=collection_name,
				metadata={
					'description': 'Ayurvedic documents and chunks',
					'embedding_dimension': embedding_dimension
				}
			)
			
			logger.info(
				f"ChromaDB initialized. Collection: {collection_name}, "
				f"Documents: {self.collection.count()}"
			)
			
		except Exception as e:
			logger.error(f"Failed to initialize ChromaDB: {str(e)}")
			raise
	
	def add_documents(
		self,
		embeddings: np.ndarray,
		documents: List[str],
		metadatas: List[Dict[str, Any]],
		ids: Optional[List[str]] = None
	) -> List[str]:
		"""
		Add documents with embeddings to the vector store
		
		Args:
			embeddings: Document embeddings (n_docs, embedding_dim)
			documents: List of document texts
			metadatas: List of metadata dicts for each document
			ids: Optional list of document IDs (auto-generated if None)
			
		Returns:
			List of document IDs
		"""
		try:
			n_docs = len(documents)
			
			if n_docs == 0:
				logger.warning("No documents to add")
				return []
			
			# Validate inputs
			if len(embeddings) != n_docs:
				raise ValueError(
					f"Embeddings count ({len(embeddings)}) != "
					f"documents count ({n_docs})"
				)
			
			if len(metadatas) != n_docs:
				raise ValueError(
					f"Metadata count ({len(metadatas)}) != "
					f"documents count ({n_docs})"
				)
			
			# Generate IDs if not provided
			if ids is None:
				current_count = self.collection.count()
				ids = [
					f"doc_{current_count + i}"
					for i in range(n_docs)
				]
			
			# Convert embeddings to list for ChromaDB
			embeddings_list = embeddings.tolist()
			
			# Add to collection
			self.collection.add(
				embeddings=embeddings_list,
				documents=documents,
				metadatas=metadatas,
				ids=ids
			)
			
			logger.info(f"Added {n_docs} documents to vector store")
			
			return ids
			
		except Exception as e:
			logger.error(f"Error adding documents: {str(e)}")
			raise
	
	def query(
		self,
		query_embedding: np.ndarray,
		n_results: int = 5,
		where: Optional[Dict[str, Any]] = None,
		where_document: Optional[Dict[str, str]] = None
	) -> Dict[str, Any]:
		"""
		Query the vector store for similar documents
		
		Args:
			query_embedding: Query embedding vector
			n_results: Number of results to return
			where: Metadata filter (e.g., {'doc_id': 'doc123'})
			where_document: Document content filter
			
		Returns:
			Dictionary with ids, documents, metadatas, distances
		"""
		try:
			# Ensure query is 1D
			if query_embedding.ndim > 1:
				query_embedding = query_embedding.flatten()
			
			# Query the collection
			results = self.collection.query(
				query_embeddings=[query_embedding.tolist()],
				n_results=n_results,
				where=where,
				where_document=where_document
			)
			
			# Flatten results (ChromaDB returns nested lists)
			flattened = {
				'ids': results['ids'][0] if results['ids'] else [],
				'documents': results['documents'][0] if results['documents'] else [],
				'metadatas': results['metadatas'][0] if results['metadatas'] else [],
				'distances': results['distances'][0] if results['distances'] else []
			}
			
			logger.debug(f"Query returned {len(flattened['ids'])} results")
			
			return flattened
			
		except Exception as e:
			logger.error(f"Error querying vector store: {str(e)}")
			raise
	
	def search(
		self,
		query_embedding: np.ndarray,
		top_k: int = 5,
		filters: Optional[Dict[str, Any]] = None
	) -> List[Dict[str, Any]]:
		"""
		Search for similar documents and return formatted results
		
		Args:
			query_embedding: Query embedding vector
			top_k: Number of top results to return
			filters: Optional metadata filters
			
		Returns:
			List of result dictionaries with content and metadata
		"""
		results = self.query(
			query_embedding=query_embedding,
			n_results=top_k,
			where=filters
		)
		
		# Format results
		formatted_results = []
		for i in range(len(results['ids'])):
			# Convert distance to similarity score (0-1)
			# ChromaDB uses L2 distance, smaller is better
			distance = results['distances'][i]
			similarity = 1 / (1 + distance)  # Convert to similarity
			
			formatted_results.append({
				'id': results['ids'][i],
				'content': results['documents'][i],
				'metadata': results['metadatas'][i],
				'distance': distance,
				'similarity': similarity
			})
		
		return formatted_results
	
	def delete_by_ids(self, ids: List[str]) -> bool:
		"""
		Delete documents by IDs
		
		Args:
			ids: List of document IDs to delete
			
		Returns:
			True if successful
		"""
		try:
			self.collection.delete(ids=ids)
			logger.info(f"Deleted {len(ids)} documents from vector store")
			return True
			
		except Exception as e:
			logger.error(f"Error deleting documents: {str(e)}")
			return False
	
	def delete_by_metadata(self, where: Dict[str, Any]) -> bool:
		"""
		Delete documents by metadata filter
		
		Args:
			where: Metadata filter (e.g., {'doc_id': 'doc123'})
			
		Returns:
			True if successful
		"""
		try:
			self.collection.delete(where=where)
			logger.info(f"Deleted documents matching filter: {where}")
			return True
			
		except Exception as e:
			logger.error(f"Error deleting documents: {str(e)}")
			return False
	
	def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
		"""
		Get documents by IDs
		
		Args:
			ids: List of document IDs
			
		Returns:
			Dictionary with documents and metadata
		"""
		try:
			results = self.collection.get(ids=ids)
			return results
			
		except Exception as e:
			logger.error(f"Error getting documents: {str(e)}")
			raise
	
	def count(self) -> int:
		"""
		Get total number of documents in the store
		
		Returns:
			Document count
		"""
		return self.collection.count()
	
	def reset(self) -> bool:
		"""
		Delete all documents from the collection
		
		Returns:
			True if successful
		"""
		try:
			self.client.delete_collection(self.collection_name)
			self.collection = self.client.create_collection(
				name=self.collection_name,
				metadata={
					'description': 'Ayurvedic documents and chunks',
					'embedding_dimension': self.embedding_dimension
				}
			)
			logger.info("Vector store reset successfully")
			return True
			
		except Exception as e:
			logger.error(f"Error resetting vector store: {str(e)}")
			return False
	
	def get_stats(self) -> Dict[str, Any]:
		"""
		Get statistics about the vector store
		
		Returns:
			Dictionary with stats
		"""
		return {
			'collection_name': self.collection_name,
			'total_documents': self.count(),
			'persist_directory': self.persist_directory,
			'embedding_dimension': self.embedding_dimension
		}


# Test the module
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	
	print("\n=== Testing Vector Store ===\n")
	
	# Initialize
	store = VectorStore(
		collection_name='test_collection',
		persist_directory='./data/test_chroma'
	)
	
	print(f"Initial stats: {store.get_stats()}\n")
	
	# Create sample embeddings
	np.random.seed(42)
	embeddings = np.random.rand(3, 384).astype(np.float32)
	
	documents = [
		"Vata dosha is associated with air and space",
		"Pitta dosha is associated with fire and water",
		"Kapha dosha is associated with earth and water"
	]
	
	metadatas = [
		{'doc_id': 'test1', 'type': 'dosha', 'dosha': 'vata'},
		{'doc_id': 'test2', 'type': 'dosha', 'dosha': 'pitta'},
		{'doc_id': 'test3', 'type': 'dosha', 'dosha': 'kapha'}
	]
	
	# Add documents
	ids = store.add_documents(
		embeddings=embeddings,
		documents=documents,
		metadatas=metadatas
	)
	print(f"Added documents with IDs: {ids}\n")
	
	# Query
	query_emb = np.random.rand(384).astype(np.float32)
	results = store.search(query_emb, top_k=2)
	
	print(f"Search results:")
	for i, result in enumerate(results, 1):
		print(f"\n{i}. Similarity: {result['similarity']:.4f}")
		print(f"   Content: {result['content'][:50]}...")
		print(f"   Metadata: {result['metadata']}")
	
	# Stats
	print(f"\nFinal stats: {store.get_stats()}")
	
	# Cleanup
	store.reset()
	print("\n✅ Vector store working perfectly!")

