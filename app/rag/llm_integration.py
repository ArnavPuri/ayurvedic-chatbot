"""
LLM Integration Module - Phase 3
Integration with OpenAI and Anthropic for response generation
"""
from typing import List, Dict, Optional, Any
import logging
import os
from openai import OpenAI
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMGenerator:
	"""
	LLM integration for generating responses based on retrieved context
	
	Supports:
	- OpenAI (GPT-4, GPT-3.5)
	- Anthropic (Claude)
	- Context-aware response generation
	- Conversation history
	- Custom system prompts
	"""
	
	def __init__(
		self,
		provider: str = 'openai',
		model: Optional[str] = None,
		api_key: Optional[str] = None,
		temperature: float = 0.7,
		max_tokens: int = 1000
	):
		"""
		Initialize LLM provider
		
		Args:
			provider: 'openai' or 'anthropic'
			model: Model name (defaults based on provider)
			api_key: API key (falls back to env variables)
			temperature: Response creativity (0-1)
			max_tokens: Maximum response length
		"""
		self.provider = provider.lower()
		self.temperature = temperature
		self.max_tokens = max_tokens
		
		# Set default models
		if model is None:
			if self.provider == 'openai':
				model = 'gpt-3.5-turbo'
			elif self.provider == 'anthropic':
				model = 'claude-3-sonnet-20240229'
			else:
				raise ValueError(f"Unsupported provider: {provider}")
		
		self.model = model
		
		# Initialize client
		try:
			if self.provider == 'openai':
				api_key = api_key or os.getenv('OPENAI_API_KEY')
				if not api_key:
					raise ValueError(
						"OpenAI API key not found. "
						"Set OPENAI_API_KEY environment variable."
					)
				self.client = OpenAI(api_key=api_key)
				
			elif self.provider == 'anthropic':
				api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
				if not api_key:
					raise ValueError(
						"Anthropic API key not found. "
						"Set ANTHROPIC_API_KEY environment variable."
					)
				self.client = Anthropic(api_key=api_key)
			
			logger.info(
				f"LLM initialized: {self.provider} - {self.model}"
			)
			
		except Exception as e:
			logger.error(f"Failed to initialize LLM: {str(e)}")
			raise
	
	def generate_response(
		self,
		query: str,
		context_chunks: List[Dict[str, Any]],
		conversation_history: Optional[List[Dict[str, str]]] = None,
		system_prompt: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Generate a response based on query and retrieved context
		
		Args:
			query: User's question
			context_chunks: Retrieved context from vector store
			conversation_history: Previous messages
			system_prompt: Custom system prompt (uses default if None)
			
		Returns:
			Dictionary with answer and metadata
		"""
		try:
			# Build context from chunks
			context = self._build_context(context_chunks)
			
			# Build system prompt
			if system_prompt is None:
				system_prompt = self._get_default_system_prompt()
			
			# Build messages
			messages = self._build_messages(
				query=query,
				context=context,
				conversation_history=conversation_history,
				system_prompt=system_prompt
			)
			
			# Generate response based on provider
			if self.provider == 'openai':
				response = self._generate_openai(messages)
			elif self.provider == 'anthropic':
				response = self._generate_anthropic(messages, system_prompt)
			
			return {
				'answer': response,
				'model': self.model,
				'provider': self.provider,
				'context_chunks_used': len(context_chunks)
			}
			
		except Exception as e:
			logger.error(f"Error generating response: {str(e)}")
			raise
	
	def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
		"""Generate response using OpenAI"""
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				temperature=self.temperature,
				max_tokens=self.max_tokens
			)
			
			return response.choices[0].message.content
			
		except Exception as e:
			logger.error(f"OpenAI API error: {str(e)}")
			raise
	
	def _generate_anthropic(
		self,
		messages: List[Dict[str, str]],
		system_prompt: str
	) -> str:
		"""Generate response using Anthropic"""
		try:
			# Remove system message from messages (Claude uses separate param)
			user_messages = [
				msg for msg in messages
				if msg['role'] != 'system'
			]
			
			response = self.client.messages.create(
				model=self.model,
				system=system_prompt,
				messages=user_messages,
				temperature=self.temperature,
				max_tokens=self.max_tokens
			)
			
			return response.content[0].text
			
		except Exception as e:
			logger.error(f"Anthropic API error: {str(e)}")
			raise
	
	def _build_context(self, context_chunks: List[Dict[str, Any]]) -> str:
		"""
		Build formatted context string from retrieved chunks
		
		Args:
			context_chunks: List of chunk dictionaries
			
		Returns:
			Formatted context string
		"""
		if not context_chunks:
			return "No relevant context found."
		
		context_parts = []
		
		for i, chunk in enumerate(context_chunks, 1):
			content = chunk.get('content', '')
			metadata = chunk.get('metadata', {})
			
			filename = metadata.get('filename', 'Unknown')
			chunk_idx = metadata.get('chunk_index', '?')
			
			context_parts.append(
				f"[Source {i} - {filename}, Chunk {chunk_idx}]\n"
				f"{content}\n"
			)
		
		return "\n---\n".join(context_parts)
	
	def _build_messages(
		self,
		query: str,
		context: str,
		conversation_history: Optional[List[Dict[str, str]]],
		system_prompt: str
	) -> List[Dict[str, str]]:
		"""
		Build messages array for LLM
		
		Args:
			query: User query
			context: Retrieved context
			conversation_history: Previous messages
			system_prompt: System prompt
			
		Returns:
			List of message dictionaries
		"""
		messages = [
			{'role': 'system', 'content': system_prompt}
		]
		
		# Add conversation history if available
		if conversation_history:
			messages.extend(conversation_history)
		
		# Add current query with context
		user_message = (
			f"Context from Ayurvedic documents:\n\n"
			f"{context}\n\n"
			f"---\n\n"
			f"User Question: {query}\n\n"
			f"Please provide a detailed and accurate answer based on "
			f"the context above. If the context doesn't contain enough "
			f"information, acknowledge that and provide what you can."
		)
		
		messages.append({'role': 'user', 'content': user_message})
		
		return messages
	
	def _get_default_system_prompt(self) -> str:
		"""
		Get default system prompt for Ayurvedic chatbot
		
		Returns:
			System prompt string
		"""
		return (
			"You are an expert Ayurvedic knowledge assistant. "
			"Your role is to provide accurate, helpful information about "
			"Ayurveda based on the provided context from authentic sources.\n\n"
			
			"Guidelines:\n"
			"- Base your answers primarily on the provided context\n"
			"- Use clear, accessible language while maintaining accuracy\n"
			"- Preserve Sanskrit terms when relevant, with explanations\n"
			"- Be specific about doshas, treatments, and concepts\n"
			"- If the context is insufficient, say so honestly\n"
			"- Cite specific sources when possible\n"
			"- Maintain a helpful, educational tone\n"
			"- Do not provide medical advice - suggest consulting practitioners\n\n"
			
			"Remember: You are sharing traditional Ayurvedic knowledge, "
			"not providing medical diagnosis or treatment."
		)
	
	def get_info(self) -> Dict[str, Any]:
		"""
		Get information about the LLM configuration
		
		Returns:
			Dictionary with configuration info
		"""
		return {
			'provider': self.provider,
			'model': self.model,
			'temperature': self.temperature,
			'max_tokens': self.max_tokens
		}


# Test the module
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	
	print("\n=== Testing LLM Integration ===\n")
	
	# Check for API keys
	openai_key = os.getenv('OPENAI_API_KEY')
	anthropic_key = os.getenv('ANTHROPIC_API_KEY')
	
	if not openai_key and not anthropic_key:
		print("⚠️  No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
		print("Skipping live tests.\n")
		print("To test:")
		print("  export OPENAI_API_KEY='your-key'")
		print("  python app/rag/llm_integration.py")
		exit(0)
	
	# Use whichever key is available
	provider = 'openai' if openai_key else 'anthropic'
	
	# Initialize
	llm = LLMGenerator(provider=provider, api_key=openai_key, temperature=0.7)
	print(f"LLM Info: {llm.get_info()}\n")
	
	# Sample context
	context_chunks = [
		{
			'content': (
				"The three doshas in Ayurveda are Vata, Pitta, and Kapha. "
				"Vata is composed of air and space elements and governs movement. "
				"Pitta is composed of fire and water and governs transformation. "
				"Kapha is composed of earth and water and governs structure."
			),
			'metadata': {
				'filename': 'ayurveda_basics.pdf',
				'chunk_index': 5
			}
		}
	]
	
	# Test query
	query = "What are the three doshas?"
	
	print(f"Query: {query}\n")
	print("Generating response...\n")
	
	try:
		response = llm.generate_response(
			query=query,
			context_chunks=context_chunks
		)
		
		print(f"Answer:\n{response['answer']}\n")
		print(f"Model used: {response['model']}")
		print(f"Context chunks: {response['context_chunks_used']}")
		
		print("\n✅ LLM integration working perfectly!")
		
	except Exception as e:
		print(f"❌ Error: {str(e)}")
		print("Make sure your API key is valid and has credits.")

