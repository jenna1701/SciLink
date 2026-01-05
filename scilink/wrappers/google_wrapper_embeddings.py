"""
GenAI SDK Embeddings Wrapper for Backward Compatibility

This module wraps the new `google-genai` SDK's embedding functionality to provide
the same interface as the legacy `google.generativeai.embed_content` function.

Usage:
    # Instead of:
    # import google.generativeai as genai
    # genai.configure(api_key=...)
    # response = genai.embed_content(model='...', content='...')
    
    # Use:
    from wrappers.genai_wrapper_embeddings import GenAIAsLegacyEmbeddingModel
    embedder = GenAIAsLegacyEmbeddingModel(model='gemini-embedding-001', api_key=...)
    response = embedder.embed_content(model='...', content='...')
    
    # Response format is identical: {'embedding': [[...]]}
"""

import logging
from typing import Any, Dict, List, Union, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


class GenAIAsLegacyEmbeddingModel:
    """
    Wraps the new google-genai SDK to provide the same embedding interface
    as the legacy google.generativeai.embed_content function.
    
    The legacy SDK interface was:
        response = genai.embed_content(
            model='models/gemini-embedding-001',
            content='Hello world',
            task_type='RETRIEVAL_DOCUMENT'
        )
        # response['embedding'] contains the embedding vector(s)
    
    This wrapper provides the same interface:
        embedder = GenAIAsLegacyEmbeddingModel(api_key=...)
        response = embedder.embed_content(
            model='gemini-embedding-001',
            content='Hello world',
            task_type='RETRIEVAL_DOCUMENT'
        )
        # response['embedding'] contains the embedding vector(s)
    """
    
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize the embedding wrapper.
        
        Args:
            model: Default model name (optional, can be overridden in embed_content)
            api_key: API key (or set GEMINI_API_KEY env var)
            base_url: Not used for Google GenAI, kept for interface compatibility
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai SDK not installed. "
                "Install with: pip install google-genai"
            )
        
        self._default_model = model
        self._api_key = api_key
        self._base_url = base_url  # Not used for GenAI but kept for compatibility
        
        # Create the client
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()
    
    def embed_content(self, 
                      model: str = None,
                      content: Union[str, List[str]] = None,
                      task_type: str = None,
                      title: str = None,
                      output_dimensionality: int = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for content.
        
        This method signature matches the legacy genai.embed_content function.
        
        Args:
            model: Model name (e.g., 'gemini-embedding-001' or 'models/gemini-embedding-001')
            content: Text or list of texts to embed
            task_type: Task type hint (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
                       Note: The new SDK may not use this directly
            title: Optional title for the content
            output_dimensionality: Optional output dimension (if model supports)
            **kwargs: Additional arguments (for forward compatibility)
            
        Returns:
            Dict with 'embedding' key containing the embedding vector(s)
            Format: {'embedding': [[float, ...], ...]} for multiple inputs
                    or {'embedding': [float, ...]} for single input (legacy behavior)
        """
        # Use default model if not specified
        effective_model = model or self._default_model
        if not effective_model:
            raise ValueError("Model must be specified either in constructor or embed_content call")
        
        # Normalize model name (remove 'models/' prefix if present)
        if effective_model.startswith('models/'):
            effective_model = effective_model[7:]
        
        # Handle content
        if content is None:
            raise ValueError("Content must be provided")
        
        # Normalize content to list for batch processing
        is_single_input = isinstance(content, str)
        contents_list = [content] if is_single_input else content
        
        try:
            # Build config if we have extra parameters
            config = None
            if output_dimensionality is not None:
                config = {'output_dimensionality': output_dimensionality}
            
            # Call the new SDK
            response = self._client.models.embed_content(
                model=effective_model,
                contents=contents_list,
                config=config
            )
            
            # Convert response to legacy format
            return self._convert_response(response, is_single_input)
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise
    
    def _convert_response(self, response: Any, is_single_input: bool) -> Dict[str, Any]:
        """
        Convert new SDK response to legacy format.
        
        The legacy format returns:
        - Single input: {'embedding': [float, float, ...]}
        - Multiple inputs: {'embedding': [[float, ...], [float, ...], ...]}
        
        The new SDK returns an object with 'embeddings' attribute containing
        a list of embedding objects.
        """
        embeddings = []
        
        # Extract embeddings from response
        raw_embeddings = getattr(response, 'embeddings', None)
        
        if raw_embeddings is None:
            # Try alternative attribute names
            raw_embeddings = getattr(response, 'embedding', None)
            if raw_embeddings is not None:
                raw_embeddings = [raw_embeddings]
        
        if raw_embeddings is None:
            logging.warning("No embeddings found in response")
            return {'embedding': [] if not is_single_input else []}
        
        # Process each embedding
        for emb in raw_embeddings:
            # The embedding might be an object with 'values' attribute
            # or a direct list of floats
            if hasattr(emb, 'values'):
                embeddings.append(list(emb.values))
            elif isinstance(emb, (list, tuple)):
                embeddings.append(list(emb))
            else:
                # Try to convert to list
                try:
                    embeddings.append(list(emb))
                except TypeError:
                    logging.warning(f"Could not convert embedding: {type(emb)}")
                    embeddings.append([])
        
        # Return in legacy format
        if is_single_input and len(embeddings) == 1:
            # Single input returns flat list (legacy behavior)
            return {'embedding': embeddings[0]}
        else:
            # Multiple inputs return list of lists
            return {'embedding': embeddings}


class LegacyEmbeddingInterface:
    """
    A module-level interface that mimics the legacy genai.embed_content function.
    
    Usage:
        embedder = LegacyEmbeddingInterface(api_key=...)
        
        # Then use like the old genai module:
        response = embedder.embed_content(
            model='gemini-embedding-001',
            content=['Hello', 'World'],
            task_type='RETRIEVAL_DOCUMENT'
        )
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the interface.
        
        Args:
            api_key: API key (or set GEMINI_API_KEY env var)
        """
        self._embedder = GenAIAsLegacyEmbeddingModel(api_key=api_key)
    
    def embed_content(self, 
                      model: str,
                      content: Union[str, List[str]],
                      task_type: str = None,
                      title: str = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings (same signature as legacy genai.embed_content).
        """
        return self._embedder.embed_content(
            model=model,
            content=content,
            task_type=task_type,
            title=title,
            **kwargs
        )


# Convenience function for direct use
def create_embedding_model(api_key: str = None, model: str = None) -> GenAIAsLegacyEmbeddingModel:
    """
    Create a legacy-compatible embedding model.
    
    Args:
        api_key: API key
        model: Default model name
        
    Returns:
        GenAIAsLegacyEmbeddingModel instance
    """
    return GenAIAsLegacyEmbeddingModel(model=model, api_key=api_key)