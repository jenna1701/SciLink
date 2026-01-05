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
            title: Optional title for the content
            output_dimensionality: Optional output dimension (if model supports)
            **kwargs: Additional arguments (for forward compatibility)
            
        Returns:
            Dict with 'embedding' key containing the embedding vector(s)
            Format: {'embedding': [[float, ...], ...]} for multiple inputs
                    or {'embedding': [float, ...]} for single input (legacy behavior)
        """
        import time
        
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
        
        # Build config if we have extra parameters
        config = {}
        if output_dimensionality is not None:
            config['output_dimensionality'] = output_dimensionality
        if task_type is not None:
            config['task_type'] = task_type
        
        config = config if config else None
        
        # Use adaptive batching based on content size
        # API limit is ~40MB, we target ~20MB per batch for safety margin
        MAX_BATCH_BYTES = 20 * 1024 * 1024  # 20MB target
        MAX_BATCH_COUNT = 100  # Also limit by count as fallback
        
        # Create size-aware batches
        batches = self._create_size_aware_batches(contents_list, MAX_BATCH_BYTES, MAX_BATCH_COUNT)
        
        if len(batches) == 1:
            # Single batch - simple case
            return {'embedding': self._embed_batch_with_retry(effective_model, batches[0], config, is_single_input)}
        else:
            # Multiple batches needed
            logging.info(f"Batching {len(contents_list)} items into {len(batches)} size-aware batches")
            all_embeddings = []
            
            for batch_num, batch in enumerate(batches, 1):
                logging.debug(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} items)")
                batch_embeddings = self._embed_batch_with_retry(
                    effective_model, batch, config, is_single_input=False, 
                    batch_info=f"{batch_num}/{len(batches)}"
                )
                all_embeddings.extend(batch_embeddings)
            
            # Return combined results
            if is_single_input:
                return {'embedding': all_embeddings[0] if all_embeddings else []}
            else:
                return {'embedding': all_embeddings}
    
    def _embed_batch_with_retry(self, model: str, batch: List[str], config: Any, 
                                 is_single_input: bool = False, 
                                 batch_info: str = "",
                                 max_retries: int = 5,
                                 base_delay: float = 2.0) -> List:
        """
        Embed a batch with automatic retry for rate limit errors.
        
        Args:
            model: Model name
            batch: List of content strings
            config: Embedding config
            is_single_input: Whether original input was a single string
            batch_info: Batch identifier for logging
            max_retries: Maximum number of retries for rate limit errors
            base_delay: Base delay in seconds (doubles with each retry)
            
        Returns:
            List of embedding vectors
        """
        import time
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config
                )
                
                # Success - convert and return
                result = self._convert_response(response, is_single_input)
                embeddings = result.get('embedding', [])
                
                # Ensure we return a list for batch processing
                if is_single_input and isinstance(embeddings, list) and len(embeddings) > 0:
                    if not isinstance(embeddings[0], list):
                        # Single embedding returned as flat list
                        return [embeddings]
                
                return embeddings if isinstance(embeddings, list) else [embeddings]
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                is_rate_limit = '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str
                
                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Rate limited on batch {batch_info}. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    continue
                else:
                    # Not a rate limit error, or max retries exceeded
                    logging.error(f"Error generating embeddings: {e}")
                    raise
        
        # Should not reach here, but just in case
        raise last_error
    
    def _create_size_aware_batches(self, contents: List[str], 
                                    max_bytes: int, 
                                    max_count: int) -> List[List[str]]:
        """
        Create batches that respect both size and count limits.
        
        Args:
            contents: List of text strings to batch
            max_bytes: Maximum estimated bytes per batch
            max_count: Maximum items per batch
            
        Returns:
            List of batches, where each batch is a list of strings
        """
        batches = []
        current_batch = []
        current_size = 0
        
        for item in contents:
            # Estimate size (UTF-8 encoding + JSON overhead)
            item_size = len(item.encode('utf-8')) + 100  # 100 bytes overhead per item
            
            # Check if adding this item would exceed limits
            would_exceed_size = (current_size + item_size) > max_bytes
            would_exceed_count = len(current_batch) >= max_count
            
            if current_batch and (would_exceed_size or would_exceed_count):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            # Handle items that are too large even for a single-item batch
            if item_size > max_bytes:
                # Truncate the item to fit (with warning)
                max_chars = max_bytes - 200  # Leave room for overhead
                logging.warning(f"Truncating oversized content from {len(item)} to {max_chars} chars")
                item = item[:max_chars]
                item_size = len(item.encode('utf-8')) + 100
            
            current_batch.append(item)
            current_size += item_size
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
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