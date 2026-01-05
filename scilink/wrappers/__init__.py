"""
API Wrappers for Multi-Backend LLM Support

This module provides wrappers that normalize different LLM APIs to a common interface.
The pattern allows your code to work with multiple backends (Google Gemini, OpenAI, local models)
without modification.

Available Wrappers:

1. OpenAI Wrappers (make OpenAI look like legacy Google GenAI):
   - OpenAIAsGenerativeModel: For text/image generation
   - OpenAIAsEmbeddingModel: For embeddings

2. GenAI Wrappers (make new google-genai SDK look like legacy google.generativeai):
   - GenAIAsLegacyGenerativeModel: For text/image generation
   - GenAIAsLegacyEmbeddingModel: For embeddings
   - GenAIClient: Module-level interface mimicking legacy genai module

3. Local Model Wrappers:
   - LocalLlamaModel: For local llama.cpp models

Usage Example:
    from wrappers import GenAIAsLegacyGenerativeModel, OpenAIAsGenerativeModel
    
    # Use new Google GenAI SDK with legacy interface
    model = GenAIAsLegacyGenerativeModel('gemini-2.0-flash', api_key='...')
    
    # Or use OpenAI with the same interface
    model = OpenAIAsGenerativeModel('gpt-4', api_key='...', base_url='...')
    
    # Both work the same way:
    response = model.generate_content("Hello!")
    print(response.text)
"""

# OpenAI wrappers (make OpenAI look like legacy Google GenAI)
from .openai_wrapper import OpenAIAsGenerativeModel
from .openai_wrapper_embeddings import OpenAIAsEmbeddingModel

# New GenAI SDK wrappers (make new SDK look like legacy SDK)
try:
    from .genai_wrapper import (
        GenAIAsLegacyGenerativeModel,
        GenAIClient,
        LegacyGenerateContentResponse,
        LegacyChatSession,
        create_model as create_genai_model,
        GENAI_AVAILABLE,
    )
    from .genai_wrapper_embeddings import (
        GenAIAsLegacyEmbeddingModel,
        LegacyEmbeddingInterface,
        create_embedding_model as create_genai_embedding_model,
    )
except ImportError:
    # google-genai not installed
    GenAIAsLegacyGenerativeModel = None
    GenAIClient = None
    LegacyGenerateContentResponse = None
    LegacyChatSession = None
    GenAIAsLegacyEmbeddingModel = None
    LegacyEmbeddingInterface = None
    create_genai_model = None
    create_genai_embedding_model = None
    GENAI_AVAILABLE = False

# Local model wrapper
try:
    from .llama_wrapper import LocalLlamaModel
except ImportError:
    LocalLlamaModel = None


__all__ = [
    # OpenAI wrappers
    'OpenAIAsGenerativeModel',
    'OpenAIAsEmbeddingModel',
    
    # GenAI wrappers (new SDK -> legacy interface)
    'GenAIAsLegacyGenerativeModel',
    'GenAIClient',
    'LegacyGenerateContentResponse',
    'LegacyChatSession',
    'GenAIAsLegacyEmbeddingModel',
    'LegacyEmbeddingInterface',
    'create_genai_model',
    'create_genai_embedding_model',
    'GENAI_AVAILABLE',
    
    # Local model wrapper
    'LocalLlamaModel',
]