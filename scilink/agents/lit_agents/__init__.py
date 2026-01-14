"""
Literature Agents Module

Provides agents for:
- Literature search via FutureHouse/Edison API (OWL, CROW)
- Novelty scoring of scientific claims
- Query optimization for scientific databases

Updated to use LiteLLM for multi-provider LLM support.
"""

from .literature_agent import (
    OwlLiteratureAgent, 
    IncarLiteratureAgent,
    FittingModelLiteratureAgent,
    LiteratureSearchAgent
)
from .novelty_scorer import (
    NoveltyScorer, 
    enhanced_novelty_assessment, 
    display_enhanced_novelty_summary,
    save_enhanced_novelty_results,
    get_novelty_priorities_for_dft,
    get_novel_claims_legacy,
    get_known_claims_legacy
)
from .optimize_query import optimize_search_query

__all__ = [
    # Literature agents
    'OwlLiteratureAgent',
    'IncarLiteratureAgent', 
    'FittingModelLiteratureAgent',
    'LiteratureSearchAgent',
    # Novelty scoring
    'NoveltyScorer',
    'enhanced_novelty_assessment',
    'display_enhanced_novelty_summary',
    'save_enhanced_novelty_results',
    'get_novelty_priorities_for_dft',
    'get_novel_claims_legacy',
    'get_known_claims_legacy',
    # Query optimization
    'optimize_search_query',
]