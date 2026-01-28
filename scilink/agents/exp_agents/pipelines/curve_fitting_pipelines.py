# pipelines/curve_fitting_pipelines.py

"""
Unified Curve Fitting Pipeline Factory

Factory functions for creating curve fitting pipelines.
All analysis now uses a single unified pipeline that handles both
single spectra (n=1) and series (n>1) identically.

Key principle: Single spectrum = Series of 1
"""

import logging
from typing import Callable, List, Any

from ..controllers.curve_fitting_controllers import (
    # Original controllers
    AnalyzeDataController,
    LiteratureSearchController,
    GenerateCurveFittingReportController,
    # Unified controllers for series support
    HumanFeedbackRefinementController,
    UnifiedSeriesProcessingController,
    ConditionalTrendAnalysisController,
    UnifiedCurveSynthesisController,
    UnifiedCurveReportController,
)
from ..controllers.base_controllers import (
    StoreAnalysisResultsController,
)
from ..instruct import (
    CURVE_ANALYSIS_INSTRUCTIONS,
    FITTING_SCRIPT_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
    FITTING_INTERPRETATION_INSTRUCTIONS,
)


def create_unified_curve_fitting_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    plot_fn: Callable,
    executor: Any,
    output_dir: str,
    preprocessor: Any | None = None,
    literature_agent: Any | None = None,
    enable_human_feedback: bool = False,
) -> List:
    """
    Factory function to create the unified curve fitting pipeline.
    
    This pipeline handles BOTH single spectra and series:
    
    1. Analyze First Spectrum Data
       - Compute statistics, create initial visualization
       
    2. Human Feedback Refinement (optional)
       - LLM plans fitting approach
       - Human can refine the plan
       - Configuration is LOCKED for series processing
       
    3. Literature Search (if enabled)
       - Search for relevant fitting models
       - Runs only once (on first spectrum context)
       
    4. Unified Series Processing
       - Fits ALL spectra using locked configuration
       - Single spectrum = series of 1
       - Reuses fitting script across series
       
    5. Conditional Trend Analysis
       - For n>=2: Generates and executes trend analysis
       - For n=1: Skipped
       
    6. Synthesis
       - For n>=2: Cross-spectrum synthesis
       - For n=1: Single-spectrum interpretation
       
    7. Store Results
       - Save analysis images and artifacts
       
    8. Report Generation
       - Adapts format based on single vs series
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
        plot_fn: Function to plot curve data
        executor: Script executor instance
        output_dir: Output directory path
        preprocessor: Optional preprocessor agent
        literature_agent: Optional literature search agent
        enable_human_feedback: Enable human-in-the-loop refinement
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = []

    # Step 1: Analyze first spectrum data (compute stats, initial plot)
    # Note: For series, this runs on the FIRST spectrum only
    # (The agent pre-processes the first spectrum before calling the pipeline)
    pipeline.append(AnalyzeDataController(logger, plot_fn))

    # Step 2: Human feedback refinement on fitting approach
    # This plans the analysis and optionally allows human refinement
    # The fitting configuration is LOCKED after this step
    pipeline.append(
        HumanFeedbackRefinementController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            instructions=CURVE_ANALYSIS_INSTRUCTIONS,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
            max_iterations=5
        )
    )

    # Step 3: Literature search (runs once, uses first spectrum context)
    pipeline.append(
        LiteratureSearchController(
            logger=logger,
            literature_agent=literature_agent,
            output_dir=output_dir
        )
    )

    # Step 4: Unified series processing
    # Fits ALL spectra using the locked configuration
    # For single spectrum, this is effectively a "series of 1"
    pipeline.append(
        UnifiedSeriesProcessingController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
            correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
            quality_instructions=FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir=output_dir,
            plot_fn=plot_fn
        )
    )

    # Step 5: Conditional trend analysis (only for n>=2)
    pipeline.append(
        ConditionalTrendAnalysisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            output_dir=output_dir,
            max_corrections=3
        )
    )

    # Step 6: Synthesis (adapts to single vs series)
    pipeline.append(
        UnifiedCurveSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            single_spectrum_instructions=FITTING_INTERPRETATION_INSTRUCTIONS,
            output_dir=output_dir
        )
    )

    # Step 7: Store analysis results/images
    pipeline.append(
        StoreAnalysisResultsController(logger, store_fn)
    )

    # Step 8a: Single spectrum report (uses existing controller)
    # This only generates output for single spectra
    pipeline.append(
        GenerateCurveFittingReportController(logger, output_dir)
    )

    # Step 8b: Series report (only generates for n>=2)
    pipeline.append(
        UnifiedCurveReportController(logger, output_dir)
    )

    logger.info(f"Unified curve fitting pipeline created: {len(pipeline)} steps")
    return pipeline


# =============================================================================
# LEGACY PIPELINE FACTORY (for backward compatibility)
# =============================================================================

def create_curve_fitting_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    plot_fn: Callable,
    executor: Any,
    output_dir: str,
    preprocessor: Any | None = None,
    literature_agent: Any | None = None,
    enable_human_feedback: bool = False,
    settings: dict | None = None,  # Deprecated
) -> List:
    """
    BACKWARD COMPATIBLE: Creates curve fitting pipeline.
    
    Now returns the unified pipeline that handles both single spectra
    and series analysis.
    
    For explicit series analysis, use create_unified_curve_fitting_pipeline().
    """
    if settings is not None:
        import warnings
        warnings.warn(
            "The 'settings' parameter is deprecated and will be ignored.",
            DeprecationWarning
        )
    
    return create_unified_curve_fitting_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        parse_fn=parse_fn,
        store_fn=store_fn,
        plot_fn=plot_fn,
        executor=executor,
        output_dir=output_dir,
        preprocessor=preprocessor,
        literature_agent=literature_agent,
        enable_human_feedback=enable_human_feedback
    )