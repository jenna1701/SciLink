"""
SAM Analysis Pipelines

Factory functions for creating SAM analysis pipelines.
"""

import logging
from typing import Callable, List

from ..controllers.sam_controllers import (
    # Single-image pipeline controllers
    RunSAMRefinementLoopController,
    CalculateSAMStatsController,
    BuildSAMPromptController,
    RunFinalInterpretationController,
    StoreAnalysisResultsController,
    # Batch pipeline controllers
    HumanFeedbackRefinementController,
    BatchImageProcessingController,
    CustomAnalysisScriptController,
    BatchSynthesisController,
    ReportGenerationController
)


def create_sam_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Factory function to create the single-image SAM analysis pipeline.
    
    This pipeline:
    1. Runs SAM segmentation with optional LLM-driven refinement
    2. Calculates morphological statistics
    3. Builds the final prompt with results
    4. Generates scientific interpretation via LLM
    5. Stores analysis images for feedback
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        settings: Pipeline settings dict
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = [
        RunSAMRefinementLoopController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            settings=settings,
            parse_fn=parse_fn
        ),
        CalculateSAMStatsController(
            logger=logger
        ),
        BuildSAMPromptController(
            logger=logger
        ),
        RunFinalInterpretationController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn
        ),
        StoreAnalysisResultsController(
            logger=logger,
            store_fn=store_fn
        )
    ]
    
    return pipeline


def create_sam_batch_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable
) -> List:
    """
    Factory function to create the batch SAM analysis pipeline.
    
    This pipeline:
    1. Human feedback refinement on first image
    2. Batch processing of all images with refined parameters
    3. Custom analysis script generation and execution
    4. Scientific synthesis of batch findings
    5. HTML report generation
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        settings: Pipeline settings dict
        parse_fn: Function to parse LLM responses
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = [
        HumanFeedbackRefinementController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        ),
        BatchImageProcessingController(
            logger=logger,
            settings=settings
        ),
        CustomAnalysisScriptController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        ),
        BatchSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        ),
        ReportGenerationController(
            logger=logger,
            settings=settings
        )
    ]
    
    return pipeline