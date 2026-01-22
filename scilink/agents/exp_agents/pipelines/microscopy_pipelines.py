"""
Microscopy Analysis Pipeline Factories.
"""

from ..controllers.microscopy_controllers import (
    GetFFTParamsController,
    RunFFTNMFController,
    RunGlobalFFTController,
    BuildFFTNMFPromptController,
    FinalLLMAnalysisController,
    SeriesLoaderController,
    FirstFrameAnalysisController,
    UserFeedbackController,
    SeriesBatchController,
    SummaryScriptController,
    ReportGenerationController
)


def create_fftnmf_pipeline(model, logger, generation_config, safety_settings, settings, parse_fn, store_fn):
    """Create single-image FFT/NMF pipeline."""
    return [
        GetFFTParamsController(model, logger, generation_config, safety_settings),
        RunGlobalFFTController(logger, settings),
        RunFFTNMFController(logger, settings),
        BuildFFTNMFPromptController(logger),
        FinalLLMAnalysisController(model, logger, generation_config, safety_settings, parse_fn, store_fn),
    ]


def create_series_pipeline(model, logger, generation_config, safety_settings, settings, 
                           parse_fn, feedback_callback=None):
    """Create series analysis pipeline with feedback, script generation, and HTML report."""
    return [
        SeriesLoaderController(logger),
        FirstFrameAnalysisController(model, logger, generation_config, safety_settings, settings),
        UserFeedbackController(logger, settings, feedback_callback),
        SeriesBatchController(logger, settings),
        SummaryScriptController(model, logger, generation_config, safety_settings, parse_fn, settings),
        ReportGenerationController(logger, settings),
    ]


def create_batch_only_pipeline(model, logger, generation_config, safety_settings, settings, parse_fn, locked_params):
    """Create batch-only pipeline (skip first-frame analysis)."""
    
    class PresetParamsController:
        def __init__(self, params):
            self.params = params
        def execute(self, state):
            state["locked_params"] = self.params
            state["first_frame_results"] = {"llm_params": self.params}
            return state
    
    return [
        SeriesLoaderController(logger),
        PresetParamsController(locked_params),
        SeriesBatchController(logger, settings),
        SummaryScriptController(model, logger, generation_config, safety_settings, parse_fn, settings),
        ReportGenerationController(logger, settings),
    ]