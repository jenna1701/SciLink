# controllers/curve_fitting_controllers.py

"""
Controllers for the curve fitting analysis pipeline.
"""

import json
import logging
import os
import numpy as np
from typing import Callable, Any


class AnalyzeDataController:
    """Compute data statistics and create initial visualization."""

    def __init__(self, logger: logging.Logger, plot_fn: Callable):
        self.logger = logger
        self.plot_fn = plot_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🔍 --- Analyzing Data ---\n")

        try:
            data = state["curve_data"]

            if data.ndim == 1:
                x = np.arange(len(data))
                y = data
            elif data.shape[0] == 2:
                x, y = data[0], data[1]
            elif data.shape[1] == 2:
                x, y = data[:, 0], data[:, 1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            state["data_statistics"] = {
                "n_points": len(x),
                "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
                "y_mean": float(np.nanmean(y)),
                "y_std": float(np.nanstd(y)),
                "has_nans": bool(np.any(np.isnan(data))),
            }

            plot_bytes = self.plot_fn(state["curve_data"], state.get("system_info", {}))
            state["original_plot_bytes"] = plot_bytes
            state["analysis_images"] = [{"label": "Raw Data", "data": plot_bytes}]

            self.logger.info(f"  Points: {state['data_statistics']['n_points']}")
            self.logger.info(f"  X: {state['data_statistics']['x_range']}")
            self.logger.info(f"  Y: {state['data_statistics']['y_range']}")

        except Exception as e:
            self.logger.error(f"❌ Data analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Data analysis failed", "details": str(e)}

        return state


class PlanAnalysisController:
    """LLM examines data and plans the fitting approach."""

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        instructions: str,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.instructions = instructions

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🧠 --- Planning Analysis ---\n")

        try:
            prompt = [
                self.instructions,
                "\n## Data Plot",
                {"mime_type": "image/png", "data": state["original_plot_bytes"]},
                "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
                "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
            ]

            if state.get("analysis_hints"):
                prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")

            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result, error = self._parse(response)

            if error or not result:
                raise ValueError(f"Failed to parse: {error}")

            state["observations"] = result.get("observations", "")
            state["analysis_approach"] = result.get("analysis_approach", "Curve fitting")
            state["physical_model"] = result.get("physical_model", "Appropriate model")
            state["parameters_to_extract"] = result.get("parameters_to_extract", [])
            state["fitting_strategy"] = result.get("fitting_strategy", "Standard fitting")
            state["literature_query"] = result.get("literature_query")

            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Model: {state['physical_model']}")

        except Exception as e:
            self.logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Fit the data with an appropriate model"
            state["physical_model"] = "To be determined"
            state["parameters_to_extract"] = []
            state["fitting_strategy"] = "Standard curve fitting"
            state["literature_query"] = None

        return state


class LiteratureSearchController:
    """Search literature if enabled and query provided."""

    def __init__(
        self,
        logger: logging.Logger,
        literature_agent: Any | None,
        output_dir: str,
    ):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_results(self, query: str, report: str) -> dict:
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, "w") as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["report_file"] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save literature: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        # Skip if no agent or no query
        if self.literature_agent is None:
            self.logger.info("\n📚 --- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n📚 --- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n📚 --- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  ✅ Success")
            else:
                state["literature_context"] = None
                self.logger.warning(f"  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class ExecuteFittingController:
    """Generate and execute fitting script with self-correction."""

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.script_instructions = script_instructions
        self.correction_instructions = correction_instructions
        self.quality_instructions = quality_instructions
        self.output_dir = output_dir

    def _generate_script(self, state: dict) -> str:
        stats = state["data_statistics"]

        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])

        prompt = self.script_instructions.format(
            analysis_approach=state.get("analysis_approach", "Fit the data"),
            physical_model=state.get("physical_model", "Appropriate model"),
            parameters_to_extract=", ".join(state.get("parameters_to_extract", [])) or "relevant parameters",
            fitting_strategy=state.get("fitting_strategy", "Standard fitting"),
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=state.get("processed_data_path") or state["data_path"],
            n_points=stats["n_points"],
            x_min=stats["x_range"][0],
            x_max=stats["x_range"][1],
            y_min=stats["y_range"][0],
            y_max=stats["y_range"][1],
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Script generation failed: {error or 'no script'}")

        return result["script"]

    def _correct_script(self, state: dict, script: str, error_msg: str) -> str:
        prompt = self.correction_instructions.format(
            analysis_approach=state.get("analysis_approach", ""),
            physical_model=state.get("physical_model", ""),
            failed_script=script,
            error_message=error_msg,
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        if "diagnosis" in result:
            self.logger.info(f"  Diagnosis: {result['diagnosis']}")

        return result["script"]

    def _assess_quality(self, state: dict, plot_bytes: bytes, metrics: dict) -> dict:
        prompt = [
            self.quality_instructions.format(
                analysis_approach=state.get("analysis_approach", ""),
                physical_model=state.get("physical_model", ""),
                metrics=json.dumps(metrics, indent=2),
            ),
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Fit Result",
            {"mime_type": "image/png", "data": plot_bytes},
        ]

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, _ = self._parse(response)

        if not result:
            return {"is_acceptable": True, "quality_score": 0.5}

        is_ok = result.get("is_acceptable", True)
        if isinstance(is_ok, str):
            is_ok = is_ok.lower() == "true"
        result["is_acceptable"] = bool(is_ok)

        return result

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n⚙️ --- Executing Fitting ---\n")

        script = None
        last_error = ""
        exec_result = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self.logger.info(f"  Attempt {attempt}/{self.MAX_ATTEMPTS}")

            try:
                if attempt == 1:
                    script = self._generate_script(state)
                else:
                    script = self._correct_script(state, script, last_error)

                exec_result = self.executor.execute_script(script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    self.logger.info("  ✅ Script executed")
                    break
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(f"  ❌ Failed: {last_error[:150]}")

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"  ❌ Error: {e}")
        else:
            state["error_dict"] = {"error": "Script generation failed", "details": last_error}
            return state

        # Parse results
        fit_results = {}
        for line in (exec_result.get("stdout") or "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
                except json.JSONDecodeError as e:
                    self.logger.warning(f"  Could not parse results: {e}")
                break

        # Load visualization
        viz_path = os.path.join(self.output_dir, "fit_visualization.png")
        if not os.path.exists(viz_path):
            state["error_dict"] = {"error": "No fit_visualization.png generated"}
            return state

        with open(viz_path, "rb") as f:
            plot_bytes = f.read()

        # Assess quality
        quality = self._assess_quality(state, plot_bytes, fit_results.get("fit_quality", {}))
        self.logger.info(f"  Quality: {quality.get('quality_score', 'N/A')}, OK: {quality.get('is_acceptable')}")

        # Store results
        state["final_script"] = script
        state["final_plot_bytes"] = plot_bytes
        state["fit_results"] = fit_results
        state["quality_assessment"] = quality
        state["analysis_images"].append({
            "label": fit_results.get("model_type", "Fit"),
            "data": plot_bytes,
        })

        state["result_json"] = {
            "model_type": fit_results.get("model_type"),
            "fitting_parameters": fit_results.get("parameters", {}),
            "fit_quality": fit_results.get("fit_quality", {}),
            "summary": fit_results.get("summary"),
            "literature_files": state.get("literature_files"),
        }

        return state


class BuildInterpretationPromptController:
    """Assemble prompt for final interpretation."""

    def __init__(self, logger: logging.Logger, instructions: str):
        self.logger = logger
        self.instructions = instructions

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n📝 --- Building Interpretation Prompt ---\n")

        fit_results = state.get("fit_results", {})

        formatted = self.instructions.format(
            model_type=fit_results.get("model_type", "Curve fit"),
            summary=fit_results.get("summary", "Fitting complete"),
        )

        state["instruction_prompt"] = formatted
        state["final_prompt_parts"] = [
            formatted,
            "\n## Original Data",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Fit Result",
            {"mime_type": "image/png", "data": state["final_plot_bytes"]},
            "\n## Parameters\n" + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit Quality\n" + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        if state.get("literature_context"):
            state["final_prompt_parts"].extend(["\n## Literature", state["literature_context"]])

        return state


class RunCurvePreprocessingController:
    """Optional data preprocessing."""

    def __init__(self, logger: logging.Logger, preprocessor: Any, output_dir: str):
        self.logger = logger
        self.preprocessor = preprocessor
        self.output_dir = output_dir

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if self.preprocessor is None:
            return state

        self.logger.info("\n🛠️ --- Preprocessing ---\n")

        try:
            processed_data, data_quality = self.preprocessor.run_preprocessing(
                state["curve_data"], state.get("system_info", {})
            )

            state["curve_data"] = processed_data
            state["data_quality"] = data_quality

            pid = os.getpid()
            processed_path = os.path.join(self.output_dir, f"temp_processed_{pid}.npy")
            np.save(processed_path, processed_data)
            state["processed_data_path"] = processed_path

            self.logger.info(f"  ✅ Saved to {processed_path}")

        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Preprocessing failed", "details": str(e)}

        return state