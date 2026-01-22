"""
FFT/NMF Controllers for Microscopy Analysis.

Single Image Controllers:
- GetFFTParamsController
- RunFFTNMFController  
- RunGlobalFFTController
- BuildFFTNMFPromptController
- FinalLLMAnalysisController

Series Controllers:
- SeriesLoaderController
- FirstFrameAnalysisController
- UserFeedbackController
- SeriesBatchController
- SummaryScriptController
- ReportGenerationController
"""

import os
import sys
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
import base64

from ..instruct import FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS
from ..instruct import SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS, SERIES_ANALYSIS_INSTRUCTIONS
from ....tools.image_processor import normalize_and_convert_to_image_bytes, calculate_global_fft
from ....tools.fft_nmf import SlidingFFTNMF



# =============================================================================
# SINGLE IMAGE CONTROLLERS
# =============================================================================

class GetFFTParamsController:
    """[🧠 LLM Step] Asks an LLM to suggest FFT/NMF parameters."""
    
    def __init__(self, model, logger, generation_config, safety_settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🧠 LLM Step: Reasoning about FFT/NMF parameters...")
        image_blob = state["image_blob"]
        system_info = state["system_info"]
        
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        if system_info:
            prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(system_info, indent=2)}")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result = json.loads(response.text)
            state["llm_params"] = result

            print("\n" + "="*60)
            print("🧠 LLM REASONING")
            print(f"   Explanation: {result.get('explanation', 'N/A')}")
            print(f"   Params: window_size_nm={result.get('window_size_nm')}, n_components={result.get('n_components')}")
            print("="*60 + "\n")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: {e}")
            state["llm_params"] = {}
            
        return state


class RunFFTNMFController:
    """[🛠️ Tool Step] Runs the FFT/NMF analysis using SlidingFFTNMF."""
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Running Sliding FFT + NMF...")
        llm_params = state.get("llm_params", {})
        
        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
            
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels,
                window_size_y=ws_pixels,
                window_step_x=step,
                window_step_y=step,
                components=nc
            )
            
            image_array = state["preprocessed_image_array"]
            components, abundances = analyzer.analyze(image_array, output_dir=None)
            
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            self.logger.info("✅ FFT/NMF complete.")
            
        except Exception as e:
            self.logger.error(f"❌ FFT/NMF failed: {e}")
            state["fft_components"] = None
            state["fft_abundances"] = None
            
        return state


class RunGlobalFFTController:
    """[🛠️ Tool Step] Calculates global FFT of the image."""
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Running Global FFT...")
        
        try:
            image_array = state["preprocessed_image_array"]
            output_dir = self.settings.get("visualization_dir", ".")
            
            base_name = os.path.splitext(os.path.basename(str(state.get("image_path", "image"))))[0]
            safe_name = "".join(c if c.isalnum() else "_" for c in base_name)
            filepath = os.path.join(output_dir, f"{safe_name}_global_fft.png")
            
            global_fft_image = calculate_global_fft(image_array, save_path=filepath)
            state["global_fft_image"] = global_fft_image
            self.logger.info("✅ Global FFT complete.")

        except Exception as e:
            self.logger.error(f"❌ Global FFT failed: {e}")
            state["global_fft_image"] = None
            
        return state


class BuildFFTNMFPromptController:
    """[📝 Prep Step] Builds the final prompt with analysis results."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("📝 Building final prompt...")
        
        prompt_parts = [state["instruction_prompt"]]
        
        if state.get("additional_top_level_context"):
            prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
        prompt_parts.append("\n\nPrimary Microscopy Image:\n")
        prompt_parts.append(state["image_blob"])

        global_fft = state.get("global_fft_image")
        if global_fft is not None:
            try:
                fft_bytes = normalize_and_convert_to_image_bytes(global_fft, log_scale=False)
                prompt_parts.append("\n\nGlobal FFT:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": fft_bytes})
                state["analysis_images"].append({"label": "Global FFT", "data": fft_bytes})
            except Exception as e:
                self.logger.error(f"Failed to add Global FFT: {e}")

        components = state.get("fft_components")
        abundances = state.get("fft_abundances")

        if components is not None and abundances is not None:
            prompt_parts.append("\n\nSliding FFT + NMF Results:")
            for i in range(components.shape[0]):
                try:
                    comp_bytes = normalize_and_convert_to_image_bytes(components[i], log_scale=True)
                    abun_bytes = normalize_and_convert_to_image_bytes(abundances[i])
                    
                    prompt_parts.append(f"\nComponent {i+1}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": comp_bytes})
                    prompt_parts.append(f"\nAbundance Map {i+1}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": abun_bytes})
                    
                    state["analysis_images"].append({"label": f"Abundance {i+1}", "data": abun_bytes})
                except Exception as e:
                    self.logger.error(f"Failed to add NMF result {i+1}: {e}")

        prompt_parts.append(f"\n\nSystem Info:\n{json.dumps(state['system_info'], indent=2)}")
        prompt_parts.append("\n\nProvide your analysis in JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        return state


class FinalLLMAnalysisController:
    """[🧠 LLM Step] Executes the final LLM call with the built prompt."""
    
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn, store_fn):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.parse_fn = parse_fn
        self.store_fn = store_fn
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
            
        self.logger.info("🧠 Final LLM Analysis: Generating structured analysis...")
        
        prompt_parts = state.get("final_prompt_parts")
        if not prompt_parts:
            state["error_dict"] = {"error": "No prompt parts found"}
            return state
        
        if self.store_fn and state.get("analysis_images"):
            self.store_fn(state["analysis_images"])
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result = self.parse_fn(response.text) if self.parse_fn else self._parse_json(response.text)
            
            if result is None:
                state["error_dict"] = {"error": "Failed to parse LLM response"}
            else:
                state["result_json"] = result
                self.logger.info("✅ Final LLM Analysis complete.")
                
        except Exception as e:
            self.logger.error(f"❌ LLM Analysis failed: {e}")
            state["error_dict"] = {"error": "LLM analysis failed", "details": str(e)}
        
        return state
    
    def _parse_json(self, text: str) -> Optional[dict]:
        """Fallback JSON parser."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None


# =============================================================================
# SERIES CONTROLLERS
# =============================================================================

class SeriesLoaderController:
    """[📂 Load Step] Load image series from directory, TIFF, or array."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def execute(self, state: dict) -> dict:
        self.logger.info("📂 Loading image series...")
        series_input = state.get("series_input")
        
        try:
            if isinstance(series_input, np.ndarray):
                if series_input.ndim == 2:
                    series_data = series_input[np.newaxis, :, :]
                elif series_input.ndim == 3:
                    series_data = series_input
                else:
                    raise ValueError(f"Array must be 2D or 3D, got {series_input.ndim}D")
                state["series_source"] = "array"
                
            elif isinstance(series_input, str):
                if os.path.isdir(series_input):
                    series_data = self._load_directory(series_input)
                    state["series_source"] = "directory"
                elif series_input.lower().endswith(('.tif', '.tiff')):
                    series_data = self._load_tiff(series_input)
                    state["series_source"] = "tiff"
                else:
                    raise ValueError(f"Unsupported: {series_input}")
            else:
                raise TypeError(f"Expected str or ndarray, got {type(series_input)}")
            
            state["series_data"] = series_data
            state["n_frames"] = series_data.shape[0]
            state["frame_shape"] = series_data.shape[1:]
            state["first_frame"] = series_data[0]
            
            self.logger.info(f"✅ Loaded: {state['n_frames']} frames, shape {state['frame_shape']}")
            
        except Exception as e:
            self.logger.error(f"❌ Load failed: {e}")
            state["error_dict"] = {"error": "Load failed", "details": str(e)}
            
        return state
    
    def _load_directory(self, directory: str) -> np.ndarray:
        from skimage import io, color
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = sorted([f for f in os.listdir(directory) if f.lower().endswith(valid_ext)])
        
        if not files:
            raise ValueError(f"No images in {directory}")
        
        frames = []
        for f in files:
            img = io.imread(os.path.join(directory, f))
            if img.ndim == 3:
                img = color.rgb2gray(img[:, :, :3])
            frames.append(img)
        
        return np.stack(frames, axis=0)
    
    def _load_tiff(self, filepath: str) -> np.ndarray:
        from skimage import io, color
        
        stack = io.imread(filepath)
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        elif stack.ndim == 4:
            stack = np.array([color.rgb2gray(f[:, :, :3]) for f in stack])
        return stack


class FirstFrameAnalysisController:
    """[🔬 Analysis Step] Analyze first frame with LLM-guided params."""
    
    def __init__(self, model, logger, generation_config, safety_settings, settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
            
        self.logger.info("🔬 Analyzing first frame...")
        
        from ....tools.image_processor import convert_numpy_to_jpeg_bytes, preprocess_image
        
        first_frame = state["first_frame"]
        
        if first_frame.dtype in [np.float32, np.float64, float]:
            frame_min, frame_max = first_frame.min(), first_frame.max()
            if frame_max > frame_min:
                first_frame = ((first_frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                first_frame = np.zeros_like(first_frame, dtype=np.uint8)
        
        preprocessed, _ = preprocess_image(first_frame)
        image_bytes = convert_numpy_to_jpeg_bytes(preprocessed)
        
        state["preprocessed_image_array"] = preprocessed
        state["image_blob"] = {"mime_type": "image/jpeg", "data": image_bytes}
        state["image_path"] = "first_frame"
        
        GetFFTParamsController(self.model, self.logger, self.generation_config, self.safety_settings).execute(state)
        RunGlobalFFTController(self.logger, self.settings).execute(state)
        RunFFTNMFController(self.logger, self.settings).execute(state)
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": state.get("llm_params", {})
        }
        
        self.logger.info("✅ First frame analysis complete.")
        return state


class UserFeedbackController:
    """[👤 Feedback Step] Collect user feedback on parameters."""
    
    def __init__(self, logger, settings, feedback_callback=None):
        self.logger = logger
        self.settings = settings
        self.feedback_callback = feedback_callback
        self.max_iterations = settings.get('max_feedback_iterations', 3)
    
    def _display_results(self, state: dict, iteration: int) -> None:
        """Display current results for review."""
        llm_params = state.get("first_frame_results", {}).get("llm_params", {})
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        # Save visualization
        output_dir = Path(self.settings.get("visualization_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        review_path = output_dir / f"review_iteration_{iteration}.png"
        
        if components is not None and abundances is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
            
            for i in range(n_comps):
                axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Abundance {i+1}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'FFT/NMF Analysis - Iteration {iteration}', fontsize=14)
            plt.tight_layout()
            plt.savefig(review_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print("\n" + "=" * 70)
        print(f"🔬 FFT/NMF ANALYSIS REVIEW - Iteration {iteration}")
        print("=" * 70)
        print(f"\n🖼️  Visualization saved to: {review_path}")
        print(f"\n📊 Results:")
        if components is not None:
            print(f"   Components: {components.shape[0]}, size: {components.shape[1]}x{components.shape[2]}")
        print(f"\n⚙️  Parameters:")
        print(f"   Window Size (nm): {llm_params.get('window_size_nm', 'auto')}")
        print(f"   NMF Components: {llm_params.get('n_components', 4)}")
        if llm_params.get('explanation'):
            print(f"\n🧠 Reasoning: {llm_params.get('explanation')}")
        print("-" * 70)
    
    def _get_user_input(self, prompt: str) -> str:
        """Get user input, handling different environments."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            return input().strip()
        except EOFError:
            return ""
    
    def _collect_feedback(self, state: dict) -> dict:
        """Collect user feedback."""
        llm_params = state.get("first_frame_results", {}).get("llm_params", {})
        
        print("\n👤 Options:")
        print("   [1] Accept (proceed to batch)")
        print("   [2] Modify parameters")
        print("   [c] Cancel")
        
        choice = self._get_user_input("\nChoice [1/2/c]: ").lower()
        
        if choice == '1' or choice == '':
            return {"action": "accept"}
        elif choice == 'c':
            return {"action": "cancel"}
        elif choice == '2':
            print("\nEnter new values (press Enter to keep current):")
            
            mods = {}
            ws = self._get_user_input(f"   Window size (nm) [{llm_params.get('window_size_nm', 'auto')}]: ")
            if ws:
                try:
                    mods['window_size_nm'] = float(ws)
                except ValueError:
                    print("   Invalid, keeping current")
            
            nc = self._get_user_input(f"   Components [{llm_params.get('n_components', 4)}]: ")
            if nc:
                try:
                    mods['n_components'] = int(nc)
                except ValueError:
                    print("   Invalid, keeping current")
            
            if mods:
                return {"action": "modify", "params": mods}
            return {"action": "accept"}
        else:
            print("Invalid choice, accepting current.")
            return {"action": "accept"}
    
    def _rerun_analysis(self, state: dict, new_params: dict) -> dict:
        """Re-run with updated parameters."""
        self.logger.info("🔄 Re-running with updated parameters...")
        
        llm_params = state.get("first_frame_results", {}).get("llm_params", {}).copy()
        llm_params.update(new_params)
        state["llm_params"] = llm_params
        
        RunFFTNMFController(self.logger, self.settings).execute(state)
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": llm_params
        }
        return state
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        print("\n\n" + "=" * 70)
        print("👤 HUMAN FEEDBACK LOOP")
        print("=" * 70)
        
        for iteration in range(1, self.max_iterations + 1):
            self._display_results(state, iteration)
            
            if self.feedback_callback:
                feedback = self.feedback_callback(state)
            else:
                feedback = self._collect_feedback(state)
            
            if feedback.get("action") == "accept":
                self.logger.info("✅ User accepted results.")
                state["locked_params"] = state.get("first_frame_results", {}).get("llm_params", {})
                return state
            
            elif feedback.get("action") == "cancel":
                self.logger.info("❌ User cancelled.")
                state["batch_cancelled"] = True
                return state
            
            elif feedback.get("action") == "modify" and feedback.get("params"):
                state = self._rerun_analysis(state, feedback["params"])
        
        self.logger.warning(f"Max iterations reached, using current parameters.")
        state["locked_params"] = state.get("first_frame_results", {}).get("llm_params", {})
        return state


class SeriesBatchController:
    """[⚡ Batch Step] Process full series with locked parameters."""
    
    def __init__(self, logger, settings):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("⚡ Processing full series...")
        
        locked_params = state.get("locked_params", {})
        series_data = state["series_data"]
        n_frames = state["n_frames"]
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        ws_nm = locked_params.get("window_size_nm", 10.0)
        n_components = locked_params.get("n_components", 4)
        
        ws_pixels = int(round(ws_nm / nm_per_pixel)) if nm_per_pixel > 0 else 64
        good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
        ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels,
                window_size_y=ws_pixels,
                window_step_x=step,
                window_step_y=step,
                components=n_components
            )
            
            print(f"⏳ Processing {n_frames} frames...")
            components, abundances = analyzer.analyze(series_data, output_dir=None)
            
            state["series_components"] = components
            state["series_abundances"] = abundances
            state["batch_params"] = {
                "window_size_pixels": ws_pixels,
                "window_size_nm": ws_nm,
                "n_components": n_components,
                "n_frames": n_frames
            }
            
            output_dir = self.settings.get("output_dir", "analysis_output")
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "series_components.npy"), components)
            np.save(os.path.join(output_dir, "series_abundances.npy"), abundances)
            
            print(f"✅ Done! Components: {components.shape}, Abundances: {abundances.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch failed: {e}")
            state["error_dict"] = {"error": "Batch failed", "details": str(e)}
        
        return state


class SummaryScriptController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes a custom Python script for trend analysis.
    Follows the same pattern as SAM's CustomAnalysisScriptController.
    """
    
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn, settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get("output_dir", "analysis_output"))
        self.max_retries = settings.get("max_script_retries", 3)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("🧠 LLM generating custom analysis script...")
        
        # First, compute basic stats to give LLM context
        basic_stats = self._compute_basic_stats(state)
        state["basic_stats"] = basic_stats
        
        # Generate script via LLM
        script_result = self._generate_analysis_script(state, basic_stats)
        
        if not script_result or "script" not in script_result:
            self.logger.warning("LLM script generation failed, using fallback template")
            script = self._fallback_script(state)
        else:
            script = script_result["script"]
            approach = script_result.get("analysis_approach", "trend_analysis")
            metrics = script_result.get("key_metrics", [])
            self.logger.info(f"   📊 Analysis approach: {approach}")
            self.logger.info(f"   📈 Key metrics: {metrics}")
        
        # Execute with retry loop
        script_path = self.output_dir / "analyze_results.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        success, stdout, script = self._execute_with_retry(script, script_path, state, basic_stats)
        
        state["analysis_script_path"] = str(script_path)
        state["script_success"] = success
        state["script_output"] = stdout
        
        return state
    
    def _compute_basic_stats(self, state: dict) -> dict:
        """Compute basic statistics to give LLM context about the data."""
        components = state.get("series_components")
        abundances = state.get("series_abundances")
        
        if components is None or abundances is None:
            return {}
        
        n_frames, n_comps = abundances.shape[:2]
        mean_abundances = abundances.mean(axis=(2, 3))  # (n_frames, n_comps)
        
        stats = {
            "n_frames": n_frames,
            "n_components": n_comps,
            "component_shape": list(components.shape),
            "abundance_shape": list(abundances.shape),
            "components": []
        }
        
        for i in range(n_comps):
            ts = mean_abundances[:, i]
            slope = float(np.polyfit(range(len(ts)), ts, 1)[0])
            
            # Detect patterns
            if len(ts) > 2:
                fft_mag = np.abs(np.fft.fft(ts - ts.mean()))
                half_len = max(1, len(ts) // 2)
                dominant_freq_idx = np.argmax(fft_mag[1:half_len]) + 1 if half_len > 1 else 1
                period = n_frames / dominant_freq_idx if dominant_freq_idx > 0 else None
                has_periodicity = bool(fft_mag[dominant_freq_idx] > 2 * np.mean(fft_mag[1:half_len])) if half_len > 1 else False
            else:
                period = None
                has_periodicity = False
            
            stats["components"].append({
                "index": i + 1,
                "mean": float(np.mean(ts)),
                "std": float(np.std(ts)),
                "min": float(np.min(ts)),
                "max": float(np.max(ts)),
                "slope": slope,
                "trend": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
                "has_periodicity": has_periodicity,
                "estimated_period": float(period) if period else None
            })
        
        # Cross-correlations
        if n_comps > 1:
            corr_matrix = np.corrcoef(mean_abundances.T)
            stats["correlations"] = []
            for i in range(n_comps):
                for j in range(i + 1, n_comps):
                    stats["correlations"].append({
                        "pair": [i + 1, j + 1],
                        "correlation": float(corr_matrix[i, j])
                    })
        
        return stats
    
    def _generate_analysis_script(self, state: dict, basic_stats: dict) -> Optional[dict]:
        """Generate custom analysis script using LLM. Returns dict with 'script' key."""
        
        output_dir_str = str(self.output_dir)
        
        prompt = f'''You are a scientific data analysis expert. Generate a Python script to analyze FFT/NMF decomposition results.

**DATA FILES** (in {output_dir_str}):
- series_components.npy: shape {basic_stats.get("component_shape", "unknown")} - NMF frequency components
- series_abundances.npy: shape {basic_stats.get("abundance_shape", "unknown")} - abundance maps (frames, components, grid_h, grid_w)

**PRE-COMPUTED STATISTICS:**
{json.dumps(basic_stats, indent=2)}

**REQUIREMENTS:**
1. Complete, runnable Python script
2. Libraries: numpy, matplotlib, scipy, json, pathlib only
3. Save figures as PNG to OUTPUT_DIR
4. Save trends.json with analysis results
5. Print summary to stdout

**ANALYSIS TO INCLUDE:**
- Plot NMF components as images
- Plot mean abundance timeseries per component  
- Compute trend directions and slopes
- If correlations > 0.5, plot correlation matrix
- Save findings to trends.json

Return a JSON object with:
{{
    "analysis_approach": "brief description of approach",
    "key_metrics": ["list", "of", "metrics"],
    "reasoning": "why this analysis is appropriate",
    "script": "complete Python script as a string"
}}

The script should start with imports and define OUTPUT_DIR = Path("{output_dir_str}")'''

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM script generation parse error: {error_dict}")
                # Try to salvage if we got partial results
                if result_json and "script" in result_json:
                    return result_json
                return None
            
            if not result_json or "script" not in result_json:
                self.logger.warning("LLM response missing 'script' key")
                return None
            
            self.logger.info("✅ LLM generated analysis script")
            return result_json
            
        except Exception as e:
            self.logger.error(f"LLM script generation failed: {e}")
            return None
    
    def _execute_with_retry(self, script: str, script_path: Path, state: dict, basic_stats: dict) -> tuple:
        """Execute script with retry on errors, asking LLM to fix."""
        import subprocess
        
        for attempt in range(self.max_retries):
            # Save script
            with open(script_path, 'w') as f:
                f.write(script)
            
            self.logger.info(f"📜 Executing script (attempt {attempt + 1}/{self.max_retries})...")
            
            try:
                result = subprocess.run(
                    ['python', str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(self.output_dir)
                )
                
                if result.returncode == 0:
                    self.logger.info("✅ Script executed successfully!")
                    print(result.stdout)
                    return True, result.stdout, script
                else:
                    error_msg = result.stderr
                    self.logger.warning(f"Script failed: {error_msg[:200]}")
                    
                    if attempt < self.max_retries - 1:
                        corrected = self._fix_script_with_llm(script, error_msg, basic_stats, attempt + 1)
                        if corrected:
                            script = corrected
                        else:
                            break
                    
            except subprocess.TimeoutExpired:
                self.logger.warning("Script timed out")
                break
            except Exception as e:
                self.logger.warning(f"Execution error: {e}")
                break
        
        # All retries failed, use fallback
        self.logger.warning("Using fallback script after retries exhausted")
        script = self._fallback_script(state)
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(self.output_dir)
            )
            print(result.stdout)
            return result.returncode == 0, result.stdout, script
        except Exception as e:
            self.logger.error(f"Fallback script also failed: {e}")
            return False, "", script
    
    def _fix_script_with_llm(self, original_script: str, error_msg: str, basic_stats: dict, attempt: int) -> Optional[str]:
        """Use LLM to correct a failed script."""
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        
        if len(error_msg) > 1000:
            error_msg = error_msg[:500] + "\n...[truncated]...\n" + error_msg[-500:]
        
        prompt = f'''Fix this Python script that failed to execute.

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_msg}
```

**DATA CONTEXT:**
- Components shape: {basic_stats.get("component_shape")}
- Abundances shape: {basic_stats.get("abundance_shape")}
- Output directory: {self.output_dir}

Return a JSON object with:
{{
    "diagnosis": "what caused the error",
    "script": "corrected complete Python script"
}}'''

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                self.logger.warning("Failed to parse correction response")
                return None
            
            diagnosis = result_json.get("diagnosis", "N/A")
            self.logger.info(f"   📋 Diagnosis: {diagnosis}")
            
            return result_json.get("script")
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None
    
    def _fallback_script(self, state: dict) -> str:
        """Simple fallback script if LLM generation fails."""
        return f'''#!/usr/bin/env python3
"""Fallback analysis script for FFT/NMF results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("{self.output_dir}")

def main():
    # Load data
    components = np.load(OUTPUT_DIR / "series_components.npy")
    abundances = np.load(OUTPUT_DIR / "series_abundances.npy")
    print(f"Loaded: components {{components.shape}}, abundances {{abundances.shape}}")
    
    n_comps = components.shape[0]
    
    # Plot components
    fig, axes = plt.subplots(1, n_comps, figsize=(4*n_comps, 4))
    if n_comps == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(components[i], cmap='viridis')
        ax.set_title(f'Component {{i+1}}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "components.png", dpi=150)
    plt.close()
    print("Saved: components.png")
    
    # Plot abundance timeseries
    mean_ab = abundances.mean(axis=(2, 3))  # (n_frames, n_comps)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_comps):
        ax.plot(mean_ab[:, i], 'o-', label=f'Component {{i+1}}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Abundance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "abundance_timeseries.png", dpi=150)
    plt.close()
    print("Saved: abundance_timeseries.png")
    
    # Compute trends
    trends = {{}}
    for i in range(n_comps):
        data = mean_ab[:, i]
        slope = np.polyfit(range(len(data)), data, 1)[0]
        trends[f"component_{{i+1}}"] = {{
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "trend": "increasing" if slope > 0 else "decreasing",
            "slope": float(slope)
        }}
    
    # Save trends
    with open(OUTPUT_DIR / "trends.json", 'w') as f:
        json.dump(trends, f, indent=2)
    print("Saved: trends.json")
    
    # Print summary
    print("\\nTrend Analysis:")
    for k, v in trends.items():
        print(f"  {{k}}: {{v['trend']}} (slope={{v['slope']:.4f}})")

if __name__ == "__main__":
    main()
'''



class ReportGenerationController:
    """
    Unified Report Generation Controller for Microscopy Analysis.
    
    Automatically detects analysis mode from state:
    - Single Image: Generates JSON + optional HTML
    - Series: Generates comprehensive HTML report
    
    Both modes use consistent scientific claims format with "Has anyone" questions.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, 
                 safety_settings, parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = settings.get('max_llm_retries', 2)
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.enable_human_feedback = settings.get('enable_report_feedback', False)
        self.generate_html_for_single = settings.get('generate_html_for_single_image', False)
    
    def execute(self, state: dict) -> dict:
        """Execute report generation - auto-detects single vs series mode."""
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        # Detect mode
        is_series = self._detect_mode(state)
        mode_str = "series" if is_series else "single image"
        self.logger.info(f"📄 Generating report ({mode_str} mode)...")
        
        if is_series:
            return self._execute_series_mode(state)
        else:
            return self._execute_single_image_mode(state)
    
    def _detect_mode(self, state: dict) -> bool:
        """Detect if this is series or single image analysis."""
        # Series indicators
        if state.get("series_data") is not None:
            return True
        if state.get("n_frames", 0) > 1:
            return True
        if state.get("series_components") is not None:
            return True
        if state.get("batch_params") is not None:
            return True
        
        # Single image indicators
        if state.get("image_path") and not state.get("series_input"):
            return False
        
        # Default to single if unclear
        return False
    
    # =========================================================================
    # SINGLE IMAGE MODE
    # =========================================================================
    
    def _execute_single_image_mode(self, state: dict) -> dict:
        """Execute single image analysis and report generation."""
        self.logger.info("🔬 Processing single image analysis...")
        
        # Compute statistics from FFT/NMF results
        stats = self._compute_single_image_stats(state)
        
        # Get LLM analysis
        llm_analysis = self._get_single_image_llm_analysis(state, stats)
        
        if llm_analysis is None:
            self.logger.warning("LLM analysis failed, using fallback")
            llm_analysis = self._generate_single_image_fallback(state, stats)
        
        # Store results in state (JSON format for compatibility)
        state["result_json"] = {
            "detailed_analysis": llm_analysis.get("detailed_analysis", ""),
            "scientific_claims": self._convert_claims_to_legacy_format(
                llm_analysis.get("scientific_claims", [])
            ),
            "component_interpretations": llm_analysis.get("component_interpretations", [])
        }
        state["llm_report_analysis"] = llm_analysis
        
        # Optionally generate HTML report
        if self.generate_html_for_single:
            self._generate_single_image_html(state, stats, llm_analysis)
        
        self.logger.info("✅ Single image analysis complete")
        return state
    
    def _compute_single_image_stats(self, state: dict) -> dict:
        """Compute statistics for single image analysis."""
        stats = {
            "has_fft_nmf": False,
            "n_components": 0,
            "components": []
        }
        
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        if components is not None and abundances is not None:
            stats["has_fft_nmf"] = True
            stats["n_components"] = components.shape[0]
            
            for i in range(components.shape[0]):
                abun = abundances[i]
                stats["components"].append({
                    "index": i + 1,
                    "abundance_mean": float(np.mean(abun)),
                    "abundance_std": float(np.std(abun)),
                    "abundance_max": float(np.max(abun)),
                    "spatial_coverage": float(np.sum(abun > np.mean(abun)) / abun.size)
                })
        
        return stats
    
    def _get_single_image_llm_analysis(self, state: dict, stats: dict) -> Optional[dict]:
        """Get LLM analysis for single image."""
        self.logger.info("🧠 LLM analyzing single image...")
        
        prompt_parts = [SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS]
        
        # Add primary image
        if state.get("image_blob"):
            prompt_parts.append("\n\n## Primary Microscopy Image\n")
            prompt_parts.append(state["image_blob"])
        
        # Add system info
        system_info = state.get("system_info", {})
        if system_info:
            prompt_parts.append(f"\n\n## System Information\n```json\n{json.dumps(system_info, indent=2)}\n```")
        
        # Add FFT/NMF results if available
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        if components is not None:
            prompt_parts.append("\n\n## FFT/NMF Components\n")
            for i in range(min(components.shape[0], 6)):
                comp_bytes = self._array_to_png_bytes(components[i])
                if comp_bytes:
                    prompt_parts.append(f"\nComponent {i+1} (frequency pattern):\n")
                    prompt_parts.append({"mime_type": "image/png", "data": comp_bytes})
        
        if abundances is not None:
            prompt_parts.append("\n\n## Abundance Maps\n")
            for i in range(min(abundances.shape[0], 6)):
                abun_bytes = self._array_to_png_bytes(abundances[i])
                if abun_bytes:
                    prompt_parts.append(f"\nAbundance Map {i+1}:\n")
                    prompt_parts.append({"mime_type": "image/png", "data": abun_bytes})
        
        # Add global FFT if available
        global_fft = state.get("global_fft_image")
        if global_fft is not None:
            fft_bytes = self._array_to_png_bytes(global_fft)
            if fft_bytes:
                prompt_parts.append("\n\n## Global FFT\n")
                prompt_parts.append({"mime_type": "image/png", "data": fft_bytes})
        
        prompt_parts.append("\n\nProvide your analysis as a JSON object. Output ONLY the JSON.")
        
        # Call LLM
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    contents=prompt_parts,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                
                result, error = self._parse_llm_response(response)
                
                if error:
                    self.logger.warning(f"Parse error (attempt {attempt+1}): {error}")
                    continue
                
                if result and "detailed_analysis" in result:
                    self.logger.info("✅ LLM analysis complete")
                    return result
                    
            except Exception as e:
                self.logger.error(f"LLM error (attempt {attempt+1}): {e}")
        
        return None
    
    def _generate_single_image_fallback(self, state: dict, stats: dict) -> dict:
        """Generate fallback analysis for single image."""
        n_comps = stats.get("n_components", 0)
        
        return {
            "detailed_analysis": f"FFT/NMF analysis identified {n_comps} frequency components. Manual inspection recommended for detailed interpretation.",
            "component_interpretations": [
                {
                    "index": c["index"],
                    "spectral_features": "Requires visual inspection",
                    "physical_meaning": "Requires expert interpretation",
                    "spatial_distribution": f"Coverage: {c.get('spatial_coverage', 0)*100:.1f}%",
                    "confidence": "low"
                }
                for c in stats.get("components", [])
            ],
            "scientific_claims": [{
                "claim": f"Microscopy image analysis reveals {n_comps} distinct structural frequency components.",
                "scientific_impact": "Identification of multiple frequency components suggests complex microstructure.",
                "has_anyone_question": f"Has anyone observed multi-component FFT/NMF decomposition patterns in similar microscopy data?",
                "keywords": ["microscopy", "FFT", "NMF", "microstructure"]
            }]
        }
    
    def _convert_claims_to_legacy_format(self, claims: list) -> list:
        """Convert new claims format to legacy format for backward compatibility."""
        legacy_claims = []
        for claim in claims:
            legacy_claims.append({
                "claim": claim.get("claim", ""),
                "scientific_impact": claim.get("scientific_impact", ""),
                "has_anyone_question": claim.get("has_anyone_question", ""),
                "keywords": claim.get("keywords", [])
            })
        return legacy_claims
    
    def _generate_single_image_html(self, state: dict, stats: dict, llm_analysis: dict) -> None:
        """Generate HTML report for single image (optional)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        detailed_analysis = llm_analysis.get("detailed_analysis", "")
        component_interps = llm_analysis.get("component_interpretations", [])
        claims = llm_analysis.get("scientific_claims", [])
        
        html = self._generate_html_header("Single Image Analysis Report", timestamp)
        
        html += """
        <section>
            <h2>1. Scientific Analysis</h2>
            <div class="analysis-content">
                <div class="analysis-subsection">
                    <h3>1.1 Detailed Analysis</h3>
                    <p>{}</p>
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.2 Component Interpretation</h3>
                    {}
                </div>
            </div>
        </section>
""".format(detailed_analysis, self._render_component_interpretations(component_interps))
        
        # Add images if available
        html += self._render_single_image_visualizations(state)
        
        # Add claims
        html += self._render_claims_section(claims)
        
        html += self._generate_html_footer()
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"✅ HTML report saved: {report_path}")
    
    def _render_single_image_visualizations(self, state: dict) -> str:
        """Render visualizations section for single image."""
        html = """
        <section>
            <h2>2. Visualizations</h2>
            <div class="image-grid">
"""
        
        images_added = 0
        
        # Primary image
        if state.get("image_blob"):
            html += self._render_image_card(
                state["image_blob"]["data"],
                "Primary Image",
                "Original microscopy image"
            )
            images_added += 1
        
        # Global FFT
        global_fft = state.get("global_fft_image")
        if global_fft is not None:
            fft_bytes = self._array_to_png_bytes(global_fft)
            if fft_bytes:
                html += self._render_image_card(
                    fft_bytes,
                    "Global FFT",
                    "Frequency content of the full image"
                )
                images_added += 1
        
        # Components and abundances
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        if components is not None:
            for i in range(min(components.shape[0], 4)):
                comp_bytes = self._array_to_png_bytes(components[i])
                if comp_bytes:
                    html += self._render_image_card(
                        comp_bytes,
                        f"Component {i+1}",
                        "NMF frequency pattern"
                    )
                    images_added += 1
        
        if abundances is not None:
            for i in range(min(abundances.shape[0], 4)):
                abun_bytes = self._array_to_png_bytes(abundances[i])
                if abun_bytes:
                    html += self._render_image_card(
                        abun_bytes,
                        f"Abundance {i+1}",
                        "Spatial distribution of component"
                    )
                    images_added += 1
        
        if images_added == 0:
            html += '<p style="color: #7f8c8d; font-style: italic;">No visualizations available.</p>'
        
        html += """
            </div>
        </section>
"""
        return html
    
    # =========================================================================
    # SERIES MODE
    # =========================================================================
    
    def _execute_series_mode(self, state: dict) -> dict:
        """Execute series analysis and report generation."""
        self.logger.info("🔬 Processing series analysis...")
        
        # Compute statistics
        stats = self._compute_series_stats(
            state.get("series_components"),
            state.get("series_abundances")
        )
        
        # Load visualizations
        visualizations = self._load_visualizations()
        
        # Optional feedback loop
        if self.enable_human_feedback:
            state = self._run_feedback_loop(state, stats, visualizations)
            if state.get("report_cancelled"):
                return state
        
        # Get LLM analysis
        llm_analysis = self._get_series_llm_analysis(state, stats, visualizations)
        
        if llm_analysis is None:
            self.logger.warning("LLM analysis failed, using fallback")
            llm_analysis = self._generate_series_fallback(state, stats)
        
        # Generate HTML report
        self._generate_series_html(state, stats, visualizations, llm_analysis)
        
        # Store in state
        state["llm_report_analysis"] = llm_analysis
        state["result_json"] = {
            "detailed_analysis": llm_analysis.get("detailed_analysis", ""),
            "scientific_claims": llm_analysis.get("scientific_claims", [])
        }
        
        self.logger.info("✅ Series analysis complete")
        return state
    
    def _compute_series_stats(self, components: Optional[np.ndarray], 
                               abundances: Optional[np.ndarray]) -> dict:
        """Compute statistics for series analysis."""
        stats = {
            "components": [],
            "correlations": [],
            "has_data": components is not None and abundances is not None
        }
        
        if not stats["has_data"]:
            return stats
        
        n_frames = abundances.shape[0]
        n_comps = abundances.shape[1]
        mean_abundances = abundances.mean(axis=(2, 3))
        
        for i in range(n_comps):
            ts = mean_abundances[:, i]
            
            slope, intercept = np.polyfit(range(len(ts)), ts, 1)
            y_pred = slope * np.arange(len(ts)) + intercept
            ss_res = np.sum((ts - y_pred) ** 2)
            ss_tot = np.sum((ts - np.mean(ts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            pct_change = ((ts[-1] - ts[0]) / ts[0]) * 100 if ts[0] != 0 else 0
            
            period, periodicity_strength = None, 0
            if len(ts) > 4:
                fft_mag = np.abs(np.fft.fft(ts - ts.mean()))
                half_len = len(ts) // 2
                if half_len > 1:
                    dominant_idx = np.argmax(fft_mag[1:half_len]) + 1
                    period = n_frames / dominant_idx if dominant_idx > 0 else None
                    mean_mag = np.mean(fft_mag[1:half_len])
                    periodicity_strength = float(fft_mag[dominant_idx] / mean_mag) if mean_mag > 0 else 0
            
            stats["components"].append({
                "index": i + 1,
                "mean": float(np.mean(ts)),
                "std": float(np.std(ts)),
                "min": float(np.min(ts)),
                "max": float(np.max(ts)),
                "slope": float(slope),
                "r_squared": float(r_squared),
                "pct_change": float(pct_change),
                "periodicity_strength": float(periodicity_strength),
                "period_frames": float(period) if period else None
            })
        
        if n_comps > 1:
            corr_matrix = np.corrcoef(mean_abundances.T)
            for i in range(n_comps):
                for j in range(i + 1, n_comps):
                    stats["correlations"].append({
                        "components": [i + 1, j + 1],
                        "correlation": float(corr_matrix[i, j])
                    })
        
        return stats
    
    def _get_series_llm_analysis(self, state: dict, stats: dict, 
                                  visualizations: list) -> Optional[dict]:
        """Get LLM analysis for series."""
        self.logger.info("🧠 LLM analyzing series...")
        
        params = state.get("batch_params", {})
        
        prompt_parts = [SERIES_ANALYSIS_INSTRUCTIONS]
        
        context = {
            "n_frames": state.get("n_frames", 0),
            "n_components": params.get("n_components", len(stats.get("components", []))),
            "window_size_nm": params.get("window_size_nm", "auto"),
            "component_statistics": stats.get("components", []),
            "correlations": stats.get("correlations", []),
            "system_info": state.get("system_info", {})
        }
        
        prompt_parts.append(f"\n\n## Analysis Context\n```json\n{json.dumps(context, indent=2)}\n```")
        
        prompt_parts.append("\n\n## Visualizations\n")
        for viz in visualizations:
            prompt_parts.append(f"\n### {viz['name']}\n")
            prompt_parts.append({"mime_type": "image/png", "data": viz["data"]})
        
        components = state.get("series_components")
        if components is not None:
            prompt_parts.append("\n\n## NMF Frequency Components\n")
            for i in range(min(components.shape[0], 6)):
                comp_bytes = self._array_to_png_bytes(components[i])
                if comp_bytes:
                    prompt_parts.append(f"\nComponent {i+1}:\n")
                    prompt_parts.append({"mime_type": "image/png", "data": comp_bytes})
        
        prompt_parts.append("\n\nProvide your analysis as a JSON object. Output ONLY the JSON.")
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    contents=prompt_parts,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                
                result, error = self._parse_llm_response(response)
                
                if error:
                    self.logger.warning(f"Parse error (attempt {attempt+1}): {error}")
                    continue
                
                if result and "detailed_analysis" in result:
                    self.logger.info("✅ LLM analysis complete")
                    return result
                    
            except Exception as e:
                self.logger.error(f"LLM error (attempt {attempt+1}): {e}")
        
        return None
    
    def _generate_series_fallback(self, state: dict, stats: dict) -> dict:
        """Generate fallback analysis for series."""
        n_frames = state.get("n_frames", 0)
        n_comps = len(stats.get("components", []))
        
        return {
            "methodology_notes": "Sliding window FFT with NMF decomposition.",
            "detailed_analysis": f"Analysis of {n_frames} frames identified {n_comps} components. Manual inspection recommended.",
            "component_interpretations": [
                {
                    "index": c["index"],
                    "spectral_features": "Requires visual inspection",
                    "physical_meaning": "Requires expert interpretation",
                    "temporal_behavior": f"Change: {c.get('pct_change', 0):+.1f}%",
                    "confidence": "low"
                }
                for c in stats.get("components", [])
            ],
            "temporal_interpretation": "See component statistics for quantitative trends.",
            "visualization_descriptions": [],
            "scientific_claims": [{
                "claim": f"Time-series FFT/NMF analysis of {n_frames} microscopy frames reveals {n_comps} distinct evolving structural components.",
                "scientific_impact": "Temporal decomposition enables tracking of structural dynamics.",
                "has_anyone_question": f"Has anyone performed FFT/NMF decomposition on in-situ microscopy time-series to track structural evolution?",
                "keywords": ["in-situ microscopy", "FFT", "NMF", "time-series", "structural dynamics"]
            }]
        }
    
    def _generate_series_html(self, state: dict, stats: dict, 
                               visualizations: list, llm_analysis: dict) -> None:
        """Generate HTML report for series analysis."""
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        n_components = params.get("n_components", len(stats.get("components", [])))
        window_size_nm = params.get("window_size_nm", "auto")
        window_size_px = params.get("window_size_pixels", "auto")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        methodology = llm_analysis.get("methodology_notes", "")
        detailed_analysis = llm_analysis.get("detailed_analysis", "")
        component_interps = llm_analysis.get("component_interpretations", [])
        temporal_interp = llm_analysis.get("temporal_interpretation", "")
        viz_descriptions = {v.get("name", ""): v.get("description", "") 
                          for v in llm_analysis.get("visualization_descriptions", [])}
        claims = llm_analysis.get("scientific_claims", [])
        
        html = self._generate_html_header("FFT/NMF Series Analysis Report", timestamp)
        
        # Scientific Analysis section
        html += f"""
        <section>
            <h2>1. Scientific Analysis</h2>
            <div class="analysis-content">
                
                <div class="analysis-subsection">
                    <h3>1.1 Methodology</h3>
                    <p>{methodology if methodology else self._default_methodology()}</p>
                    <table class="param-table">
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Frames Analyzed</td><td><strong>{n_frames}</strong></td></tr>
                        <tr><td>NMF Components</td><td><strong>{n_components}</strong></td></tr>
                        <tr><td>Window Size</td><td><strong>{window_size_nm} nm</strong> ({window_size_px} px)</td></tr>
                    </table>
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.2 Detailed Analysis</h3>
                    <p>{detailed_analysis}</p>
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.3 Component Interpretation</h3>
                    {self._render_component_interpretations(component_interps)}
                </div>
                
                <div class="analysis-subsection">
                    <h3>1.4 Temporal Dynamics</h3>
                    <p>{temporal_interp if temporal_interp else "See component analysis for temporal details."}</p>
                </div>
                
            </div>
        </section>
"""
        
        # Visualizations section
        html += """
        <section>
            <h2>2. Visualizations</h2>
            <div class="image-grid">
"""
        
        for viz in visualizations:
            name = viz["name"]
            display_name = name.replace('_', ' ').replace('-', ' ').title()
            description = viz_descriptions.get(name, "Analysis visualization from FFT/NMF processing.")
            html += self._render_image_card(viz["data"], display_name, description)
        
        if not visualizations:
            html += '<p style="color: #7f8c8d; font-style: italic;">No visualizations available.</p>'
        
        html += """
            </div>
        </section>
"""
        
        # Claims section
        html += self._render_claims_section(claims)
        
        html += self._generate_html_footer()
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"✅ Report saved: {report_path}")
        print(f"\n📊 Report: {report_path}")
    
    # =========================================================================
    # HUMAN FEEDBACK LOOP
    # =========================================================================
    
    def _run_feedback_loop(self, state: dict, stats: dict, visualizations: list) -> dict:
        """Human-in-the-loop review before final report generation."""
        self.logger.info("\n\n👤 --- REPORT REVIEW LOOP --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            self._display_results_for_review(state, stats, visualizations, iteration)
            
            feedback = self._collect_human_feedback()
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted results.")
                break
            elif feedback["action"] == "get_assessment":
                assessment = self._get_llm_preliminary_assessment(state, stats, visualizations)
                self._display_llm_assessment(assessment)
            elif feedback["action"] == "cancel":
                self.logger.info("❌ User cancelled.")
                state["report_cancelled"] = True
                return state
        
        return state
    
    def _display_results_for_review(self, state: dict, stats: dict, 
                                     visualizations: list, iteration: int) -> None:
        """Display results for human review."""
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        
        print("\n" + "=" * 80)
        print(f"🔬 ANALYSIS REVIEW - Iteration {iteration}")
        print("=" * 80)
        
        print(f"\n🖼️  Visualizations: {self.output_dir}")
        for viz in visualizations[:5]:
            print(f"   - {viz['name']}.png")
        
        print(f"\n📊 Summary:")
        print(f"   - Frames: {n_frames}")
        print(f"   - Components: {params.get('n_components', len(stats.get('components', [])))}")
        
        print(f"\n📈 Component Statistics:")
        for comp in stats.get("components", []):
            print(f"   C{comp['index']}: mean={comp['mean']:.4f}, change={comp['pct_change']:+.1f}%")
        
        print("-" * 80)
    
    def _collect_human_feedback(self) -> dict:
        """Collect human feedback."""
        print("\n👤 Options:")
        print("   [1] Accept and generate report")
        print("   [2] Get LLM assessment")
        print("   [c] Cancel")
        
        try:
            choice = input("\nChoice [1/2/c]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return {"action": "accept"}
        
        if choice == '1' or choice == '':
            return {"action": "accept"}
        elif choice == '2':
            return {"action": "get_assessment"}
        elif choice == 'c':
            return {"action": "cancel"}
        else:
            return {"action": "accept"}
    
    def _get_llm_preliminary_assessment(self, state: dict, stats: dict, 
                                         visualizations: list) -> dict:
        """Get quick LLM assessment."""
        prompt_parts = [
            "Briefly assess these FFT/NMF results.",
            f"\n\nStatistics:\n```json\n{json.dumps(stats, indent=2)}\n```"
        ]
        
        for viz in visualizations[:3]:
            prompt_parts.append({"mime_type": "image/png", "data": viz["data"]})
        
        prompt_parts.append("""
Return JSON: {"quality": "good/moderate/poor", "observations": ["..."], "issues": ["..."]}""")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, _ = self._parse_llm_response(response)
            return result or {"quality": "unknown", "observations": [], "issues": []}
        except Exception as e:
            return {"quality": "error", "observations": [], "issues": [str(e)]}
    
    def _display_llm_assessment(self, assessment: dict) -> None:
        """Display LLM assessment."""
        print(f"\n🤖 LLM Assessment: {assessment.get('quality', 'N/A')}")
        for obs in assessment.get("observations", []):
            print(f"   • {obs}")
        for issue in assessment.get("issues", []):
            print(f"   ⚠️ {issue}")
        print("-" * 80)
    
    # =========================================================================
    # HTML HELPERS
    # =========================================================================
    
    def _generate_html_header(self, title: str, timestamp: str) -> str:
        """Generate HTML header with CSS."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f9;
        }}
        .container {{
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #9b59b6;
        }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        h2 {{
            color: #8e44ad;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        h3 {{ color: #5b2c6f; margin-top: 25px; margin-bottom: 15px; font-size: 1.1em; }}
        .analysis-content {{
            background-color: #fafafa;
            padding: 25px 30px;
            border-radius: 8px;
            border: 1px solid #eee;
            font-size: 0.95em;
        }}
        .analysis-content p {{ margin-bottom: 15px; text-align: justify; }}
        .analysis-subsection {{
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px dashed #ddd;
        }}
        .analysis-subsection:first-of-type {{ margin-top: 0; padding-top: 0; border-top: none; }}
        .param-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        .param-table th, .param-table td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .param-table th {{ background-color: #f5eef8; color: #5b2c6f; font-weight: 600; }}
        .component-card {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #9b59b6;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .component-card h4 {{ margin: 0 0 15px 0; color: #5b2c6f; display: flex; align-items: center; gap: 10px; }}
        .confidence-badge {{
            font-size: 0.75em;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .confidence-high {{ background-color: #d5f5e3; color: #1e8449; }}
        .confidence-medium {{ background-color: #fef9e7; color: #9a7b0a; }}
        .confidence-low {{ background-color: #fadbd8; color: #922b21; }}
        .component-section {{ margin-bottom: 12px; }}
        .component-section-title {{
            font-weight: 600;
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .component-section p {{ margin: 0; color: #333; }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .image-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-card img {{ width: 100%; height: auto; display: block; }}
        .image-info {{ padding: 15px 20px; border-top: 1px solid #eee; }}
        .image-label {{ font-weight: 600; color: #2c3e50; font-size: 1em; margin-bottom: 8px; }}
        .image-description {{ font-size: 0.9em; color: #666; line-height: 1.6; }}
        .claim-block {{
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .claim-content {{
            background-color: #e8f6f3;
            padding: 20px 25px;
            border-left: 5px solid #1abc9c;
        }}
        .claim-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
        .claim-number {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            background: #1abc9c;
            color: white;
            border-radius: 50%;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .claim-title {{ font-weight: 600; color: #0e6655; font-size: 1em; }}
        .claim-text {{ color: #1a5246; font-size: 0.95em; margin-left: 40px; }}
        .claim-impact {{
            font-size: 0.9em;
            color: #148f77;
            margin-left: 40px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed #a3e4d7;
        }}
        .question-content {{
            background-color: #fef9e7;
            padding: 15px 25px 15px 65px;
            border-left: 5px solid #f39c12;
            position: relative;
        }}
        .question-content::before {{
            content: "↳";
            position: absolute;
            left: 25px;
            top: 15px;
            color: #d4ac0d;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .question-label {{
            font-size: 0.8em;
            color: #9a7b0a;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .question-text {{ color: #7d6608; font-size: 0.95em; }}
        .keywords {{ margin-top: 10px; padding-top: 10px; border-top: 1px dashed #f9e79f; }}
        .keyword-tag {{
            display: inline-block;
            background-color: #fcf3cf;
            color: #7d6608;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.8em;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 {title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
"""
    
    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return """
        <div class="footer">
            Generated by Microscopy Analysis Agent
        </div>
    </div>
</body>
</html>
"""
    
    def _render_image_card(self, image_data: str, title: str, description: str) -> str:
        """Render a single image card."""
        return f"""
                <div class="image-card">
                    <img src="data:image/png;base64,{image_data}" alt="{title}" loading="lazy">
                    <div class="image-info">
                        <div class="image-label">{title}</div>
                        <div class="image-description">{description}</div>
                    </div>
                </div>
"""
    
    def _render_component_interpretations(self, interps: list) -> str:
        """Render component interpretation cards."""
        if not interps:
            return "<p>No component interpretations available.</p>"
        
        html = ""
        for interp in interps:
            idx = interp.get("index", "?")
            confidence = interp.get("confidence", "medium").lower()
            confidence_class = f"confidence-{confidence}" if confidence in ["high", "medium", "low"] else "confidence-medium"
            
            html += f"""
                <div class="component-card">
                    <h4>
                        Component {idx}
                        <span class="confidence-badge {confidence_class}">{confidence}</span>
                    </h4>
"""
            
            for field, title in [
                ("spectral_features", "Spectral Features"),
                ("physical_meaning", "Physical Interpretation"),
                ("temporal_behavior", "Temporal Behavior"),
                ("spatial_distribution", "Spatial Distribution")
            ]:
                value = interp.get(field, "")
                if value:
                    html += f"""
                    <div class="component-section">
                        <div class="component-section-title">{title}</div>
                        <p>{value}</p>
                    </div>
"""
            
            html += "                </div>\n"
        
        return html
    
    def _render_claims_section(self, claims: list) -> str:
        """Render the claims and questions section."""
        html = """
        <section>
            <h2>3. Scientific Claims & Research Questions</h2>
"""
        
        for i, claim_data in enumerate(claims, 1):
            claim = claim_data.get("claim", "")
            impact = claim_data.get("scientific_impact", "")
            question = claim_data.get("has_anyone_question", "")
            keywords = claim_data.get("keywords", [])
            
            keywords_html = ""
            if keywords:
                keywords_html = '<div class="keywords"><strong>Keywords:</strong> '
                keywords_html += " ".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
                keywords_html += "</div>"
            
            html += f"""
            <div class="claim-block">
                <div class="claim-content">
                    <div class="claim-header">
                        <span class="claim-number">{i}</span>
                        <span class="claim-title">Scientific Claim</span>
                    </div>
                    <p class="claim-text">{claim}</p>
                    <p class="claim-impact"><strong>Scientific Impact:</strong> {impact}</p>
                </div>
                <div class="question-content">
                    <div class="question-label">Research Question</div>
                    <p class="question-text">{question}</p>
                    {keywords_html}
                </div>
            </div>
"""
        
        html += "        </section>\n"
        return html
    
    def _default_methodology(self) -> str:
        """Default methodology text."""
        return ("Sliding Window FFT combined with Non-negative Matrix Factorization (NMF) "
                "was used to decompose the microscopy data into frequency-domain components.")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _load_visualizations(self) -> list:
        """Load PNG visualizations from output directory."""
        visualizations = []
        
        for png_path in sorted(self.output_dir.glob("*.png")):
            if png_path.name.startswith("review_iteration"):
                continue
            
            try:
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    visualizations.append({
                        "name": png_path.stem,
                        "path": str(png_path),
                        "data": b64
                    })
            except Exception as e:
                self.logger.warning(f"Failed to load {png_path}: {e}")
        
        return visualizations
    
    def _array_to_png_bytes(self, array: np.ndarray) -> Optional[str]:
        """Convert numpy array to base64 PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(array, cmap='viridis')
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"Array to PNG failed: {e}")
            return None
