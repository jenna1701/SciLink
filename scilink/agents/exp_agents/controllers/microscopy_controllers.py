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

from ..instruct import FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS
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
    """[📄 Report Step] Generates HTML report with analysis, visualizations, and research questions."""
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        self.logger.info("📄 Generating HTML report...")
        self._generate_report(state)
        return state
    
    def _generate_report(self, state: dict) -> None:
        import base64
        
        params = state.get("batch_params", {})
        n_frames = state.get("n_frames", 0)
        n_components = params.get("n_components", 4)
        
        # Load trends
        trends = {}
        trend_file = self.output_dir / "trends.json"
        if trend_file.exists():
            with open(trend_file) as f:
                trends = json.load(f)
        
        # Embed images
        images = []
        for png in sorted(self.output_dir.glob("*.png")):
            if "review_iteration" in png.name:
                continue
            with open(png, 'rb') as f:
                images.append({
                    "name": png.stem.replace('_', ' ').title(),
                    "data": base64.b64encode(f.read()).decode()
                })
        
        # Generate research questions based on trends
        research_questions = self._generate_research_questions(trends, n_frames, n_components)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Scientific Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .meta {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .analysis {{ background: #f8f9fa; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; }}
        .viz {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .viz img {{ width: 100%; border-radius: 5px; border: 1px solid #ddd; }}
        .viz-caption {{ text-align: center; font-weight: 500; margin-top: 8px; color: #555; }}
        .questions {{ background: #fff3cd; padding: 20px; border-radius: 5px; margin-top: 20px; }}
        .questions h3 {{ color: #856404; margin-top: 0; }}
        .questions ul {{ margin: 0; padding-left: 20px; }}
        .questions li {{ margin: 10px 0; color: #533f03; }}
        .claims {{ background: #d4edda; padding: 20px; border-radius: 5px; margin-top: 20px; }}
        .claims h3 {{ color: #155724; margin-top: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Scientific Analysis Report</h1>
        
        <div class="meta">
            <strong>Date:</strong> {timestamp}<br>
            <strong>Frames Analyzed:</strong> {n_frames}<br>
            <strong>NMF Components:</strong> {n_components}<br>
            <strong>Window Size:</strong> {params.get('window_size_nm', 'auto')} nm
        </div>
        
        <h2>1. Scientific Analysis</h2>
        <div class="analysis">
            <p>Sliding window FFT combined with Non-negative Matrix Factorization (NMF) was used to decompose 
            {n_frames} frames into {n_components} frequency-domain components. Each component represents a 
            distinct periodic pattern, with abundance maps showing spatial distribution over time.</p>
            
            {self._format_trends(trends)}
        </div>
        
        <h2>2. Visualizations</h2>
        <div class="viz">
'''
        
        for img in images:
            html += f'''
            <div>
                <img src="data:image/png;base64,{img['data']}" alt="{img['name']}">
                <div class="viz-caption">{img['name']}</div>
            </div>
'''
        
        html += f'''
        </div>
        
        <h2>3. Research Questions</h2>
        <div class="questions">
            <h3>❓ "Has anyone..." / Follow-up Questions</h3>
            <ul>
'''
        
        for q in research_questions:
            html += f'                <li>{q}</li>\n'
        
        html += '''
            </ul>
        </div>
        
        <div class="claims">
            <h3>📋 Potential Claims from this Analysis</h3>
            <ul>
'''
        
        claims = self._generate_claims(trends, n_frames, n_components)
        for claim in claims:
            html += f'                <li>{claim}</li>\n'
        
        html += '''
            </ul>
        </div>
    </div>
</body>
</html>
'''
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"✅ Report saved: {report_path}")
        print(f"\n📊 Report: {report_path}")
    
    def _format_trends(self, trends) -> str:
        if not trends:
            return "<p>No trend data available.</p>"
        
        # Handle both dict and list formats
        if isinstance(trends, list):
            # Convert list to dict format
            trends_dict = {}
            for i, item in enumerate(trends):
                if isinstance(item, dict):
                    key = item.get('component', item.get('name', f'component_{i+1}'))
                    trends_dict[key] = item
                else:
                    trends_dict[f'component_{i+1}'] = {'value': item}
            trends = trends_dict
        
        if not isinstance(trends, dict):
            return f"<p>Trend data: {trends}</p>"
        
        html = "<p><strong>Observed Trends:</strong></p><ul>"
        for comp, data in trends.items():
            if isinstance(data, dict):
                direction = data.get('trend', data.get('direction', 'stable'))
                slope = data.get('slope', 0)
                mean = data.get('mean', data.get('mean_abundance', 0))
                html += f"<li><strong>{comp.replace('_', ' ').title()}</strong>: {direction} (slope={slope:.4f}, mean={mean:.4f})</li>"
            else:
                html += f"<li><strong>{comp}</strong>: {data}</li>"
        html += "</ul>"
        return html
    
    def _generate_research_questions(self, trends, n_frames: int, n_components: int) -> list:
        questions = [
            f"Has anyone observed similar {n_components}-component decomposition patterns in related materials?",
            "Has anyone correlated FFT/NMF abundance changes with specific physical processes?",
            "Has anyone developed methods to automatically classify these frequency components?",
        ]
        
        if not trends:
            return questions
        
        # Handle both dict and list formats
        if isinstance(trends, list):
            trends_dict = {}
            for i, item in enumerate(trends):
                if isinstance(item, dict):
                    key = item.get('component', item.get('name', f'component_{i+1}'))
                    trends_dict[key] = item
            trends = trends_dict
        
        if isinstance(trends, dict):
            increasing = [k for k, v in trends.items() if isinstance(v, dict) and v.get('trend', v.get('direction', '')) == 'increasing']
            decreasing = [k for k, v in trends.items() if isinstance(v, dict) and v.get('trend', v.get('direction', '')) == 'decreasing']
            
            if increasing:
                questions.append(f"Has anyone investigated the mechanism behind increasing {', '.join(increasing)} abundance?")
            if decreasing:
                questions.append(f"Has anyone observed similar decay patterns in {', '.join(decreasing)}?")
            if increasing and decreasing:
                questions.append("Has anyone studied the correlation between opposing trends in different components?")
        
        return questions
    
    def _generate_claims(self, trends, n_frames: int, n_components: int) -> list:
        claims = [
            f"The sample exhibits {n_components} distinct frequency-domain patterns as revealed by NMF decomposition.",
            f"Temporal analysis over {n_frames} frames reveals dynamic evolution of structural features.",
        ]
        
        if not trends:
            return claims
        
        # Handle both dict and list formats
        if isinstance(trends, list):
            trends_dict = {}
            for i, item in enumerate(trends):
                if isinstance(item, dict):
                    key = item.get('component', item.get('name', f'component_{i+1}'))
                    trends_dict[key] = item
            trends = trends_dict
        
        if isinstance(trends, dict):
            for comp, data in trends.items():
                if isinstance(data, dict):
                    direction = data.get('trend', data.get('direction', 'stable'))
                    claims.append(f"{comp.replace('_', ' ').title()} shows a {direction} trend over the observation period.")
        
        return claims