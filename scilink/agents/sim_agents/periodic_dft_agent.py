# scilink/agents/sim_agents/periodic_dft_agent.py
"""
Periodic-DFT simulation agent. Handles planewave / pseudopotential
DFT codes (VASP, QE, ABINIT, CP2K, ...) via skill bundles.

Software-specific behavior (INCAR vs &control namelist vs ABINIT input,
PAW pseudopotentials vs norm-conserving, etc.) lives in the per-engine
skill bundles at scilink/skills/periodic_dft/<engine>/<engine>.md and
the sibling tools modules. The agent class is scale-aware (periodic
DFT) and software-agnostic.

Today VASP is the only fully wired engine. Adding QE / ABINIT / CP2K
is a sibling skill bundle drop-in.
"""

import os
import re
import json
import logging
from typing import Optional
from ...auth import (
    APIKeyNotFoundError,
    get_api_key,
    get_internal_proxy_key,
    infer_provider,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .instruct import VASP_INPUT_GENERATION_INSTRUCTIONS
from ._deprecation import normalize_params


class PeriodicDFTAgent:
    """Periodic / pseudopotential DFT agent.

    Scale-aware (periodic DFT), software-agnostic. Engine-specific
    behavior lives in skill bundles at
    ``scilink/skills/periodic_dft/<engine>/<engine>.md``.

    Currently supports VASP via the ``vasp`` skill bundle; QE / ABINIT /
    CP2K are extension points (drop in a sibling bundle).
    """

    SKILL_DOMAIN = "periodic_dft"

    SUPPORTED_SOFTWARE = ("vasp",)

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 config: Optional["VASPProjectConfig"] = None,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):
        """
        Initialize PeriodicDFTAgent.

        Parameters
        ----------
        api_key : str, optional
            API key for the LLM provider.
        model_name : str, optional
            Model name to use.
        base_url : str, optional
            Base URL for internal proxy.
        config : VASPProjectConfig, optional
            Project configuration for deterministic parameter enforcement.
            If provided, applied automatically after LLM generation.
        """
        self.logger = logging.getLogger(__name__)
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="PeriodicDFTAgent"
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public path: infer provider from the model name (LiteLLM
            # routes by model prefix, so the key has to match the
            # model's provider). Fall back to SCILINK_API_KEY when no
            # provider-specific key is set in env. Same fix shape as
            # DFTOrchestrator.
            if api_key is None:
                provider = infer_provider(model_name) or "google"
                api_key = get_api_key(provider) or get_internal_proxy_key()
                if not api_key:
                    raise APIKeyNotFoundError(provider)
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None
        self.config = config

    # Backward-compat: previous skill name was "vasp_input_generation" living
    # under the old skills/vasp/ tree. After the periodic-DFT refactor it's
    # just "vasp" under skills/periodic_dft/.
    _LEGACY_SKILL_ALIASES = {"vasp_input_generation": "vasp"}

    def _load_skill(self, skill: str) -> dict:
        """
        Load a periodic-DFT skill bundle (default: ``vasp``).

        Parameters
        ----------
        skill : str
            Skill name (resolved from ``scilink/skills/periodic_dft/``)
            or path to a .md file.

        Returns
        -------
        dict with skill_name and skill_sections, or empty skill state
        on failure.
        """
        resolved = self._LEGACY_SKILL_ALIASES.get(skill, skill)
        try:
            from ...skills.loader import load_skill
            parsed = load_skill(resolved, domain=self.SKILL_DOMAIN)
            self.logger.info(
                f"Loaded {self.SKILL_DOMAIN} skill: {parsed.get('name', resolved)}"
            )
            return {
                "skill_name": parsed.get("name", resolved),
                "skill_sections": parsed,
            }
        except FileNotFoundError:
            self.logger.warning(
                f"Skill '{resolved}' not found under '{self.SKILL_DOMAIN}' — "
                f"proceeding without domain skill"
            )
            return {"skill_name": None, "skill_sections": None}
        except Exception as e:
            self.logger.warning(f"Failed to load skill '{resolved}': {e}")
            return {"skill_name": None, "skill_sections": None}

    def _build_prompt(self, poscar_content: str, original_request: str,
                      skill_sections: Optional[dict] = None) -> str:
        """
        Build the full prompt, injecting skill content if available.

        Parameters
        ----------
        poscar_content : str
            Contents of the POSCAR file.
        original_request : str
            User's description of the calculation.
        skill_sections : dict, optional
            Parsed skill sections from load_skill().
        """
        # Start with the base instructions
        prompt = VASP_INPUT_GENERATION_INSTRUCTIONS.format(
            poscar_content=poscar_content,
            original_request=original_request
        )

        # Inject skill sections into the prompt
        if skill_sections:
            skill_parts = []

            if skill_sections.get("planning"):
                skill_parts.append(
                    "## VASP Best Practices\n" + skill_sections["planning"]
                )

            if skill_sections.get("generation"):
                skill_parts.append(
                    "## VASP Generation Rules\n" + skill_sections["generation"]
                )

            if skill_sections.get("validation"):
                skill_parts.append(
                    "## VASP Validation Criteria\n" + skill_sections["validation"]
                )

            if skill_parts:
                prompt = "\n\n".join(skill_parts) + "\n\n---\n\n" + prompt

        return prompt

    def _parse_response(self, response_text: str) -> dict:
        """
        Robustly parse LLM response, handling common formatting issues.
        """
        text = response_text.strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")

    def _apply_config(self, result: dict,
                      config: Optional["VASPProjectConfig"] = None) -> dict:
        """
        Apply VASPProjectConfig to the generated INCAR.

        Parameters
        ----------
        result : dict
            Generation result with "incar" key.
        config : VASPProjectConfig, optional
            Config to apply. If None, uses self.config.
        """
        active_config = config or self.config
        if active_config is None or result.get("status") != "success":
            return result

        if "incar" not in result:
            return result

        config_result = active_config.apply_to_incar(result["incar"])
        result["incar"] = config_result["incar"]
        result["config_changes"] = config_result["changes"]

        if config_result["changes"]:
            self.logger.info(
                f"Applied {len(config_result['changes'])} config corrections"
            )

        return result

    def generate_vasp_inputs(self, poscar_path: str,
                             original_request: str,
                             skill: Optional[str] = "vasp",
                             config: Optional["VASPProjectConfig"] = None) -> dict:
        """
        Generate VASP INCAR and KPOINTS files.

        Parameters
        ----------
        poscar_path : str
            Path to the POSCAR file.
        original_request : str
            Description of the calculation.
        skill : str, optional
            Domain skill name or path to a .md skill file.
            Default loads 'vasp_input_generation' from scilink/skills/vasp/.
            Set to None to disable.
        config : VASPProjectConfig, optional
            Per-call config override. If not provided, uses the instance config.

        Returns
        -------
        dict with keys:
            status : str ("success" or "error")
            incar : str (INCAR content, with config applied if set)
            kpoints : str (KPOINTS content)
            config_changes : list (changes made by config, if any)
        """
        # Read POSCAR
        try:
            with open(poscar_path, 'r') as f:
                poscar_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read POSCAR: {e}"}

        # Load skill if provided
        skill_sections = None
        if skill:
            skill_state = self._load_skill(skill)
            skill_sections = skill_state.get("skill_sections")

        # Build prompt (with skill content if loaded)
        prompt = self._build_prompt(poscar_content, original_request,
                                    skill_sections=skill_sections)

        # Get LLM response
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = self._parse_response(response.text)
            result["status"] = "success"
        except Exception as e:
            return {"status": "error", "message": f"Generation failed: {e}"}

        # Apply config (per-call override or instance config)
        result = self._apply_config(result, config=config)

        return result

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """Save INCAR and KPOINTS files."""
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}

        os.makedirs(output_dir, exist_ok=True)
        saved = {}

        try:
            # Save INCAR
            incar_path = os.path.join(output_dir, "INCAR")
            with open(incar_path, 'w') as f:
                f.write(result["incar"])
            saved["incar"] = incar_path

            # Save KPOINTS
            kpoints_path = os.path.join(output_dir, "KPOINTS")
            with open(kpoints_path, 'w') as f:
                f.write(result["kpoints"])
            saved["kpoints"] = kpoints_path

            # Save config changes log if any were applied
            if result.get("config_changes"):
                log_path = os.path.join(output_dir, "config_changes.log")
                with open(log_path, 'w') as f:
                    f.write("# Changes applied by VASPProjectConfig\n")
                    for change in result["config_changes"]:
                        f.write(f"  {change}\n")
                saved["config_log"] = log_path

            return saved

        except Exception as e:
            return {"error": f"Save failed: {e}"}

    def apply_improvements(self, original_incar: str, validation_result: dict,
                           poscar_path: str, original_request: str,
                           output_dir: str = ".",
                           skill: Optional[str] = "vasp_input_generation",
                           config: Optional["VASPProjectConfig"] = None) -> dict:
        """
        Regenerate INCAR using LLM with improvement instructions.

        Parameters
        ----------
        original_incar : str
            The original INCAR content to improve.
        validation_result : dict
            Validation result with suggested adjustments.
        poscar_path : str
            Path to the POSCAR file.
        original_request : str
            Original description of the calculation.
        output_dir : str
            Directory to save improved INCAR.
        skill : str, optional
            Domain skill to load. Default: 'vasp_input_generation'.
        config : VASPProjectConfig, optional
            Per-call config override.
        """
        if validation_result.get("validation_status") != "needs_adjustment":
            return {
                "status": "no_changes",
                "message": "No improvements needed - INCAR is already good"
            }

        adjustments = validation_result.get("suggested_adjustments", [])
        if not adjustments:
            return {"status": "error", "message": "No adjustments available"}

        # Read POSCAR content
        try:
            with open(poscar_path, 'r') as f:
                poscar_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read POSCAR: {e}"}

        # Build improvement instructions
        improvement_instructions = "IMPROVEMENT INSTRUCTIONS:\n"
        improvement_instructions += "Please modify the provided INCAR file based on these literature-validated suggestions:\n\n"

        for adj in adjustments:
            improvement_instructions += f"• {adj.get('parameter')}: {adj.get('current_value')} → {adj.get('suggested_value')}\n"
            improvement_instructions += f"  Reason: {adj.get('reason')}\n\n"

        improvement_instructions += f"Literature assessment: {validation_result.get('overall_assessment', '')}\n\n"
        improvement_instructions += "Generate an improved INCAR file incorporating these changes."

        # Load skill
        skill_sections = None
        if skill:
            skill_state = self._load_skill(skill)
            skill_sections = skill_state.get("skill_sections")

        # Build prompt with skill injection
        base_prompt = self._build_prompt(poscar_content, original_request,
                                         skill_sections=skill_sections)

        prompt = f"""{base_prompt}

## ORIGINAL INCAR TO IMPROVE:
{original_incar}

## {improvement_instructions}

Please generate an improved INCAR file based on the improvement instructions above."""

        # Get improved INCAR from LLM
        try:
            response = self.model.generate_content(
                prompt, generation_config=self.generation_config
            )
            result = self._parse_response(response.text)

            if result.get("incar"):
                result["status"] = "success"

                # Apply config
                result = self._apply_config(result, config=config)

                # Save improved INCAR
                os.makedirs(output_dir, exist_ok=True)
                improved_path = os.path.join(output_dir, "INCAR_improved")

                with open(improved_path, 'w') as f:
                    f.write(result["incar"])

                result.update({
                    "improvements_applied": True,
                    "adjustments_count": len(adjustments),
                    "improved_incar_path": improved_path
                })

                self.logger.info(
                    f"Generated improved INCAR with "
                    f"{len(adjustments)} literature-based improvements"
                )
                return result
            else:
                return {"status": "error", "message": "No INCAR generated in LLM response"}

        except Exception as e:
            self.logger.error(f"Failed to generate improved INCAR: {e}")
            return {"status": "error", "message": f"Failed to generate improved INCAR: {e}"}
