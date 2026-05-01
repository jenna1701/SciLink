"""
Tool registry for the SimulationOrchestratorAgent.

Mirrors the shape of AnalysisOrchestratorTools — each tool is a closure
registered via _register_tool with an OpenAI-format JSONSchema. Tools are
dispatched from the chat loop's manual tool-call handler.

Step 1 (skeleton): only a single placeholder `session_status` tool is wired
up so the chat-loop dispatch path can be smoke-tested. Real tools land in
subsequent commits — see the v1 tool surface in CLAUDE.md and the branch's
PR description.
"""

import json
import logging
from typing import Any, Callable, Dict


class SimulationOrchestratorTools:
    """Tool registry + dispatch for SimulationOrchestratorAgent.

    Each tool is registered as a closure so it can capture a reference
    to the parent orchestrator (and therefore its session state).
    """

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent
                SimulationOrchestratorAgent.
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        self._register_all_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_all_tools(self) -> None:
        """Register all tools with OpenAI format. Called once from __init__."""

        # =====================================================================
        # 0. SESSION STATUS  (placeholder — proves dispatch wiring works)
        # =====================================================================
        def session_status() -> str:
            """Report the orchestrator's current session state.

            Useful as a smoke test of the tool-call dispatch path; will
            remain in the registry as a low-cost diagnostic the LLM can
            call when it needs to remember what's been generated.
            """
            structures = self.orch.generated_structures or []
            params = self.orch.default_calc_params or {}
            return json.dumps({
                "status": "ok",
                "session_dir": str(self.orch.base_dir),
                "structures_generated": len(structures),
                "structures": [
                    {
                        "slug": s.get("slug"),
                        "description": s.get("description"),
                        "poscar_path": s.get("poscar_path"),
                    } for s in structures
                ],
                "default_calc_params": params,
                "simulation_mode": self.orch.simulation_mode.value,
            })

        self._register_tool(
            func=session_status,
            name="session_status",
            description=(
                "Report the current simulation session state — structures "
                "generated so far, sticky calculation parameters, output "
                "directory. Free to call; useful when you need to remember "
                "what's already been built before deciding the next step."
            ),
            parameters={},
            required=[],
        )

        # ↓↓↓ Real tools land in subsequent commits (steps 2–6 of the branch).
        # See CLAUDE.md "v1 tool surface" for the full list.

    # ------------------------------------------------------------------
    # Registration + dispatch primitives (mirror analyze-mode shapes)
    # ------------------------------------------------------------------

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None,
    ) -> None:
        """Register a tool in OpenAI format."""
        self.functions_map[name] = func
        self.openai_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        })

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given arguments. Always returns a
        JSON string the chat loop can hand back to the LLM."""
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found",
            })
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })
