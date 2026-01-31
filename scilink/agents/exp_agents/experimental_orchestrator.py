"""
ExperimentalAnalysisOrchestrator - Interactive chat interface for experimental data analysis.

This orchestrator provides an LLM-powered chat interface for analyzing experimental data.
Users can have conversations about their data, and the orchestrator will select and run
the appropriate analysis agents via tool calls.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from enum import Enum
from datetime import datetime

from .fft_microscopy_agent import FFTMicroscopyAnalysisAgent
from .sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from .hyperspectral_analysis_agent import HyperspectralAnalysisAgent
from .curve_fitting_agent import CurveFittingAgent
from .metadata_converter import generate_metadata_json_from_text
from ._deprecation import normalize_params

# Try to import image processing tools
try:
    from ...tools.image_processor import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
except ImportError:
    try:
        from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
    except ImportError:
        load_image = None
        preprocess_image = None
        convert_numpy_to_jpeg_bytes = None

# Try to import auth utilities
try:
    from ...auth import get_internal_proxy_key
except ImportError:
    def get_internal_proxy_key():
        return os.getenv("SCILINK_API_KEY")

# Try to import LLM wrappers
try:
    from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
    from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
except ImportError:
    OpenAIAsGenerativeModel = None
    LiteLLMGenerativeModel = None


# =============================================================================
# AGENT REGISTRY
# =============================================================================

class AgentType(Enum):
    """Available analysis agent types."""
    FFT_MICROSCOPY = "fft_microscopy"
    SAM_MICROSCOPY = "sam_microscopy"
    HYPERSPECTRAL = "hyperspectral"
    CURVE_FITTING = "curve_fitting"


AGENT_REGISTRY = {
    AgentType.FFT_MICROSCOPY: {
        "class": FFTMicroscopyAnalysisAgent,
        "description": "FFT/NMF-based microscopy analysis for microstructure, phases, and periodic patterns",
        "data_types": ["microscopy", "image"],
        "file_extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    },
    AgentType.SAM_MICROSCOPY: {
        "class": SAMMicroscopyAnalysisAgent,
        "description": "Segment Anything Model for particle/object detection and morphological analysis",
        "data_types": ["microscopy", "image", "particle"],
        "file_extensions": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    },
    AgentType.HYPERSPECTRAL: {
        "class": HyperspectralAnalysisAgent,
        "description": "Hyperspectral/spectroscopic data analysis with NMF unmixing",
        "data_types": ["spectroscopy", "hyperspectral", "eels", "eds"],
        "file_extensions": [".npy"],
    },
    AgentType.CURVE_FITTING: {
        "class": CurveFittingAgent,
        "description": "1D curve fitting for spectra, diffractograms, and time series",
        "data_types": ["spectrum", "curve", "xrd", "pl", "raman", "absorption"],
        "file_extensions": [".csv", ".txt", ".npy", ".xlsx"],
    },
}


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are SciLink's Experimental Analysis Assistant - an expert system for analyzing scientific experimental data.

You help researchers analyze their microscopy images, spectroscopic data, and curve data through an interactive conversation.

**Your Capabilities (via tools):**
1. `analyze_microscopy_fft` - FFT/NMF analysis for periodic structures, domains, phases
2. `analyze_microscopy_sam` - Particle detection and morphological analysis  
3. `analyze_hyperspectral` - Spectroscopic unmixing and mapping (EELS, EDS, etc.)
4. `analyze_curve` - 1D curve fitting (Raman, XRD, PL, absorption)
5. `select_microscopy_agent` - Visually examine an image to choose FFT vs SAM
6. `convert_metadata` - Convert text description to structured metadata
7. `list_available_agents` - Show all available analysis agents

**Workflow:**
1. The user provides a data file path and metadata (as file, text, or you help them create it)
2. If metadata is missing, ask for it or help create it interactively
3. For microscopy images, use `select_microscopy_agent` to visually choose FFT or SAM
4. Run the appropriate analysis tool
5. Discuss the results and suggest follow-up analyses

**Important:**
- Always require metadata before running analysis (experiment type, technique, sample info)
- If the user only provides a file path, ask them to describe their experiment
- Be conversational and helpful - explain what you're doing and why
- After analysis, summarize key findings and offer to explore further

**Data file location:** The user's data files should be in the current working directory or they'll provide the full path.
"""


# =============================================================================
# VISUAL MICROSCOPY AGENT SELECTOR
# =============================================================================

VISUAL_MICROSCOPY_SELECTOR_PROMPT = """You are an expert microscopist selecting between two analysis approaches for a microscopy image.

**Option 1: FFT_MICROSCOPY (FFT/NMF Analysis)**
- Best for: Periodic structures, crystalline lattices, domains, phases, Moiré patterns
- Look for: Regular repeating patterns, lattice fringes, grain structures, oriented domains

**Option 2: SAM_MICROSCOPY (Particle Detection)**
- Best for: Discrete countable objects, particles, pores, cells
- Look for: Isolated objects with clear boundaries, scattered particles, droplets, voids

**Decision Rules:**
1. Distinct, countable objects scattered across the image → SAM_MICROSCOPY
2. Periodic/repeating patterns or continuous textures → FFT_MICROSCOPY
3. Objects with clear boundaries that could be individually measured → SAM_MICROSCOPY
4. Interesting features are spatial frequencies or domains → FFT_MICROSCOPY

**Output:**
Return a JSON object:
{
    "selected_agent": "fft_microscopy" | "sam_microscopy",
    "reasoning": "Brief explanation based on what you see in the image",
    "confidence": "high" | "medium" | "low"
}
"""


class MicroscopyAgentSelector:
    """
    Simple visual selector that chooses between FFT and SAM microscopy agents
    by examining the image content.
    """
    
    def __init__(self, model, logger=None, generation_config=None, safety_settings=None):
        self.model = model
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.generation_config = generation_config
        self.safety_settings = safety_settings
    
    def select(self, image_path: str, metadata: Optional[Dict] = None, analysis_goal: Optional[str] = None) -> tuple:
        """Examine an image and select the appropriate microscopy agent."""
        if self.model is None or load_image is None:
            return AgentType.FFT_MICROSCOPY, {
                "reasoning": "No LLM/image processing available, defaulting to FFT",
                "confidence": "low",
                "method": "fallback"
            }
        
        try:
            image = load_image(image_path)
            if preprocess_image:
                processed, _ = preprocess_image(image, max_dim=512)
            else:
                processed = image
            image_bytes = convert_numpy_to_jpeg_bytes(processed)
            
            prompt_parts = [VISUAL_MICROSCOPY_SELECTOR_PROMPT]
            if metadata or analysis_goal:
                prompt_parts.append("\n--- Context ---")
                if metadata:
                    prompt_parts.append(f"Technique: {metadata.get('experiment', {}).get('technique', 'Unknown')}")
                    prompt_parts.append(f"Material: {metadata.get('sample', {}).get('material', 'Unknown')}")
                if analysis_goal:
                    prompt_parts.append(f"Goal: {analysis_goal}")
            
            prompt_parts.append("\n--- Image ---")
            prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
            
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result = self._parse_response(response)
            if result and result.get("selected_agent") == "sam_microscopy":
                return AgentType.SAM_MICROSCOPY, {
                    "reasoning": result.get("reasoning", ""),
                    "confidence": result.get("confidence", "medium"),
                    "method": "visual_llm"
                }
            return AgentType.FFT_MICROSCOPY, {
                "reasoning": result.get("reasoning", "") if result else "Parse failed",
                "confidence": result.get("confidence", "low") if result else "low",
                "method": "visual_llm"
            }
        except Exception as e:
            self.logger.error(f"Visual selection failed: {e}")
            return AgentType.FFT_MICROSCOPY, {"reasoning": str(e), "confidence": "low", "method": "fallback"}
    
    def _parse_response(self, response) -> Optional[Dict]:
        try:
            raw_text = response.text if hasattr(response, 'text') else str(response)
            first_brace, last_brace = raw_text.find('{'), raw_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                return json.loads(raw_text[first_brace:last_brace + 1])
        except:
            pass
        return None


# =============================================================================
# EXPERIMENTAL ANALYSIS ORCHESTRATOR
# =============================================================================

class ExperimentalAnalysisOrchestrator:
    """
    Interactive chat-based orchestrator for experimental data analysis.
    
    This orchestrator provides an LLM-powered conversational interface where users
    can discuss their data and the system will run appropriate analysis tools.
    
    Args:
        api_key: API key for the LLM provider
        model_name: Model name for the chat LLM
        base_url: Base URL for OpenAI-compatible endpoint
        output_dir: Base directory for analysis outputs
        enable_human_feedback: Enable human-in-the-loop feedback in sub-agents
        
    Example:
        orchestrator = ExperimentalAnalysisOrchestrator(api_key="...")
        
        # Start interactive chat session
        orchestrator.start_chat_session()
        
        # Or process a single message
        response = orchestrator.chat("Analyze my TEM image at sample.tif")
    """
    
    MAX_TOOL_ITERATIONS = 10
    MAX_HISTORY_MESSAGES = 50
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        output_dir: str = "./analysis_outputs",
        enable_human_feedback: bool = False,
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Normalize parameters
        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="ExperimentalAnalysisOrchestrator"
        )
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_human_feedback = enable_human_feedback
        
        # Initialize LLM
        self._initialize_model()
        
        # Initialize visual selector
        self._visual_selector = None
        if self.model is not None:
            self._visual_selector = MicroscopyAgentSelector(
                model=self.model,
                logger=self.logger,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
        
        # Build tool registry
        self._build_tools()
        
        # Agent cache
        self._agent_cache: Dict[AgentType, Any] = {}
        
        # Chat history
        self.history: List[Dict[str, Any]] = []
        
        # Session state
        self.state = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metadata": None,
            "last_analysis": None,
            "analyses_run": [],
        }
        
        self.logger.info(f"ExperimentalAnalysisOrchestrator initialized. Output: {self.output_dir}")
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model."""
        self.model = None
        self.generation_config = None
        self.safety_settings = None
        self.use_openai = False
        
        if self.base_url:
            if self.api_key is None:
                self.api_key = get_internal_proxy_key()
            
            if self.api_key and OpenAIAsGenerativeModel:
                self.logger.info(f"Using OpenAI-compatible model: {self.base_url}")
                self.model = OpenAIAsGenerativeModel(
                    model=self.model_name,
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.use_openai = True
        else:
            if LiteLLMGenerativeModel:
                self.logger.info(f"Using LiteLLM model: {self.model_name}")
                self.model = LiteLLMGenerativeModel(
                    model=self.model_name,
                    api_key=self.api_key
                )
    
    def _build_tools(self) -> None:
        """Build the tool registry with OpenAI-compatible schemas."""
        self.tools_map: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []
        
        # Tool 1: Analyze with FFT Microscopy
        def analyze_microscopy_fft(data_path: str, metadata_json: str) -> str:
            return self._run_analysis(AgentType.FFT_MICROSCOPY, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_microscopy_fft,
            name="analyze_microscopy_fft",
            description="Analyze microscopy image using FFT/NMF for periodic structures, domains, and phases",
            parameters={
                "data_path": {"type": "string", "description": "Path to the microscopy image file"},
                "metadata_json": {"type": "string", "description": "JSON string with experiment metadata"}
            },
            required=["data_path", "metadata_json"]
        )
        
        # Tool 2: Analyze with SAM Microscopy
        def analyze_microscopy_sam(data_path: str, metadata_json: str) -> str:
            return self._run_analysis(AgentType.SAM_MICROSCOPY, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_microscopy_sam,
            name="analyze_microscopy_sam",
            description="Analyze microscopy image using SAM for particle detection and morphological analysis",
            parameters={
                "data_path": {"type": "string", "description": "Path to the microscopy image file"},
                "metadata_json": {"type": "string", "description": "JSON string with experiment metadata"}
            },
            required=["data_path", "metadata_json"]
        )
        
        # Tool 3: Analyze Hyperspectral
        def analyze_hyperspectral(data_path: str, metadata_json: str) -> str:
            return self._run_analysis(AgentType.HYPERSPECTRAL, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_hyperspectral,
            name="analyze_hyperspectral",
            description="Analyze hyperspectral/spectroscopic data (3D datacube) with NMF unmixing",
            parameters={
                "data_path": {"type": "string", "description": "Path to the .npy hyperspectral data file"},
                "metadata_json": {"type": "string", "description": "JSON string with experiment metadata"}
            },
            required=["data_path", "metadata_json"]
        )
        
        # Tool 4: Analyze Curve
        def analyze_curve(data_path: str, metadata_json: str) -> str:
            return self._run_analysis(AgentType.CURVE_FITTING, data_path, metadata_json)
        
        self._register_tool(
            func=analyze_curve,
            name="analyze_curve",
            description="Analyze 1D curve data (Raman, XRD, PL, absorption spectra) with fitting",
            parameters={
                "data_path": {"type": "string", "description": "Path to the curve data file (.csv, .txt, .npy)"},
                "metadata_json": {"type": "string", "description": "JSON string with experiment metadata"}
            },
            required=["data_path", "metadata_json"]
        )
        
        # Tool 5: Visual Microscopy Selection
        def select_microscopy_agent(image_path: str, analysis_goal: Optional[str] = None) -> str:
            if self._visual_selector is None:
                return json.dumps({"error": "Visual selector not available"})
            
            metadata = self.state.get("metadata")
            agent_type, info = self._visual_selector.select(image_path, metadata, analysis_goal)
            return json.dumps({
                "selected_agent": agent_type.value,
                "reasoning": info.get("reasoning"),
                "confidence": info.get("confidence"),
                "recommendation": f"Use analyze_microscopy_{agent_type.value.replace('_microscopy', '')} for this image"
            })
        
        self._register_tool(
            func=select_microscopy_agent,
            name="select_microscopy_agent",
            description="Visually examine a microscopy image to choose between FFT (periodic structures) and SAM (particles) analysis",
            parameters={
                "image_path": {"type": "string", "description": "Path to the microscopy image"},
                "analysis_goal": {"type": "string", "description": "Optional: specific analysis goal"}
            },
            required=["image_path"]
        )
        
        # Tool 6: Convert Metadata
        def convert_metadata(description: str) -> str:
            temp_path = self.output_dir / "temp_metadata_input.txt"
            try:
                with open(temp_path, 'w') as f:
                    f.write(description)
                
                result = generate_metadata_json_from_text(
                    input_text_filepath=str(temp_path),
                    api_key=self.api_key,
                    model_name=self.model_name,
                    base_url=self.base_url
                )
                temp_path.unlink(missing_ok=True)
                
                if result:
                    self.state["metadata"] = result
                    return json.dumps({"status": "success", "metadata": result})
                return json.dumps({"status": "error", "message": "Conversion failed"})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})
        
        self._register_tool(
            func=convert_metadata,
            name="convert_metadata",
            description="Convert a natural language experiment description to structured metadata JSON",
            parameters={
                "description": {"type": "string", "description": "Natural language description of the experiment (technique, material, conditions)"}
            },
            required=["description"]
        )
        
        # Tool 7: List Agents
        def list_available_agents() -> str:
            agents = {
                at.value: {
                    "description": info["description"],
                    "data_types": info["data_types"],
                    "file_extensions": info["file_extensions"]
                }
                for at, info in AGENT_REGISTRY.items()
            }
            return json.dumps({"agents": agents})
        
        self._register_tool(
            func=list_available_agents,
            name="list_available_agents",
            description="List all available analysis agents and their capabilities",
            parameters={},
            required=[]
        )
    
    def _register_tool(self, func: Callable, name: str, description: str, 
                       parameters: Dict, required: List[str]) -> None:
        """Register a tool with its schema."""
        self.tools_map[name] = func
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
        })
    
    def _get_or_create_agent(self, agent_type: AgentType) -> Any:
        """Get or create an analysis agent instance."""
        if agent_type not in self._agent_cache:
            agent_info = AGENT_REGISTRY[agent_type]
            agent_output_dir = self.output_dir / agent_type.value
            agent_output_dir.mkdir(parents=True, exist_ok=True)
            
            self._agent_cache[agent_type] = agent_info["class"](
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                output_dir=str(agent_output_dir),
                enable_human_feedback=self.enable_human_feedback,
            )
        return self._agent_cache[agent_type]
    
    def _run_analysis(self, agent_type: AgentType, data_path: str, metadata_json: str) -> str:
        """Run analysis with the specified agent."""
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "message": "Invalid metadata JSON"})
        
        if not Path(data_path).exists():
            return json.dumps({"status": "error", "message": f"File not found: {data_path}"})
        
        self.logger.info(f"Running {agent_type.value} analysis on {data_path}...")
        
        try:
            agent = self._get_or_create_agent(agent_type)
            result = agent.analyze(data=data_path, system_info=metadata)
            
            # Store in state
            self.state["last_analysis"] = {
                "agent": agent_type.value,
                "data_path": data_path,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            self.state["analyses_run"].append(self.state["last_analysis"])
            
            # Return summary for LLM
            return json.dumps({
                "status": result.get("status", "unknown"),
                "agent": agent_type.value,
                "output_directory": result.get("output_directory"),
                "detailed_analysis": result.get("detailed_analysis", "")[:1500],
                "claims_count": len(result.get("scientific_claims", [])),
                "scientific_claims": result.get("scientific_claims", [])[:3]
            })
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})
    
    def _execute_tool(self, name: str, arguments: Dict) -> str:
        """Execute a tool by name."""
        if name not in self.tools_map:
            return json.dumps({"error": f"Unknown tool: {name}"})
        
        try:
            return self.tools_map[name](**arguments)
        except Exception as e:
            self.logger.error(f"Tool execution error ({name}): {e}")
            return json.dumps({"error": str(e)})
    
    def _trim_history(self) -> None:
        """Trim chat history if it gets too long."""
        if len(self.history) > self.MAX_HISTORY_MESSAGES:
            trimmed = self.history[:1] + self.history[-(self.MAX_HISTORY_MESSAGES - 1):]
            self.history = trimmed
            self.logger.info(f"Trimmed history to {len(self.history)} messages")
    
    def chat(self, user_message: str) -> str:
        """
        Process a single chat message and return the assistant's response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The assistant's response text
        """
        if self.model is None:
            return "Error: No LLM model available. Please check your API key configuration."
        
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Build messages for LLM
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        
        # Chat loop with tool calling
        for iteration in range(self.MAX_TOOL_ITERATIONS):
            try:
                if self.use_openai:
                    response = self._handle_openai_chat(messages)
                else:
                    response = self._handle_gemini_chat(messages)
                
                # Check if we have tool calls
                tool_calls = self._extract_tool_calls(response)
                
                if tool_calls:
                    # Execute tools and add results to messages
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["arguments"]
                        
                        self.logger.info(f"Executing tool: {tool_name}")
                        print(f"  🔧 Running: {tool_name}...")
                        tool_result = self._execute_tool(tool_name, tool_args)
                        
                        # Add tool call and result to messages
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", tool_name),
                            "content": tool_result
                        })
                    
                    # Continue loop to get LLM response to tool results
                    continue
                
                # No tool calls - we have the final response
                assistant_message = self._extract_text(response)
                self.history.append({"role": "assistant", "content": assistant_message})
                self._trim_history()
                
                return assistant_message
                
            except Exception as e:
                self.logger.error(f"Chat error: {e}", exc_info=True)
                return f"I encountered an error: {str(e)}. Please try again."
        
        return "I've reached the maximum number of tool calls. Please try a simpler request."
    
    def _handle_openai_chat(self, messages: List[Dict]) -> Any:
        """Handle chat with OpenAI-compatible API."""
        return self.model.chat_completion(
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto"
        )
    
    def _handle_gemini_chat(self, messages: List[Dict]) -> Any:
        """Handle chat with Gemini API."""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                contents.append(msg["content"])
            elif msg["role"] == "user":
                contents.append(msg["content"])
            elif msg["role"] == "assistant" and msg.get("content"):
                contents.append(msg["content"])
        
        return self.model.generate_content(
            contents=contents,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
    
    def _extract_tool_calls(self, response: Any) -> List[Dict]:
        """Extract tool calls from LLM response."""
        tool_calls = []
        
        # OpenAI format
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        })
        
        # Gemini format
        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            fc = part.function_call
                            tool_calls.append({
                                "id": fc.name,
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {}
                            })
        
        return tool_calls
    
    def _extract_text(self, response: Any) -> str:
        """Extract text content from LLM response."""
        # OpenAI format
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content or ""
        
        # Gemini format
        if hasattr(response, 'text'):
            return response.text
        
        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            return part.text
        
        return str(response)
    
    def start_chat_session(self) -> None:
        """
        Start an interactive chat session in the terminal.
        
        The session continues until the user types 'exit', 'quit', or 'q'.
        """
        print("\n" + "="*60)
        print("🔬 SCILINK EXPERIMENTAL ANALYSIS")
        print("    Interactive Chat Session")
        print("="*60)
        print("\nI'm your experimental analysis assistant. I can help you analyze:")
        print("  • Microscopy images (TEM, STEM, SEM, AFM)")
        print("  • Hyperspectral/spectroscopic data (EELS, EDS)")
        print("  • 1D curves (Raman, XRD, PL, absorption)")
        print("\nTo get started, tell me about your data file and experiment.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Session ended. Results saved to:", self.output_dir)
                    break
                
                print("\n🤔 Processing...\n")
                response = self.chat(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Results saved to:", self.output_dir)
                break
            except EOFError:
                print("\n\n👋 Session ended.")
                break
    
    def reset_session(self) -> None:
        """Reset the chat session (clear history and state)."""
        self.history = []
        self.state = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metadata": None,
            "last_analysis": None,
            "analyses_run": [],
        }
        self.logger.info("Session reset")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.state["session_id"],
            "analyses_run": len(self.state["analyses_run"]),
            "history_length": len(self.history),
            "current_metadata": self.state.get("metadata"),
            "output_directory": str(self.output_dir)
        }
    
    # =========================================================================
    # Convenience method for single-shot analysis (backward compatible)
    # =========================================================================
    
    def analyze(
        self,
        data_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        metadata_path: Optional[str] = None,
        metadata_text: Optional[str] = None,
        analysis_goal: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Single-shot analysis (convenience method).
        
        For interactive analysis, use start_chat_session() or chat() instead.
        """
        # Build a prompt for the chat
        prompt_parts = [f"Please analyze the data file at: {data_path}"]
        
        if metadata:
            self.state["metadata"] = metadata
            prompt_parts.append(f"\nMetadata: {json.dumps(metadata)}")
        elif metadata_path:
            prompt_parts.append(f"\nLoad metadata from: {metadata_path}")
        elif metadata_text:
            prompt_parts.append(f"\nExperiment description: {metadata_text}")
        
        if analysis_goal:
            prompt_parts.append(f"\nAnalysis goal: {analysis_goal}")
        
        if agent_type:
            prompt_parts.append(f"\nUse the {agent_type} agent for this analysis.")
        
        # Run through chat
        response = self.chat(" ".join(prompt_parts))
        
        # Return last analysis result if available
        if self.state.get("last_analysis"):
            return self.state["last_analysis"]["result"]
        
        return {
            "status": "completed",
            "response": response,
            "output_directory": str(self.output_dir)
        }
