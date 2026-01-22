# scilink/agents/sim_agents/lammps_analysis_agent.py

import os
import re
import ast
import sys
import time
import json
import logging
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...executors import ScriptExecutor
from ._deprecation import normalize_params


class LAMMPSAnalysisAgent:
    """
    Flexible agent for analyzing LAMMPS simulations.
    
    Generates and executes custom Python code based on available 
    simulation files and research goals.
    
    Supports multiple package management modes:
    - 'strict': Only use pre-installed packages
    - 'permissive': Install missing packages at runtime
    - 'dockerfile': Generate custom Dockerfile
    """
    
    PACKAGE_MODES = {
        'strict': 'Only use pre-installed packages (recommended for containers)',
        'permissive': 'Install missing packages at runtime (local development)',
        'dockerfile': 'Generate custom Dockerfile for this analysis'
    }
    
    STANDARD_PACKAGES = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
        'json', 'csv', 'os', 'sys', 're', 'math', 'pathlib',
        'collections', 'itertools', 'functools', 'warnings'
    ]
    
    OPTIONAL_PACKAGES = [
        'MDAnalysis', 'plotly', 'statsmodels', 'sklearn'
    ]
    
    def __init__(self, 
                 sim_dir: str,
                 output_dir: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 executor_timeout: int = 120,
                 enforce_sandbox: bool = True,
                 package_mode: str = 'strict',
                 max_refinement_attempts: int = 2,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the LAMMPS Analysis agent.
        
        Args:
            sim_dir: Directory containing simulation files
            output_dir: Directory for analysis results
            model_name: Model name to use
            api_key: API key for LLM provider
            base_url: Optional base URL for internal proxy
            executor_timeout: Timeout for script execution
            enforce_sandbox: Whether to enforce sandbox restrictions
            package_mode: 'strict', 'permissive', or 'dockerfile'
            max_refinement_attempts: Max attempts to refine failed analyses
            local_model: Deprecated, use base_url
            google_api_key: Deprecated, use api_key
        """
        # Validate paths
        self.sim_dir = Path(sim_dir).resolve()
        if not self.sim_dir.exists():
            raise ValueError(f"Simulation directory does not exist: {sim_dir}")
        
        self.output_dir = Path(output_dir).resolve() if output_dir else self.sim_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Validate package mode
        if package_mode not in self.PACKAGE_MODES:
            raise ValueError(f"package_mode must be one of {list(self.PACKAGE_MODES.keys())}")
        
        self.package_mode = package_mode
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="LAMMPSAnalysisAgent"
        )
        
        # Initialize model
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy")
            
            self.logger.info(f"Using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.logger.info(f"Using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None
        
        # Initialize script executor
        self.executor = ScriptExecutor(
            timeout=executor_timeout,
            enforce_sandbox=enforce_sandbox,
            allow_unsafe_override=False
        )
        
        # Detect container environment
        self.in_container = self._detect_container_environment()
        
        # Storage
        self.input_files = {}
        self.output_files = {}
        self.analysis_code = {}
        self.required_packages = set()
        self.max_refinement_attempts = max_refinement_attempts
        self._sim_details_cache = None
        
        # Log initialization
        self.logger.info(f"LAMMPSAnalysisAgent initialized")
        self.logger.info(f"  Mode: {package_mode}")
        self.logger.info(f"  Container: {self.in_container}")
        self.logger.info(f"  Sim dir: {self.sim_dir}")
        self.logger.info(f"  Output dir: {self.output_dir}")
    
    # ============================================================================
    # HELPER METHODS FOR LLM CALLS
    # ============================================================================
    
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response from LLM."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            raise ValueError(f"Could not parse JSON: {e}")
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text response from LLM."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise
    
    # ============================================================================
    # UPDATE EXISTING METHODS TO USE HELPERS
    # ============================================================================
    
    def _ask_llm_to_interpret_outputs(self, output_commands: Dict[str, Dict[str, Any]], 
                                      script_excerpt: str) -> Dict[str, Dict[str, Any]]:
        """Use LLM to interpret what each output command produces."""
        
        commands_summary = []
        for cmd_id, cmd_info in output_commands.items():
            summary = {
                "id": cmd_id,
                "type": cmd_info["command_type"],
                "filename": cmd_info.get("filename", ""),
                "command": cmd_info.get("context", "")
            }
            
            if "referenced_computes" in cmd_info:
                summary["computes"] = {
                    cid: cinfo.get("type", "") for cid, cinfo in cmd_info["referenced_computes"].items()
                }
            
            commands_summary.append(summary)
        
        prompt = f"""
Analyze LAMMPS output commands and describe what data each produces.

LAMMPS OUTPUT COMMANDS:
{json.dumps(commands_summary, indent=2)}

SCRIPT CONTEXT:
{script_excerpt[:2000]}

For each output command, determine:
1. Physical/computational quantity being output
2. Type of analysis this enables
3. Clear description

Return JSON mapping command ID to data info:
{{
  "command_id": {{
    "data_type": "trajectory|time_series|correlation_function|...",
    "physical_quantity": "positions|energy|temperature|density|...",
    "description": "Clear description",
    "analysis_potential": "What analysis this enables",
    "is_time_series": true/false,
    "dimensionality": "scalar|vector|..."
  }}
}}

Return ONLY JSON.
"""
        
        try:
            return self._generate_json(prompt)  # ✅ Use helper
        except Exception as e:
            self.logger.error(f"LLM interpretation failed: {e}")
            return {}
    
    def _generate_analysis_plan(self, research_goal: str, sim_details: Dict[str, Any],
                             output_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis plan."""
        params = sim_details.get('parameters', {})
        
        example_json = """{
      "analyses": [
        {
          "name": "thermodynamic_analysis",
          "description": "Analyze thermodynamic properties",
          "required_data": ["thermodynamics"],
          "outputs": ["temperature_plot.png", "energy_plot.png"],
          "properties_to_calculate": ["average temperature", "average energy"],
          "importance": "high",
          "connection_to_research_goal": "Validates equilibration"
        }
      ],
      "summary": "Brief summary"
    }"""
        
        prompt = f"""
I need to analyze a LAMMPS simulation. Generate a comprehensive analysis plan.

RESEARCH GOAL:
{research_goal}

SIMULATION DETAILS:
- Type: {sim_details.get('summary', 'Unknown')}
- Ensemble: {params.get('ensemble', 'Unknown')}
- Temperature: {params.get('temperature', 'Unknown')}
- Simulation time: {params.get('simulation_time', 'Unknown')}

AVAILABLE OUTPUT DATA:
{json.dumps([{{"name": name, "description": info["description"], "analysis_potential": info.get("analysis_potential", "")}} for name, info in output_data.items()], indent=2)}

Create analysis plan with this structure:
{example_json}

IMPORTANT:
- Include only analyses with available data
- Be specific about calculations
- Connect to research goal

Return ONLY JSON.
"""
        
        try:
            analysis_plan = self._generate_json(prompt)  # ✅ Use helper
            
            # Filter analyses with available data
            filtered_analyses = []
            for analysis in analysis_plan.get("analyses", []):
                required_data = set(analysis.get("required_data", []))
                available_data = set(output_data.keys())
                
                if required_data.issubset(available_data):
                    filtered_analyses.append(analysis)
                else:
                    missing = required_data - available_data
                    self.logger.warning(f"Skipping '{analysis['name']}' - missing: {missing}")
            
            analysis_plan["analyses"] = filtered_analyses
            return analysis_plan
            
        except Exception as e:
            self.logger.error(f"Error generating analysis plan: {e}")
            return {
                "analyses": [],
                "summary": "Fallback plan due to error"
            }
    
    def _generate_analysis_code(self, analysis: Dict[str, Any],
                             sim_details: Dict[str, Any],
                             output_data: Dict[str, Dict[str, Any]]) -> str:
        """Generate Python code for analysis."""
        # ... [build prompt - unchanged from before] ...
        
        # [Your existing prompt building code]
        
        try:
            code = self._generate_text(prompt)  # ✅ Use helper
            
            # Clean code
            code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            code = code.strip()
            
            # Replace placeholders
            code = code.replace('DATA_FILES_PLACEHOLDER', json.dumps(data_files))
            code = code.replace('OUTPUT_DIR_PLACEHOLDER', str(self.output_dir))
            code = code.replace('"OUTPUT_DIR_PLACEHOLDER"', f'"{self.output_dir}"')
            
            return code
            
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            # Return error-reporting script
            return f"""import json
def main(data_files, output_dir):
    return {{'status': 'error', 'message': 'Failed to generate: {str(e)}'}}
if __name__ == "__main__":
    results = main({{}}, "{self.output_dir}")
    print(json.dumps(results))
"""
    
    def _generate_quality_check_plan(self, 
                                     research_goal: str, 
                                     sim_details: Dict[str, Any],
                                     output_data: Dict[str, Dict[str, Any]], 
                                     stage: str) -> Dict[str, Any]:
        """Generate quality check plan."""
        # ... [build prompt - from earlier] ...
        
        try:
            plan = self._generate_json(prompt)  # ✅ Use helper
            
            # Filter checks with available data
            available_checks = []
            for check in plan.get("checks", []):
                required = set(check.get("required_data", []))
                available = set(output_data.keys())
                
                if required.issubset(available):
                    available_checks.append(check)
            
            plan["checks"] = available_checks
            return plan
            
        except Exception as e:
            self.logger.error(f"Error generating quality plan: {e}")
            return {"checks": [], "critical_thresholds": {}, "stage_notes": ""}
    
    def _synthesize_quality_assessment(self, 
                                       check_results: Dict[str, Any], 
                                       research_goal: str,
                                       stage: str, 
                                       sim_details: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize quality check results into assessment."""
        # ... [build prompt and extract metrics - from earlier] ...
        
        try:
            assessment = self._generate_json(prompt)  # ✅ Use helper
            
            assessment['quality_metrics'] = metrics
            assessment['stage'] = stage
            assessment['failed_checks'] = failed_checks
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error synthesizing assessment: {e}")
            return {
                "status": "unknown",
                "can_continue": True,
                "issues": [],
                "recommendations": ["Manual review recommended"],
                "assessment_summary": "Assessment synthesis failed",
                "next_action": "investigate"
            }
    
    def _generate_final_report(self, research_goal: str, 
                             results: Dict[str, Dict[str, Any]],
                             analysis_plan: Dict[str, Any]) -> str:
        """Generate final HTML report."""
        # ... [build prompt - unchanged] ...
        
        try:
            html_content = self._generate_text(prompt)  # ✅ Use helper
            
            # Clean markdown
            html_content = re.sub(r'```html\s*', '', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'```\s*', '', html_content, flags=re.MULTILINE)
            html_content = html_content.strip()
            
            # Save report
            report_path = self.output_dir / "md_analysis_report.html"
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            # Use fallback
            fallback_html = self._generate_fallback_report(research_goal, results, [])
            report_path = self.output_dir / "md_analysis_report.html"
            with open(report_path, 'w') as f:
                f.write(fallback_html)
            return str(report_path)
    
    def _generate_fallback_report(self, research_goal: str, 
                                 results: Dict[str, Dict[str, Any]],
                                 figures: List[Dict[str, str]]) -> str:
        """Generate a simple fallback HTML report."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <title>MD Analysis Report</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }',
            '        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }',
            '        h2 { color: #3498db; margin-top: 30px; }',
            '        .success { color: #27ae60; font-weight: bold; }',
            '        .error { color: #e74c3c; font-weight: bold; }',
            '        .analysis-section { background: #ecf0f1; padding: 15px; margin: 15px 0; border-radius: 5px; }',
            '        img { max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #bdc3c7; }',
            '        .figure-caption { font-style: italic; color: #7f8c8d; margin-top: -10px; margin-bottom: 20px; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <h1>Molecular Dynamics Analysis Report</h1>',
            '    ',
            '    <div class="analysis-section">',
            '        <h2>Research Goal</h2>',
            f'        <p>{research_goal}</p>',
            '    </div>',
            '    ',
            '    <h2>Analysis Results</h2>'
        ]
        
        for name, data in results.items():
            status_class = 'success' if data.get('status') == 'success' else 'error'
            html_parts.extend([
                '    <div class="analysis-section">',
                f'        <h3>{name.replace("_", " ").title()}</h3>',
                f'        <p>Status: <span class="{status_class}">{data.get("status", "unknown")}</span></p>'
            ])
            
            if data.get('status') == 'success':
                html_parts.append(f'        <p>Execution time: {data.get("execution_time", "N/A")}</p>')
            else:
                html_parts.append(f'        <p>Error: {data.get("message", "Unknown error")}</p>')
            
            html_parts.append('    </div>')
        
        # Add figures
        if figures:
            html_parts.append('\n    <h2>Generated Figures</h2>')
            for fig in figures:
                rel_path = os.path.relpath(fig["path"], self.output_dir)
                html_parts.extend([
                    '    <div>',
                    f'        <img src="{rel_path}" alt="{fig["title"]}">',
                    f'        <p class="figure-caption">{fig["title"]} ({fig["analysis"]})</p>',
                    '    </div>'
                ])
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    # ==================================================================================
    # DOCKERFILE MODE (unchanged)
    # ==================================================================================
    
    def _dockerfile_workflow(self, research_goal: str) -> Dict[str, Any]:
        """
        Generate a custom Dockerfile and provide instructions to user.
        """
        print(f"\n📦 Dockerfile Generation Mode")
        print(f"{'='*60}")
        print("Analyzing requirements to generate custom container...")
        
        # Do the analysis planning steps to understand requirements
        self._inventory_files()
        sim_details = self._analyze_lammps_input()
        if sim_details["status"] != "success":
            return {"status": "error", "message": "Cannot generate Dockerfile without valid simulation"}
        
        output_data = self._identify_output_data()
        analysis_plan = self._generate_analysis_plan(research_goal, sim_details, output_data)
        
        # Generate code for all analyses to understand package requirements
        for analysis in analysis_plan['analyses']:
            code = self._generate_analysis_code(analysis, sim_details, output_data)
            self.analysis_code[analysis['name']] = code
            self._extract_required_packages(code)
        
        # Generate the Dockerfile
        dockerfile_path = self._generate_custom_dockerfile(self.required_packages, research_goal)
        build_script_path = self._generate_build_script(dockerfile_path)
        
        # Provide instructions
        instructions = f"""
    {'='*60}
    📦 Custom Dockerfile Generated
    {'='*60}
    A custom Dockerfile has been generated at:
      {dockerfile_path}
    
    This Dockerfile includes all required packages:
      {', '.join(sorted(self.required_packages))}
    
    TO USE THIS ENVIRONMENT:
    1. Review the Dockerfile (optional)
    2. Build the container:
       sbatch {build_script_path.name}
       
    3. Wait for the build to complete (~20-30 minutes)
    4. Re-run your analysis with package_mode='strict':
       
       agent = MDAnalysisAgent(
           sim_dir="{self.sim_dir}",
           package_mode='strict'
       )
       agent.run_analysis(research_goal)
    {'='*60}
        """
        
        print(instructions)
        
        return {
            "status": "dockerfile_generated",
            "dockerfile_path": str(dockerfile_path),
            "build_script_path": str(build_script_path),
            "required_packages": sorted(list(self.required_packages)),
            "instructions": instructions
        }
    
    def _generate_custom_dockerfile(self, required_packages: Set[str], research_goal: str) -> Path:
        """Generate a custom Dockerfile based on analysis requirements."""
        
        # Build package list for pip install
        pkg_list = ' '.join(sorted(required_packages))
        
        # Create Dockerfile content
        dockerfile_lines = [
            '# Custom Dockerfile for MD Analysis',
            f'# Generated for: {research_goal}',
            f'# Required packages: {", ".join(sorted(required_packages))}',
            '',
            'FROM python:3.12-slim',
            '',
            '# Install system dependencies',
            'RUN apt-get update && apt-get install -y --no-install-recommends \\',
            '    build-essential \\',
            '    gcc \\',
            '    gfortran \\',
            '    libopenblas-dev \\',
            '    liblapack-dev \\',
            '    libgomp1 \\',
            '    git \\',
            '    libgl1 \\',
            '    libglib2.0-0 \\',
            '    && rm -rf /var/lib/apt/lists/*',
            '',
            'WORKDIR /app',
            '',
            '# Upgrade pip',
            'RUN pip install --upgrade pip',
            '',
            '# Install core scientific stack',
            'RUN pip install --no-cache-dir \\',
            '    numpy \\',
            '    scipy \\',
            '    matplotlib \\',
            '    pandas \\',
            '    seaborn',
            '',
            '# Install analysis-specific packages',
            f'RUN pip install --no-cache-dir {pkg_list}' if pkg_list else '# No additional packages needed',
            '',
            '# Install Google Generative AI',
            'RUN pip install --no-cache-dir google-generativeai',
            '',
            '# Copy SciLink',
            'COPY . /app/scilink',
            'WORKDIR /app/scilink',
            'RUN pip install --no-cache-dir -e .',
            '',
            '# Create non-root user',
            'RUN useradd -m -u 1000 scilink && \\',
            '    chown -R scilink:scilink /app',
            '',
            'USER scilink',
            '',
            'ENV RUNNING_IN_CONTAINER=true \\',
            '    PYTHONUNBUFFERED=1 \\',
            '    PYTHONDONTWRITEBYTECODE=1 \\',
            '    MPLBACKEND=Agg',
            '',
            'CMD ["/bin/bash"]'
        ]
        
        dockerfile_content = '\n'.join(dockerfile_lines)
        
        # Save Dockerfile
        dockerfile_path = self.output_dir / "Dockerfile.analysis"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Custom Dockerfile generated: {dockerfile_path}")
        return dockerfile_path
    
    def _generate_build_script(self, dockerfile_path: Path) -> Path:
        """Generate an sbatch script to build the custom container on HPC."""
        
        script_lines = [
            '#!/bin/bash',
            '#SBATCH -A CHANGEME',
            '#SBATCH -t "00:30:00"',
            '#SBATCH -N 1',
            '#SBATCH -p short',
            '#SBATCH -J scilink_analysis_build',
            '#SBATCH -o scilink_analysis_build_%j.out',
            '#SBATCH -e scilink_analysis_build_%j.err',
            '',
            '# Load modules',
            'source /etc/profile.d/modules.sh',
            'module purge',
            'module load apptainer/1.2.4',
            '',
            '# Set up scratch space',
            'export APPTAINER_TMPDIR=/scratch/$USER/APPTAINER',
            'export APPTAINER_CACHEDIR=/scratch/$USER/APPTAINER',
            '',
            'rm -rf $APPTAINER_TMPDIR',
            'rm -rf $APPTAINER_CACHEDIR',
            'mkdir -p $APPTAINER_TMPDIR',
            'mkdir -p $APPTAINER_CACHEDIR',
            '',
            'echo "Converting Dockerfile to Apptainer definition..."',
            f'spython recipe --force --parser docker --writer singularity {dockerfile_path.name} scilink_analysis.def',
            '',
            'echo "Building Apptainer container..."',
            'apptainer build --force --fakeroot scilink_analysis.sif scilink_analysis.def',
            '',
            'echo "✓ Container built successfully: scilink_analysis.sif"',
            'echo "  You can now run your analysis with package_mode=\'strict\'"'
        ]
        
        build_script_content = '\n'.join(script_lines)
        
        script_path = self.output_dir / "build_scilink_analysis.sbatch"
        with open(script_path, 'w') as f:
            f.write(build_script_content)
        
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"Build script generated: {script_path}")
        return script_path
