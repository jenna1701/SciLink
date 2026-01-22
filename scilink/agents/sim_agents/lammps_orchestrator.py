# scilink/agents/sim_agents/simulation_orchestrator.py

import os
import re
import subprocess
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .lammps_agent import LAMMPSSimulationAgent
from .lammps_updater import LAMMPSUpdater
from .lammps_analysis_agent import LAMMPSAnalysisAgent


class LAMMPSOrchestrator:
    """
    Orchestrates LAMMPS simulations with adaptive quality monitoring.
    
    Does NOT directly use LLMs - delegates to sub-agents.
    """
    
    def __init__(self,
                 working_dir: str,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 lammps_command: str = "lmp",
                 max_stage_attempts: int = 3,
                 stage_timeout: int = 3600):
        """
        Initialize the simulation orchestrator.
        
        Args:
            working_dir: Working directory for simulation
            api_key: API key for LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            lammps_command: Command to run LAMMPS
            max_stage_attempts: Max correction attempts per stage
            stage_timeout: Timeout per stage in seconds
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(exist_ok=True, parents=True)
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required for orchestrator")
        
        self.model_name = model_name
        self.base_url = base_url
        self.lammps_command = lammps_command
        self.max_stage_attempts = max_stage_attempts
        self.stage_timeout = stage_timeout
        
        # Initialize sub-agents (lazy loading) - pass through API config
        self._sim_agent = None
        self._analysis_agent = None
        self._updater = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Tracking
        self.quality_history = []
        self.correction_history = []
        self.stage_results = {}
    
    @property
    def sim_agent(self):
        """Lazy-load simulation agent."""
        if self._sim_agent is None:
            self._sim_agent = LAMMPSSimulationAgent(
                working_dir=str(self.working_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url
            )
        return self._sim_agent
    
    @property
    def analysis_agent(self):
        """Lazy-load analysis agent."""
        if self._analysis_agent is None:
            self._analysis_agent = LAMMPSAnalysisAgent(
                sim_dir=str(self.working_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                package_mode='strict',
                executor_timeout=120
            )
        return self._analysis_agent
    
    @property
    def updater(self):
        """Lazy-load updater agent."""
        if self._updater is None:
            self._updater = LAMMPSUpdater(
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url
            )
        return self._updater
    
    def run_supervised_simulation(self,
                                  data_file: str,
                                  research_goal: str,
                                  system_description: Optional[str] = None,
                                  force_field_files: Optional[Dict[str, str]] = None,
                                  run_final_analysis: bool = True,
                                  **kwargs) -> Dict[str, Any]:
        """
        Run a fully supervised simulation with quality checks and adaptive corrections.
        
        Args:
            data_file: Path to LAMMPS data file
            research_goal: Research objective
            system_description: System description (optional)
            force_field_files: Force field parameter files (optional)
            run_final_analysis: Whether to run comprehensive analysis at end
            **kwargs: Additional parameters for LAMMPSSimulationAgent
            
        Returns:
            Dictionary with complete results:
                - status: "success", "failed", or "partial"
                - stage_results: Results for each stage
                - quality_history: Quality checks performed
                - correction_history: Corrections made
                - final_analysis: Comprehensive analysis (if run_final_analysis=True)
        """
        print(f"\n{'='*80}")
        print(f"🎯 SUPERVISED LAMMPS SIMULATION")
        print(f"{'='*80}")
        print(f"Research Goal: {research_goal}")
        print(f"Working Directory: {self.working_dir}")
        print(f"LAMMPS Command: {self.lammps_command}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # ========================================================================
        # STAGE 1: Generate Staged Simulation
        # ========================================================================
        print(f"📝 STAGE 1: Generating staged simulation")
        print(f"{'─'*80}")
        
        try:
            sim_info = self.sim_agent.generate_staged_simulation(
                data_file=data_file,
                research_goal=research_goal,
                system_description=system_description,
                force_field_files=force_field_files,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to generate simulation: {e}")
            return self._failed_result(f"Simulation generation failed: {e}")
        
        stages = sim_info.get("stages", [])
        stage_scripts = sim_info.get("staged_scripts", {})
        
        if not stages or not stage_scripts:
            # Fallback to single script
            self.logger.warning("No staged scripts generated, using single script")
            stages = ["full_simulation"]
            stage_scripts = {"full_simulation": sim_info["script_path"]}
        
        print(f"✓ Generated {len(stages)} stages: {', '.join(stages)}")
        print(f"{'─'*80}\n")
        
        # ========================================================================
        # STAGE 2: Execute Each Stage with Quality Monitoring
        # ========================================================================
        print(f"🔬 STAGE 2: Executing simulation stages with quality monitoring")
        print(f"{'='*80}\n")
        
        completed_stages = []
        
        for stage_idx, stage_name in enumerate(stages):
            print(f"\n{'─'*80}")
            print(f"🔬 STAGE: {stage_name.upper()} ({stage_idx + 1}/{len(stages)})")
            print(f"{'─'*80}")
            
            stage_script_path = stage_scripts.get(stage_name)
            if not stage_script_path or not os.path.exists(stage_script_path):
                self.logger.error(f"Script not found for stage: {stage_name}")
                return self._partial_result(
                    completed_stages, 
                    f"Script not found for {stage_name}"
                )
            
            # Execute stage with correction loop
            stage_success = False
            
            for attempt in range(1, self.max_stage_attempts + 1):
                print(f"\n  🔄 Attempt {attempt}/{self.max_stage_attempts}")
                
                # ============================================================
                # Step A: Execute LAMMPS
                # ============================================================
                print(f"  ▶  Running LAMMPS for {stage_name}...")
                
                exec_result = self._execute_lammps(stage_script_path)
                
                if exec_result["status"] == "lammps_error":
                    print(f"  ❌ LAMMPS error detected")
                    print(f"     Error: {exec_result.get('error', 'Unknown')[:100]}")
                    
                    # Try to fix LAMMPS error
                    print(f"  🔧 Attempting LAMMPS error correction...")
                    corrected, new_script_path, correction_info = self._fix_lammps_error(
                        stage_script_path,
                        research_goal,
                        sim_info
                    )
                    
                    if corrected:
                        print(f"  ✓  LAMMPS error corrected")
                        stage_script_path = new_script_path
                        self.correction_history.append({
                            "stage": stage_name,
                            "attempt": attempt,
                            "type": "lammps_error",
                            "correction": correction_info
                        })
                        continue  # Retry
                    else:
                        print(f"  ✗  Could not fix LAMMPS error")
                        return self._partial_result(
                            completed_stages,
                            f"Unrecoverable LAMMPS error in {stage_name}"
                        )
                
                # LAMMPS completed successfully
                print(f"  ✓  LAMMPS completed")
                
                # ============================================================
                # Step B: Quality Check
                # ============================================================
                print(f"  🔍 Running quality check...")
                
                quality_result = self.analysis_agent.run_quality_check(
                    research_goal=research_goal,
                    stage=stage_name
                )
                
                self.quality_history.append({
                    "stage": stage_name,
                    "attempt": attempt,
                    "result": quality_result
                })
                
                status = quality_result.get("status", "unknown")
                can_continue = quality_result.get("can_continue", True)
                
                # Print quality summary
                self._print_quality_summary(quality_result)
                
                # ============================================================
                # Step C: Decide Action Based on Quality
                # ============================================================
                
                if status == "healthy":
                    print(f"  ✅ Stage {stage_name} passed - quality is healthy")
                    stage_success = True
                    completed_stages.append(stage_name)
                    self.stage_results[stage_name] = {
                        "status": "success",
                        "attempts": attempt,
                        "quality": quality_result
                    }
                    break  # Move to next stage
                
                elif status == "warning" and can_continue:
                    print(f"  ⚠️  Warnings detected but continuing")
                    print(f"     Issues: {len(quality_result.get('issues', []))}")
                    stage_success = True
                    completed_stages.append(stage_name)
                    self.stage_results[stage_name] = {
                        "status": "warning",
                        "attempts": attempt,
                        "quality": quality_result
                    }
                    break  # Move to next stage
                
                elif status == "critical" or not can_continue:
                    print(f"  ❌ Critical quality issues detected")
                    
                    # Try to fix quality issues
                    if attempt < self.max_stage_attempts:
                        print(f"  🔧 Attempting quality-based correction...")
                        corrected, new_script_path, correction_info = self._fix_quality_issues(
                            stage_script_path,
                            quality_result,
                            research_goal,
                            sim_info,
                            stage_name
                        )
                        
                        if corrected:
                            print(f"  ✓  Script adjusted for quality issues")
                            stage_script_path = new_script_path
                            self.correction_history.append({
                                "stage": stage_name,
                                "attempt": attempt,
                                "type": "quality_issue",
                                "correction": correction_info
                            })
                            continue  # Retry with corrected script
                        else:
                            print(f"  ✗  Could not correct quality issues")
                    
                    # If we've exhausted attempts
                    print(f"  ✗  Failed {stage_name} after {attempt} attempts")
                    return self._partial_result(
                        completed_stages,
                        f"Critical quality issues in {stage_name}"
                    )
                
                else:  # Unknown status
                    print(f"  ❓ Unknown quality status: {status}")
                    if attempt < self.max_stage_attempts:
                        print(f"     Retrying...")
                        continue
                    else:
                        stage_success = True  # Continue with warnings
                        completed_stages.append(stage_name)
                        break
            
            # Check if stage succeeded
            if not stage_success:
                return self._partial_result(
                    completed_stages,
                    f"Failed {stage_name} after {self.max_stage_attempts} attempts"
                )
        
        # ========================================================================
        # STAGE 3: All simulation stages completed successfully
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"✅ ALL SIMULATION STAGES COMPLETED")
        print(f"{'='*80}")
        print(f"Completed stages: {', '.join(completed_stages)}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Quality checks: {len(self.quality_history)}")
        print(f"Corrections made: {len(self.correction_history)}")
        print(f"{'='*80}\n")
        
        # ========================================================================
        # STAGE 4: Final Comprehensive Analysis (optional)
        # ========================================================================
        final_analysis = None
        
        if run_final_analysis:
            print(f"📊 STAGE 3: Running final comprehensive analysis")
            print(f"{'─'*80}")
            
            try:
                final_analysis = self.analysis_agent.run_analysis(research_goal)
                print(f"✓ Final analysis complete")
            except Exception as e:
                self.logger.error(f"Final analysis failed: {e}")
                final_analysis = {
                    "status": "error",
                    "message": f"Final analysis failed: {e}"
                }
        
        # ========================================================================
        # Return Complete Results
        # ========================================================================
        return {
            "status": "success",
            "working_directory": str(self.working_dir),
            "simulation_info": sim_info,
            "completed_stages": completed_stages,
            "stage_results": self.stage_results,
            "quality_history": self.quality_history,
            "correction_history": self.correction_history,
            "total_quality_checks": len(self.quality_history),
            "total_corrections": len(self.correction_history),
            "final_analysis": final_analysis,
            "execution_time": time.time() - start_time
        }
    
    # ============================================================================
    # LAMMPS EXECUTION
    # ============================================================================
    
    def _execute_lammps(self, script_path: str) -> Dict[str, Any]:
        """
        Execute a LAMMPS script.
        
        Args:
            script_path: Path to LAMMPS input script
            
        Returns:
            Execution result with status and any errors
        """
        log_file = self.working_dir / "log.lammps"
        
        # Backup previous log if it exists
        if log_file.exists():
            backup_log = self.working_dir / f"log.lammps.bak{int(time.time())}"
            log_file.rename(backup_log)
        
        try:
            self.logger.info(f"Executing: {self.lammps_command} -in {script_path}")
            
            result = subprocess.run(
                [self.lammps_command, "-in", script_path],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.stage_timeout
            )
            
            # Check for LAMMPS errors
            if result.returncode != 0:
                self.logger.warning(f"LAMMPS exited with code {result.returncode}")
                return {
                    "status": "lammps_error",
                    "error": result.stderr or result.stdout,
                    "returncode": result.returncode
                }
            
            # Check log for ERROR
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    if "ERROR" in log_content:
                        # Extract error message
                        error_lines = [line for line in log_content.split('\n') if 'ERROR' in line]
                        return {
                            "status": "lammps_error",
                            "error": '\n'.join(error_lines[:5]),
                            "returncode": result.returncode
                        }
            
            self.logger.info("LAMMPS execution completed successfully")
            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"LAMMPS execution timed out after {self.stage_timeout}s")
            return {
                "status": "lammps_error",
                "error": f"LAMMPS execution timed out after {self.stage_timeout}s"
            }
        except FileNotFoundError:
            self.logger.error(f"LAMMPS executable not found: {self.lammps_command}")
            return {
                "status": "lammps_error",
                "error": f"LAMMPS executable not found: {self.lammps_command}"
            }
        except Exception as e:
            self.logger.error(f"LAMMPS execution failed: {e}")
            return {
                "status": "lammps_error",
                "error": str(e)
            }
    
    # ============================================================================
    # ERROR CORRECTION
    # ============================================================================
    
    def _fix_lammps_error(self,
                         script_path: str,
                         research_goal: str,
                         sim_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Fix LAMMPS errors using LAMMPSUpdater.
        
        Args:
            script_path: Path to failed script
            research_goal: Research objective
            sim_info: Simulation information
            
        Returns:
            Tuple of (success, new_script_path, correction_info)
        """
        try:
            log_file = self.working_dir / "log.lammps"
            
            if not log_file.exists():
                self.logger.error("No log file found for error analysis")
                return False, script_path, {"error": "No log file"}
            
            # Use updater to refine
            corrected_script, analysis = self.updater.refine_inputs(
                input_path=script_path,
                research_goal=research_goal,
                data_path=sim_info.get("data_path"),
                lammps_log=str(log_file)
            )
            
            # Save corrected script with timestamp
            script_name = Path(script_path).stem
            new_script_path = self.working_dir / f"{script_name}_corrected_{int(time.time())}.lammps"
            
            with open(new_script_path, 'w') as f:
                f.write(corrected_script)
            
            correction_info = {
                "original_script": script_path,
                "corrected_script": str(new_script_path),
                "analysis": analysis,
                "correction_type": "lammps_error"
            }
            
            self.logger.info(f"LAMMPS error corrected: {new_script_path}")
            return True, str(new_script_path), correction_info
            
        except Exception as e:
            self.logger.error(f"Error correction failed: {e}")
            return False, script_path, {"error": str(e)}
    
    def _fix_quality_issues(self,
                           script_path: str,
                           quality_result: Dict[str, Any],
                           research_goal: str,
                           sim_info: Dict[str, Any],
                           stage: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Fix quality issues (not LAMMPS errors) using LAMMPSUpdater.
        
        Args:
            script_path: Path to current script
            quality_result: Quality assessment from LAMMPSAnalysisAgent
            research_goal: Research objective
            sim_info: Simulation information
            stage: Current stage name
            
        Returns:
            Tuple of (success, new_script_path, correction_info)
        """
        try:
            # Use updater's quality refinement method
            corrected_script, correction_analysis = self.updater.refine_for_quality_issues(
                input_path=script_path,
                research_goal=research_goal,
                quality_assessment=quality_result,
                system_info=sim_info.get("system_info", {}),
                stage=stage
            )
            
            # Save corrected script
            script_name = Path(script_path).stem
            new_script_path = self.working_dir / f"{script_name}_quality_fix_{int(time.time())}.lammps"
            
            with open(new_script_path, 'w') as f:
                f.write(corrected_script)
            
            correction_info = {
                "original_script": script_path,
                "corrected_script": str(new_script_path),
                "quality_issues": quality_result.get("issues", []),
                "recommendations_applied": quality_result.get("recommendations", []),
                "correction_type": "quality_issue",
                "analysis": correction_analysis
            }
            
            self.logger.info(f"Quality issues corrected: {new_script_path}")
            return True, str(new_script_path), correction_info
            
        except Exception as e:
            self.logger.error(f"Quality correction failed: {e}")
            return False, script_path, {"error": str(e)}
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _print_quality_summary(self, quality_result: Dict[str, Any]):
        """Print a formatted summary of quality check results."""
        status = quality_result.get("status", "unknown")
        can_continue = quality_result.get("can_continue", True)
        issues = quality_result.get("issues", [])
        
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓"
        }
        
        print(f"  {status_emoji.get(status, '❓')} Quality: {status.upper()}")
        print(f"     Can continue: {'Yes' if can_continue else 'No'}")
        
        if issues:
            print(f"     Issues found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3
                severity = issue.get("severity", "unknown")
                desc = issue.get("description", "No description")
                print(f"       [{severity.upper()}] {desc[:70]}")
            if len(issues) > 3:
                print(f"       ... and {len(issues) - 3} more")
        
        # Print key metrics if available
        metrics = quality_result.get("quality_metrics", {})
        if metrics:
            print(f"     Key metrics:")
            for check_name, check_metrics in list(metrics.items())[:2]:
                for key, value in list(check_metrics.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"       {key}: {value:.4g}")
    
    def _failed_result(self, reason: str) -> Dict[str, Any]:
        """Generate a failed result dictionary."""
        return {
            "status": "failed",
            "reason": reason,
            "working_directory": str(self.working_dir),
            "quality_history": self.quality_history,
            "correction_history": self.correction_history,
            "stage_results": self.stage_results
        }
    
    def _partial_result(self, completed_stages: List[str], reason: str) -> Dict[str, Any]:
        """Generate a partial success result dictionary."""
        return {
            "status": "partial",
            "reason": reason,
            "working_directory": str(self.working_dir),
            "completed_stages": completed_stages,
            "stage_results": self.stage_results,
            "quality_history": self.quality_history,
            "correction_history": self.correction_history
        }
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the orchestrated simulation.
        
        Returns:
            Path to generated report
        """
        report_path = self.working_dir / "simulation_orchestration_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Supervised Simulation Report\n\n")
            f.write(f"**Working Directory:** `{self.working_dir}`\n\n")
            
            # Stage results
            f.write("## Stage Results\n\n")
            for stage_name, result in self.stage_results.items():
                status = result.get("status", "unknown")
                attempts = result.get("attempts", 0)
                
                status_emoji = {"success": "✅", "warning": "⚠️", "failed": "❌"}
                f.write(f"### {status_emoji.get(status, '❓')} {stage_name}\n")
                f.write(f"- Status: {status}\n")
                f.write(f"- Attempts: {attempts}\n\n")
            
            # Quality checks
            f.write("## Quality Checks\n\n")
            f.write(f"Total checks performed: {len(self.quality_history)}\n\n")
            
            for i, check in enumerate(self.quality_history, 1):
                stage = check.get("stage", "unknown")
                attempt = check.get("attempt", 0)
                result = check.get("result", {})
                status = result.get("status", "unknown")
                
                f.write(f"{i}. **{stage}** (attempt {attempt}): {status}\n")
                
                issues = result.get("issues", [])
                if issues:
                    for issue in issues[:3]:
                        f.write(f"   - [{issue.get('severity', '?')}] {issue.get('description', 'No description')}\n")
            
            f.write("\n")
            
            # Corrections
            f.write("## Corrections Made\n\n")
            
            if not self.correction_history:
                f.write("No corrections needed - simulation ran cleanly!\n\n")
            else:
                for i, correction in enumerate(self.correction_history, 1):
                    stage = correction.get("stage", "unknown")
                    attempt = correction.get("attempt", 0)
                    corr_type = correction.get("type", "unknown")
                    
                    f.write(f"{i}. **{stage}** (attempt {attempt})\n")
                    f.write(f"   - Type: {corr_type}\n")
                    
                    if corr_type == "lammps_error":
                        analysis = correction.get("correction", {}).get("analysis", {})
                        issues = analysis.get("issues", [])
                        if issues:
                            f.write(f"   - Issues fixed: {len(issues)}\n")
                            for issue in issues[:2]:
                                f.write(f"     - {issue.get('error_text', 'Unknown')}\n")
                    
                    elif corr_type == "quality_issue":
                        issues = correction.get("correction", {}).get("quality_issues", [])
                        if issues:
                            f.write(f"   - Quality issues addressed: {len(issues)}\n")
                            for issue in issues[:2]:
                                f.write(f"     - {issue.get('description', 'Unknown')}\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Summary report generated: {report_path}")
        return str(report_path)
    
    # ============================================================================
    # CONVENIENCE METHOD FOR SIMPLE USAGE
    # ============================================================================
    
    @classmethod
    def quick_run(cls,
                  data_file: str,
                  research_goal: str,
                  working_dir: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Convenience method for quick supervised simulation runs.
        
        Usage:
            results = SimulationOrchestrator.quick_run(
                data_file="system.data",
                research_goal="Calculate diffusion coefficients"
            )
        
        Args:
            data_file: LAMMPS data file
            research_goal: Research objective
            working_dir: Working directory (auto-generated if not provided)
            **kwargs: Additional parameters
            
        Returns:
            Complete simulation results
        """
        # Auto-generate working dir if not provided
        if working_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            working_dir = f"supervised_sim_{timestamp}"
        
        # Create orchestrator
        orchestrator = cls(working_dir=working_dir, **kwargs)
        
        # Run supervised simulation
        results = orchestrator.run_supervised_simulation(
            data_file=data_file,
            research_goal=research_goal,
            **kwargs
        )
        
        # Generate summary report
        if results.get("status") in ["success", "partial"]:
            report_path = orchestrator.generate_summary_report()
            results["summary_report"] = report_path
        
        return results
