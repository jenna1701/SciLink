"""
Backward-compat alias for the previous VaspInputAgent class.

The VASP-specific agent has been generalized into PeriodicDFTAgent,
which handles VASP / QE / ABINIT / CP2K / ... via skill bundles
under scilink/skills/periodic_dft/<engine>/.

New code should import PeriodicDFTAgent directly:

    from scilink.agents.sim_agents import PeriodicDFTAgent
    agent = PeriodicDFTAgent(...)
    agent.generate_vasp_inputs(poscar_path, request)   # method name unchanged

This shim keeps existing callers working:

    from scilink.agents.sim_agents.vasp_agent import VaspInputAgent
    agent = VaspInputAgent(...)
"""

from .periodic_dft_agent import PeriodicDFTAgent as VaspInputAgent

__all__ = ["VaspInputAgent"]
