"""
Local smoke tests for MLIPAgent assembly + LAMMPS input generation.

These cover the relocation/wiring contract without making any LLM
calls and without invoking real MACE / NequIP / DeePMD backends.

What they exercise:
  - MLIPAgent constructs and auto-loads the `general` skill from the
    machine_learning_potentials bundle directory.
  - _load_backend_skill('mace') merges the backend bundle into
    skill_sections (each section now contains a `--- MACE SPECIFIC ---`
    block).
  - mlip_tools.generate_lammps_input emits a syntactically plausible
    LAMMPS input file for the MACE pair_style, in `metal` units.
"""

import os
import tempfile
from pathlib import Path


def _agent_kwargs():
    return dict(api_key="sk-smoke-not-real", model_name="gpt-4o-mini")


def test_mlip_agent_assembly():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, **_agent_kwargs())

        assert agent.skill_name == "general", (
            f"Expected `general` skill to auto-load, got {agent.skill_name!r}. "
            f"If this fails, the skill loader's domain mapping has drifted."
        )
        assert agent.skill_sections is not None
        for section in ("planning", "validation", "implementation"):
            content = agent.skill_sections.get(section, "")
            assert content, (
                f"`general` skill is missing the {section!r} section -- "
                f"the agent will hand empty context to the LLM."
            )

        ctx = agent._get_skill_context(section="validation")
        assert "MLIP" in ctx or "Energy MAE" in ctx, (
            "validation context did not include the MLIP rules"
        )


def test_mlip_backend_skill_merge():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, skill="mace", **_agent_kwargs())

        impl = agent.skill_sections.get("implementation", "")
        assert "MACE SPECIFIC" in impl, (
            "Backend skill merge marker missing. _load_backend_skill should "
            "append a `--- MACE SPECIFIC ---` block onto each section."
        )
        assert "pair_style" in impl and "mace" in impl, (
            "MACE LAMMPS pair_style guidance missing from merged context"
        )

        planning = agent.skill_sections.get("planning", "")
        assert "mace-mp-0" in planning or "mace-off23" in planning, (
            "MACE foundation-model selection guidance not merged into planning"
        )


def test_mlip_lammps_input_generation():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        path = mlip_tools.generate_lammps_input(
            backend="mace",
            model_file="/path/to/mace-mp-0.model",
            elements=["Cu"],
            working_dir=td,
            timestep=0.5,
            temperature=300.0,
            pressure=None,
        )

        assert Path(path).exists()
        content = Path(path).read_text()

        for required in (
            "units          metal",
            "atom_style     atomic",
            "pair_style     mace no_domain_decomposition",
            "pair_coeff     * * /path/to/mace-mp-0.model Cu",
            "fix 1 all nvt",
            "thermo_style",
            "dump",
        ):
            assert required in content, (
                f"Generated LAMMPS input missing required line: {required!r}\n"
                f"--- content ---\n{content}"
            )

        assert "npt" not in content, (
            "NVT ensemble requested (pressure=None) but NPT slipped in"
        )

        path_npt = mlip_tools.generate_lammps_input(
            backend="mace",
            model_file="/path/to/mace-mp-0.model",
            elements=["Cu"],
            working_dir=td,
            timestep=0.5,
            temperature=300.0,
            pressure=1.0,
        )
        npt_content = Path(path_npt).read_text()
        assert "fix 1 all npt" in npt_content, (
            "pressure=1.0 should produce an NPT fix line"
        )


def test_mlip_unknown_backend_raises():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        try:
            mlip_tools.generate_lammps_input(
                backend="not_a_real_backend",
                model_file="/tmp/x.model",
                elements=["Cu"],
                working_dir=td,
            )
        except ValueError as exc:
            assert "Unknown backend" in str(exc)
        else:
            raise AssertionError("Unknown backend should raise ValueError")


def test_ase_script_generation_nvt():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        path = mlip_tools.generate_ase_script(
            backend="mace",
            model_name="mace-mp-0",
            elements=["Cu"],
            working_dir=td,
            structure_file="cu_bulk.data",
            timestep=1.0,
            temperature=300.0,
            pressure=None,
            n_steps=100,
            output_interval=10,
            device="cuda",
        )

        content = Path(path).read_text()
        assert Path(path).name == "run_md.py"

        for required in (
            "from mace.calculators import mace_mp",
            "mace_mp(",
            "from ase.md.langevin import Langevin",
            "Langevin(",
            "MaxwellBoltzmannDistribution",
            "Trajectory(\"traj.traj\"",
            "thermo.log",
            "dyn.run(100)",
            "read_lammps_data('cu_bulk.data'",
            "ELEMENTS = ['Cu']",
        ):
            assert required in content, (
                f"ASE script missing expected token: {required!r}"
            )

        assert "from ase.md.npt import NPT" not in content, (
            "NVT requested (pressure=None) but NPT import slipped in"
        )


def test_ase_script_generation_npt():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        path = mlip_tools.generate_ase_script(
            backend="mace",
            model_name="mace-mp-0",
            elements=["Cu"],
            working_dir=td,
            timestep=1.0,
            temperature=300.0,
            pressure=1.0,
        )
        content = Path(path).read_text()
        assert "from ase.md.npt import NPT" in content
        assert "externalstress=1.0 * units.bar" in content


def test_ase_script_mace_off_model():
    """mace-off23 should pick the mace_off loader, not mace_mp."""
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        path = mlip_tools.generate_ase_script(
            backend="mace",
            model_name="mace-off23",
            elements=["C", "H", "O"],
            working_dir=td,
        )
        content = Path(path).read_text()
        assert "from mace.calculators import mace_off" in content
        assert "mace_off(" in content
        assert "mace_mp(" not in content, (
            "off23 model should not use mace_mp loader"
        )


def test_ase_runner_unknown_backend():
    from scilink.skills._shared import mlip_tools

    with tempfile.TemporaryDirectory() as td:
        try:
            mlip_tools.generate_ase_script(
                backend="nequip",
                model_name="whatever",
                elements=["Cu"],
                working_dir=td,
            )
        except ValueError as exc:
            assert "mace" in str(exc).lower()
        else:
            raise AssertionError("Non-mace backend should raise ValueError")


def test_agent_runner_validation():
    """Early kwargs validation in deploy_pretrained -- no LLM/MACE needed."""
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, **_agent_kwargs())

        try:
            agent.deploy_pretrained(
                system_info={"elements": {"Cu": 4}, "n_atoms": 4},
                research_goal="test",
                runner="not_a_runner",
            )
        except ValueError as exc:
            assert "runner" in str(exc)
        else:
            raise AssertionError("Invalid runner should raise ValueError")

        try:
            agent.deploy_pretrained(
                system_info={"elements": {"Cu": 4}, "n_atoms": 4},
                research_goal="test",
                runner="ase",
            )
        except ValueError as exc:
            assert "structure_file" in str(exc)
        else:
            raise AssertionError(
                "runner='ase' without structure_file should raise"
            )


if __name__ == "__main__":
    print("=== smoke 1: MLIPAgent assembly + skill load ===")
    test_mlip_agent_assembly()
    print("  OK")
    print()
    print("=== smoke 2: MACE backend skill merge ===")
    test_mlip_backend_skill_merge()
    print("  OK")
    print()
    print("=== smoke 3: MACE LAMMPS input generation ===")
    test_mlip_lammps_input_generation()
    print("  OK")
    print()
    print("=== smoke 4: unknown backend raises ===")
    test_mlip_unknown_backend_raises()
    print("  OK")
    print()
    print("All smokes passed.")
