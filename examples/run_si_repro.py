"""Reproduce yesterday's Si bulk DFT run, CLI-style (no wizard).

Generates POSCAR / INCAR / KPOINTS via DFTOrchestrator and writes a SLURM
submit script alongside them. You then rsync the output dir to the cluster
and sbatch. Same underlying input-generation logic the wizard uses.

Usage (locally, where your LLM API key is set):

    python examples/run_si_repro.py

    # Or override the description:
    python examples/run_si_repro.py --description "diamond Si, 2-atom primitive cell"

    # Custom output dir (default: si_repro_output/)
    python examples/run_si_repro.py --output si_run_2/
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path


def write_slurm_script(work_dir: Path, *, job_name: str = "scilink_si_repro") -> Path:
    """Mirror what the VASP wizard generated yesterday: build POTCAR from
    PSEUDO_DIR, run mpirun vasp_std. The settings here match what the
    wizard's defaults produced — adjust to match your account / partition."""
    submit_path = work_dir / "submit.sh"
    submit_path.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time=00:30:00
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=16
        #SBATCH --output={job_name}_%j.out
        #SBATCH --error={job_name}_%j.err

        set -e
        cd "$SLURM_SUBMIT_DIR"

        # ---- Module environment (edit to match deception) ----
        # module purge
        # module load vasp/6.4.2 intel-mpi/2021

        # ---- Assemble POTCAR from pseudo dir ----
        # EDIT THIS: path to your potpaw_PBE on deception
        PSEUDO_DIR="/path/to/potpaw_PBE.54"

        # Parse element symbols from POSCAR line 6
        ELEMENTS=($(sed -n '6p' POSCAR))
        if [ ${{#ELEMENTS[@]}} -eq 0 ]; then
            echo "No elements parsed from POSCAR; aborting." >&2
            exit 1
        fi
        : > POTCAR
        for e in "${{ELEMENTS[@]}}"; do
            if [ ! -f "$PSEUDO_DIR/$e/POTCAR" ]; then
                echo "Missing POTCAR for element $e at $PSEUDO_DIR/$e/POTCAR" >&2
                exit 1
            fi
            cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
        done

        # ---- Run VASP ----
        mpirun vasp_std
    """))
    submit_path.chmod(0o755)
    return submit_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        default=(
            "diamond Si, 2-atom primitive cell. "
            "Compute the ground-state SCF energy and band gap with PBE."
        ),
        help="Natural-language request passed to DFTOrchestrator.",
    )
    parser.add_argument(
        "--output",
        default="si_repro_output",
        help="Output directory. (Default: si_repro_output/)",
    )
    parser.add_argument(
        "--method",
        choices=["llm", "atomate2"],
        default="llm",
        help="vasp_generator_method for DFTOrchestrator. (Default: llm)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=4,
        help="max_refinement_cycles for DFTOrchestrator. (Default: 4)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model name. Default matches the wizard's setup (claude-opus-4-6); "
             "DFTOrchestrator infers the provider and resolves the matching API key "
             "from env (provider-specific key, or SCILINK_API_KEY as a proxy fallback).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit LLM API key. If unset, DFTOrchestrator auto-discovers "
             "based on --model.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating VASP inputs to: {output_dir}")
    print(f"Description: {args.description}")
    print()

    # Lazy import so --help works without the dependency
    from scilink.agents.sim_agents.dft_orchestrator import DFTOrchestrator

    orchestrator = DFTOrchestrator(
        api_key=args.api_key,
        generator_model=args.model,
        validator_model=args.model,
        output_dir=str(output_dir),
        vasp_generator_method=args.method,
        max_refinement_cycles=args.max_cycles,
    )
    result = orchestrator.run_complete_workflow(args.description)

    # The orchestrator already writes POSCAR/INCAR/KPOINTS into output_dir;
    # surface the most useful bits of the result dict.
    print("\n=== run_complete_workflow result ===")
    for key in ("status", "structure_path", "incar_path", "kpoints_path", "errors"):
        if key in result:
            print(f"  {key}: {result[key]}")

    # Write a SLURM submit template alongside the generated inputs.
    submit_path = write_slurm_script(output_dir)

    print(f"\nWrote SLURM submit script: {submit_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {submit_path}: set PSEUDO_DIR (and uncomment module loads)")
    print(f"  2. rsync -av {output_dir}/ alle927@deception.pnl.gov:/people/alle927/{output_dir.name}/")
    print(f"  3. ssh alle927@deception.pnl.gov 'cd /people/alle927/{output_dir.name} && sbatch submit.sh'")
    print(f"  4. After it finishes, scp the *.out / OUTCAR back for diagnosis")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
