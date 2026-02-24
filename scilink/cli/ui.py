"""CLI entry point for launching the SciLink Streamlit UI."""

import subprocess
import sys
from pathlib import Path


def main():
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    app_path = ui_dir / "app.py"
    # Run from the ui/ directory so Streamlit picks up .streamlit/config.toml
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--", *sys.argv[1:]],
        cwd=str(ui_dir),
    )
