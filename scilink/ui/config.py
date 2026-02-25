"""Shared constants and defaults for the SciLink Streamlit UI."""

MODEL_OPTIONS = [
    "claude-opus-4-6",
    "gemini-3-pro-preview",
    "gpt-5.2",
]

# ── Mode registry ────────────────────────────────────────────────
APP_MODES = [
    {"key": "analyze", "label": "Analyze", "description": "Multi-modal data analysis"},
    {"key": "plan",    "label": "Plan",    "description": "Experimental design & optimization"},
    # {"key": "simulate", "label": "Simulate", "description": "MD/DFT simulations"},
]

SESSION_DIR_PREFIXES = {
    "analyze": "analysis_session",
    "plan": "planning_session",
}

# ── File extensions ──────────────────────────────────────────────
SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
)

SUPPORTED_METADATA_EXTENSIONS = (".json", ".txt")

SUPPORTED_KNOWLEDGE_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".xlsx", ".tsv")
SUPPORTED_CODE_EXTENSIONS = (".py", ".txt", ".md", ".json", ".yaml", ".yml")
SUPPORTED_PLANNING_DATA_EXTENSIONS = (".csv", ".xlsx", ".tsv", ".txt", ".npy")

AVATAR_USER = "\U0001f9d1\u200d\U0001f52c"    # scientist
AVATAR_AGENT = "\U0001f916"                    # robot
