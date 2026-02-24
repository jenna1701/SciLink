"""Shared constants and defaults for the SciLink Streamlit UI."""

MODEL_OPTIONS = [
    "claude-opus-4-6",
    "gemini-3-pro-preview",
    "gpt-5.2",
]

SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
)

SUPPORTED_METADATA_EXTENSIONS = (".json", ".txt")

AVATAR_USER = "\U0001f9d1\u200d\U0001f52c"    # scientist
AVATAR_AGENT = "\U0001f916"                    # robot
