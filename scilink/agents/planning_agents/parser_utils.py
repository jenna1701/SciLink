import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import json
import pandas as pd

# Match these to the extensions you check in planning_agent.py
SUPPORTED_EXTENSIONS = {
    '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', 
    '.csv', '.txt', '.md', '.pdf'
}

def get_files_from_directory(directory_path: str) -> List[str]:
    """
    Recursively finds all supported files in a directory, ignoring hidden files.
    """
    found_files = []
    path = Path(directory_path)
    
    if not path.exists():
        print(f"  - ⚠️ Directory not found: {directory_path}")
        return []

    print(f"  - 📂 Scanning directory: {path.name}...")

    for root, dirs, files in os.walk(path):
        # In-place modification to skip hidden dirs and common junk
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'venv', 'env', 'node_modules', '.git')]
        
        for file in files:
            if file.startswith('.'): continue
            
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found_files.append(str(file_path))
                
    print(f"    -> Found {len(found_files)} files in directory.")
    return found_files

def generate_repo_map(root_dir: str) -> str:
    """
    Generates a visual tree structure of the repository.
    Useful for giving the LLM context on where files live for imports.
    """
    root = Path(root_dir)
    if not root.exists(): return ""

    tree_lines = [f"{root.name}/"]
    
    for path in sorted(root.rglob('*')):
        # Skip hidden files/dirs
        if any(part.startswith('.') or part in ('__pycache__', 'venv', 'env') for part in path.parts):
            continue
        
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel_path = path.relative_to(root)
            depth = len(rel_path.parts)
            indent = '    ' * (depth - 1)
            tree_lines.append(f"{indent}├── {path.name}")
            
    return "\n".join(tree_lines)

def table_to_markdown(table: List[List[str]]) -> str:
    """Converts a 2D list representation of a table into Markdown format."""
    if not table or not table[0]: return ""
    # Ensure all cells are strings before joining
    cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    header, *rows = cleaned_table
    md = f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n"
    for row in rows:
        # Pad rows that are shorter than the header
        while len(row) < len(header): row.append("")
        # Truncate rows that are longer than the header
        md += f"| {' | '.join(row[:len(header)])} |\n"
    return md


def parse_json_from_response(resp) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Robustly extracts and parses JSON from an LLM response object.
    Matches the logic originally defined in rag_engine.py.
    """
    json_text = ""
    
    # 1. Extract Text (Protected against Safety Filter blocks)
    try:
        if hasattr(resp, 'text'): 
            json_text = resp.text.strip()
        elif hasattr(resp, 'parts') and resp.parts: 
            json_text = resp.parts[0].text.strip()
        elif isinstance(resp, str):
            json_text = resp.strip()
        else:
            return None, f"LLM response format unexpected: {type(resp)}"
            
    except ValueError as e:
        # Google GenAI raises ValueError on .text access if response was blocked
        return None, f"Response blocked or empty (Safety Filter): {e}"
    except Exception as e:
        return None, f"Error extracting text from response: {e}"

    # 2. Strip Markdown Code Blocks
    if json_text.startswith("```json"):
        json_text = json_text[len("```json"):].strip()
    elif json_text.startswith("```"):
        json_text = json_text[len("```"):].strip()
    
    if json_text.endswith("```"):
        json_text = json_text[:-len("```")].strip()

    # 3. Parse
    try:
        return json.loads(json_text), None
    except json.JSONDecodeError as e:
        return None, f"Failed to decode JSON: {str(e)}"

def append_experiment_result(file_path: str, parameters: Dict[str, float], results: Dict[str, float]):
    """
    Appends a completed experiment (Params + Results) to the cumulative dataset.
    This 'closes the loop' for the BO Agent.
    """
    path = Path(file_path)
    
    # Merge input parameters and lab results into one row
    new_row = {**parameters, **results}
    
    if not path.exists():
        # Create new if doesn't exist
        df = pd.DataFrame([new_row])
    else:
        if path.suffix == '.xlsx':
            df = pd.read_excel(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        # Append
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save back
    if path.suffix == '.xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"✅ Appended result to {path.name}. New size: {len(df)}")