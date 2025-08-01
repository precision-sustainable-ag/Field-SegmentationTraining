from pathlib import Path
from typing import Dict, Any    
import yaml

def read_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)