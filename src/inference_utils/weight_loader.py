# src/inference_utils/weight_loader.py

from __future__ import annotations
import torch
from pathlib import Path
from typing import Tuple, Dict

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes=("model.", "module.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pref in prefixes:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out

def load_state_dict_flex(model: torch.nn.Module, weights_path: str | Path, strict: bool = False) -> Tuple[list, list]:
    """Load a checkpoint 'state_dict' saved from Lightning into a bare SMP model."""
    p = Path(weights_path)
    obj = torch.load(p, map_location="cpu")
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {p}")
    sd = _strip_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected
