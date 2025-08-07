# src/train_utils/data/dataset_utils.py

import re
from pathlib import Path
from typing import List, Union

def natural_base_key(p: Path) -> List[Union[str,int]]:
    """
    1) Take p.stem
    2) If it ends with "_mask", drop that suffix
    3) Split on digit runs, converting digit substrings to int
    """
    stem = p.stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    parts = re.split(r"(\d+)", stem)
    # convert any numeric substring to integer, leave text as-is
    return [int(tok) if tok.isdigit() else tok for tok in parts]
