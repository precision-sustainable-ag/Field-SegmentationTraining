import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml


class PipelineLogger:
    """
    Collects high‐level metadata for each pipeline run:
      - start/end timestamps
      - duration
      - warnings & errors
      - task name
    Dumps everything to a single `pipeline_log.yaml` in your run directory.
    """

    def __init__(self, run_dir: Path, task: str) -> None:
        self.run_dir = run_dir
        self.task = task
        self.start_time = time.time()
        self.data: Dict[str, Any] = {
            "task": task,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "warnings": [],  # type: List[str]
            "errors": [],    # type: List[Dict[str,Any]]
        }

    def add_warning(self, msg: str) -> None:
        self.data["warnings"].append(msg)

    def add_error(self, exc: Exception) -> None:
        self.data["errors"].append({
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })

    def finalize(self, success: bool) -> None:
        """
        Fill in end_time, duration, status and write YAML.
        """
        self.data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["duration_sec"] = round(time.time() - self.start_time, 2)
        self.data["status"] = "completed" if success else "failed"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "pipeline_log.yaml", "w") as fp:
            yaml.safe_dump(self.data, fp)
