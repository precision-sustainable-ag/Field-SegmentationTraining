# src/llm.py

import sys
from pathlib import Path
from typing import Callable, Dict

# Make `src` importable when running this file directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import logging
from omegaconf import DictConfig

# --- Task implementations live in llm_utils ---
from src.llm_utils.utils import run_tabulate_hparams, run_summarize_run

log = logging.getLogger(__name__)

def _build_task_registry() -> Dict[str, Callable[[DictConfig], None]]:
    """
    Map simple task names to callables that accept only (cfg).
    """
    return {
        "tabulate_hparams": run_tabulate_hparams,
        "summarize_run": run_summarize_run,
    }

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Entrypoint for LLM reporting tasks.
    """
    log.info("Starting LLM reporting tasks...")
    
    tasks = _build_task_registry()
    ran_any = False

    # Iterate flags under cfg.tasks.llm and run the enabled ones if present.
    llm_tasks = getattr(cfg.tasks, "llm", {})
    
    for task_name, enabled in llm_tasks.items():
        if bool(enabled) and task_name in tasks:
            log.info(f"[task] llm.{task_name}")
            tasks[task_name](cfg)
            ran_any = True

    if not ran_any:
        log.warning("Nothing to do: no valid tasks enabled in cfg.tasks.llm.")
    else:
        log.info("LLM reporting complete.")

if __name__ == "__main__":
    main()