# src/llm_utils/utils.py

import logging
from pathlib import Path
import ollama
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

def _generate_llm_response(prompt: str, cfg: DictConfig) -> str:
    """Helper to query the local Ollama instance."""
    model_name = getattr(cfg.llm, "model", "granite4")
    options = OmegaConf.to_container(getattr(cfg.llm, "options", {}), resolve=True)
    host = getattr(cfg.llm, "host", "http://localhost:11434")
    
    # Configure ollama client if host is non-default
    client = ollama.Client(host=host)

    log.info(f"Querying Ollama with model: {model_name}...")
    try:
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options=options
        )
        return response.get('response', '')
    except Exception as e:
        log.error(f"Failed to generate LLM response: {e}")
        return f"> **Error generating response:** {e}"

def _save_to_markdown(content: str, cfg: DictConfig, section_title: str):
    """Appends the generated content to the target markdown report."""
    out_path = Path(cfg.llm.output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if out_path.exists() else 'w'
    with open(out_path, mode, encoding="utf-8") as f:
        # If it's a new file, add a header
        if mode == 'w':
            f.write(f"# Pipeline LLM Report - Project: {cfg.project.name}\n\n")
            
        f.write(f"## {section_title}\n\n")
        f.write(content.strip())
        f.write("\n\n---\n\n")
        
    log.info(f"Appended '{section_title}' to {out_path}")

def run_tabulate_hparams(cfg: DictConfig) -> None:
    """Task: Parses train/model configs and generates a markdown table."""
    # Convert relevant config blocks to raw yaml strings
    train_yaml = OmegaConf.to_yaml(cfg.train, resolve=True)
    model_yaml = OmegaConf.to_yaml(cfg.model, resolve=True)
    augment_yaml = OmegaConf.to_yaml(cfg.augment, resolve=True)

    prompt = f"""
You are a highly capable AI assistant embedded in a PyTorch deep learning pipeline.
Your task is to analyze the following YAML configurations for a semantic segmentation experiment and extract the key hyperparameters.

Please generate a clean, professional Markdown table that summarizes the following categories:
1. Model Architecture details
2. Training Hyperparameters (Learning Rate, Optimizer, Scheduler, Batch Size, Epochs)
3. Data Augmentation techniques enabled

Configuration Data:
--- Model ---
{model_yaml}

--- Train ---
{train_yaml}

--- Augment ---
{augment_yaml}

Return ONLY the Markdown table, without conversational filler.
"""
    response = _generate_llm_response(prompt, cfg)
    _save_to_markdown(response, cfg, "Hyperparameter Configuration")

def run_summarize_run(cfg: DictConfig) -> None:
    """Task: Generates a high-level summary of the pipeline's current configuration state."""
    
    prompt = f"""
You are an AI assistant documenting a machine learning pipeline run.
Based on the current environment variables and targets, write a concise 1-2 paragraph summary of what this pipeline is configured to do.

Current Pipeline Mode: {cfg.mode}
Project Name: {cfg.project.name}
Output Directory: {cfg.paths.project_dir}

Highlight the active tasks and any notable hardware settings. Do not invent metrics; simply summarize the intended execution based on the setup.
"""
    response = _generate_llm_response(prompt, cfg)
    _save_to_markdown(response, cfg, "Execution Summary")