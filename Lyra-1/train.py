#!/usr/bin/env python3
"""Main training script for Lyra-1.

This script handles the training pipeline for the Lyra model,
supporting both single-GPU and multi-GPU training via Accelerate.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__, log_level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a Lyra-1 model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/base_config.yaml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory specified in the config.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Overrides config value if set.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return the training configuration from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_logging(accelerator: Accelerator) -> None:
    """Configure logging for the training run."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # Suppress verbose logs on non-main processes
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    """Entry point for the Lyra-1 training pipeline."""
    args = parse_args()
    config = load_config(args.config)

    # Allow CLI overrides
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        mixed_precision=config.get("mixed_precision", "no"),
        log_with=config.get("report_to", None),
        project_dir=config.get("output_dir", "outputs"),
    )

    setup_logging(accelerator)

    # Default seed changed to 0 for my own experiments so results are easier to track
    seed = config.get("seed", 0)
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    output_dir = Path(config.get("output_dir", "outputs"))
    if accelerator.is_main_proce
