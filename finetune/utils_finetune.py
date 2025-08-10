import argparse
import os
import json
from distutils.util import strtobool

def get_cg_args():
    """
    Parses command-line arguments for the Comment Generation fine-tuning script.
    """
    # First, parse the --config argument to load the JSON file
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON config file."
    )
    base_args, remaining_args = base_parser.parse_known_args()

    config = {}
    if base_args.config and os.path.exists(base_args.config):
        with open(base_args.config, "r") as f:
            config = json.load(f)
            print("Loaded configuration from:", base_args.config)

    # Main parser for all arguments
    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument("--model_path", type=str, default=config.get("model_path", "microsoft/codebert-base"), help="Path to the base model.")
    parser.add_argument("--dataset_name", type=str, default=config.get("dataset_name", "code_search_net"), help="Dataset to use.")
    parser.add_argument("--dataset_subset", type=str, default=config.get("dataset_subset", "python"), help="Subset of the dataset.")
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir", "models/PInG/comment_generation_codebert"))

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=config.get("learning_rate", 5e-5), help="The initial learning rate for AdamW.")
    parser.add_argument("--num_epochs", type=int, default=config.get("num_epochs", 3))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 32), help="Per-device batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.get("gradient_accumulation_steps", 1))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=config.get("resume", False))
    parser.add_argument("--fp16", type=lambda x: bool(strtobool(x)), default=config.get("fp16", True), help="Enable mixed-precision training.")
    parser.add_argument("--warmup_ratio", type=float, default=config.get("warmup_ratio", 0.01))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.01))

    # Seq2Seq specific parameters
    parser.add_argument("--max_source_length", type=int, default=config.get("max_source_length", 256), help="Max token length for source (code).")
    parser.add_argument("--max_target_length", type=int, default=config.get("max_target_length", 128), help="Max token length for target (comment).")

    # Parse the remaining arguments (command-line overrides)
    args = parser.parse_args(remaining_args)
    return args

def get_cr_args():
    """
    Parses command-line arguments for the Code Refinement fine-tuning script.
    """
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    base_args, remaining_args = base_parser.parse_known_args()

    config = {}
    if base_args.config and os.path.exists(base_args.config):
        with open(base_args.config, "r") as f:
            config = json.load(f)

    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument("--exp_name", type=str, default=config.get("exp_name", "ping_code_refinement"))
    parser.add_argument("--model_path", type=str, default=config.get("model_path", "deepseek-ai/deepseek-coder-6.7b-base"), help="Path to the base model for fine-tuning.")
    parser.add_argument("--dataset_name", type=str, default=config.get("dataset_name", "bigcode/the-stack"), help="Dataset to use.")
    parser.add_argument("--dataset_subset", type=str, default=config.get("dataset_subset", "data/python"), help="Subset of the dataset.")

    # Training parameters
    parser.add_argument("--output_dir", type=str, default=config.get("output_dir"))
    parser.add_argument("--num_epochs", type=int, default=config.get("num_epochs", 3))
    parser.add_argument("--learning_rate", type=float, default=config.get("learning_rate", 2e-5), help="Learning rate for the optimizer.")
    parser.add_argument("--micro_batch_size", type=int, default=config.get("micro_batch_size", 8))
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.get("gradient_accumulation_steps", 8))
    parser.add_argument("--max_seq_length", type=int, default=config.get("max_seq_length", 2048))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=config.get("resume", False))
    parser.add_argument("--bf16", type=lambda x: bool(strtobool(x)), default=config.get("bf16", True))
    parser.add_argument("--gradient_checkpointing", type=lambda x: bool(strtobool(x)), default=config.get("gradient_checkpointing", True))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.01))
    parser.add_argument("--max_grad_norm", type=float, default=config.get("max_grad_norm", 1.0))
    parser.add_argument("--warmup_ratio", type=float, default=config.get("warmup_ratio", 0.01))
    parser.add_argument("--lr_scheduler_type", type=str, default=config.get("lr_scheduler_type", "cosine"))
    
    args = parser.parse_args(remaining_args)

    if not args.output_dir:
        args.output_dir = f"models/PInG/code_refinement/{args.exp_name}"

    return args





