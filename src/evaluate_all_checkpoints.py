#!/usr/bin/env python3
"""
Evaluate all checkpoints from a specific Weights & Biases run using src/evaluate_checkpoint.py.
Runs both gradient_ascent and random_search for each checkpoint and logs results to CSV.

USAGE EXAMPLES:
==============

1. BASIC USAGE (evaluate all checkpoints with default budgets):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json

2. LIMIT CHECKPOINTS (evaluate only 5 evenly spaced checkpoints):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --max_checkpoints 5 \
     --checkpoint_strategy even

3. CUSTOM BUDGETS (evaluate with custom budget range):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --budget_start 1 \
     --budget_end 50 \
     --budget_period 10

4. QUICK TEST (limit tasks and checkpoints for fast testing):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --only_n_tasks 5 \
     --max_checkpoints 3 \
     --budget_period 50

5. DATASET EVALUATION (use custom dataset instead of JSON):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --dataset_folder pattern2d_eval \
     --dataset_length 100 \
     --max_checkpoints 5

6. METHOD SELECTION (only evaluate and plot specific methods):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --plot_methods gradient_ascent evolutionary_search \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --only_n_tasks 5
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
import subprocess
import wandb
from visualization import visualize_optimization_comparison


def get_all_checkpoints(
    run_name: str,
    project_name: str = "LPN-ARC",
    entity: str = "ga624-imperial-college-london",
) -> List[Dict[str, Any]]:
    """Return all checkpoint artifacts from the specified W&B run."""
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project_name}/{run_name}")
        artifacts = run.logged_artifacts()
        
        checkpoints: List[Dict[str, Any]] = []
        for artifact in artifacts:
            # Only keep artifacts that look like checkpoints
            if "checkpoint" not in artifact.name.lower():
                continue

            # Try to extract a step number from the artifact name
            step_match: Optional[int] = None
            if "--checkpoint" in artifact.name:
                name_part = artifact.name.split("--checkpoint")[0]
                nums = re.findall(r"\d+", name_part)
                if nums:
                    step_match = int(nums[-1])

            # Fallback to alias pattern: num_steps_XXX
            if step_match is None:
                for alias in artifact.aliases:
                    if alias.startswith("num_steps_"):
                        try:
                            step_match = int(alias.split("_")[-1])
                            break
                        except ValueError:
                            pass

            checkpoints.append(
                {
                    "artifact": artifact,
                    "name": artifact.name,      # usually "artifact_name:version"
                    "step": step_match,
                    "aliases": artifact.aliases,
                }
            )

        # Sort by step if available
        checkpoints.sort(key=lambda x: x["step"] if x["step"] is not None else -1)
        
        print(f"Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"  - {cp['name']} (Step: {cp['step']})")
        return checkpoints
        
    except Exception as e:
        print(f"Error accessing run: {e}")
        return []
    

def run_evaluation(
    artifact_path: str,
    method: str,
    method_kwargs: Dict[str, Any],
    json_challenges: Optional[str] = None,
    json_solutions: Optional[str] = None,
    only_n_tasks: Optional[int] = None,
    dataset_folder: Optional[str] = None,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: bool = True,
    dataset_seed: int = 0,
) -> Tuple[bool, Optional[float], Dict[str, Optional[float]], str, float]:
    """Invoke evaluate_checkpoint.py for a specific method and checkpoint."""
    cmd = [sys.executable, "src/evaluate_checkpoint.py", "-w", artifact_path, "-i", method]

    # Choose eval source
    if json_challenges and json_solutions:
        cmd.extend(["-jc", json_challenges, "-js", json_solutions])
        if only_n_tasks is not None:
            cmd.extend(["--only-n-tasks", str(only_n_tasks)])
    elif dataset_folder:
        cmd.extend(["-d", dataset_folder])
        if dataset_length is not None:
            cmd.extend(["--dataset-length", str(dataset_length)])
        if dataset_batch_size is not None:
            cmd.extend(["--dataset-batch-size", str(dataset_batch_size)])
        cmd.extend(["--dataset-use-hf", str(dataset_use_hf).lower()])
        cmd.extend(["--dataset-seed", str(dataset_seed)])
        if only_n_tasks is not None:
            cmd.extend(["--only-n-tasks", str(only_n_tasks)])
    else:
        print("❌ You must provide either JSON files or a dataset folder.")
        return False, None, {}, ""

    # Method-specific args
    if method == "gradient_ascent":
        cmd.extend(
            [
                "--num-steps",
                str(method_kwargs.get("num_steps", 100)),
                "--lr",
                str(method_kwargs.get("lr", 0.1)),
                "--optimizer",
                method_kwargs.get("optimizer", "adam"),
                "--lr-schedule",
                str(method_kwargs.get("lr_schedule", False)).lower(),
                "--lr-schedule-exponent",
                str(method_kwargs.get("lr_schedule_exponent", 0.5)),
            ]
        )
    elif method == "random_search":
        cmd.extend(
            [
                "--num-samples",
                str(method_kwargs.get("num_samples", 100)),
                "--scale",
                str(method_kwargs.get("scale", 1.0)),
                "--scan-batch-size",
                str(method_kwargs.get("scan_batch_size", 10)),
                "--random-search-seed",
                str(method_kwargs.get("random_search_seed", 0)),
            ]
        )
    elif method == "evolutionary_search":
        cmd.extend(
            [
                "--population-size",
                str(method_kwargs.get("population_size", 32)),
                "--num-generations",
                str(method_kwargs.get("num_generations", 25)),
                "--mutation-std",
                str(method_kwargs.get("mutation_std", 0.2)),
            ]
        )
    else:
        print(f"❌ Unknown method: {method}")
        return False, None, {}, ""

    # Avoid creating a W&B run inside evaluate_checkpoint
    cmd.extend(["--no-wandb-run", "true"])

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        import time
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()
        execution_time = end_time - start_time
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Parse metrics from stdout
        metrics: Dict[str, Optional[float]] = {}
        acc: Optional[float] = None
        
        # Parse overall accuracy (case-insensitive)
        try:
            m = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", stdout.lower())
            if m:
                acc = float(m.group(1))
                metrics["overall_accuracy"] = acc
        except Exception:
            acc = None
            metrics["overall_accuracy"] = None

        # Parse additional metrics
        metric_patterns = {
            "top_1_shape_accuracy": r"top_1_shape_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_1_accuracy": r"top_1_accuracy:\s*([0-9]*\.?[0-9]+)", 
            "top_1_pixel_correctness": r"top_1_pixel_correctness:\s*([0-9]*\.?[0-9]+)",
            "top_2_shape_accuracy": r"top_2_shape_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_2_accuracy": r"top_2_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_2_pixel_correctness": r"top_2_pixel_correctness:\s*([0-9]*\.?[0-9]+)",
            # Dataset evaluation metrics
            "correct_shapes": r"correct_shapes:\s*([0-9]*\.?[0-9]+)",
            "pixel_correctness": r"pixel_correctness:\s*([0-9]*\.?[0-9]+)",
        }
        
        for metric_name, pattern in metric_patterns.items():
            try:
                m = re.search(pattern, stdout.lower())
                if m:
                    metrics[metric_name] = float(m.group(1))
                else:
                    metrics[metric_name] = None
            except Exception:
                metrics[metric_name] = None

        if result.returncode == 0:
            print(
                f"✅ {method} evaluation completed successfully"
                + (f" | accuracy={acc}" if acc is not None else "")
                + (f" | shape_acc={metrics.get('top_1_shape_accuracy', 'N/A')}" if metrics.get('top_1_shape_accuracy') is not None else "")
                + (f" | pixel_acc={metrics.get('top_1_pixel_correctness', 'N/A')}" if metrics.get('top_1_pixel_correctness') is not None else "")
                + (f" | correct_shapes={metrics.get('correct_shapes')}" if metrics.get('correct_shapes') is not None else "")
                + (f" | pixel_correctness={metrics.get('pixel_correctness')}" if metrics.get('pixel_correctness') is not None else "")
                + f" | time={execution_time:.2f}s"
            )
            return True, acc, metrics, stdout, execution_time
        else:
            # Retry random_search with smaller scan_batch_size if certain errors show up
            should_retry = (
                (method == "random_search")
                and (
                    ("gpu_fusible" in stderr.lower())
                    or ("fusion root" in stderr.lower())
                    or (result.returncode != 0)
                )
            )
            if should_retry:
                try:
                    current_sbs = int(method_kwargs.get("scan_batch_size", 10) or 10)
                    new_sbs = max(1, min(8, current_sbs // 2 if current_sbs > 2 else 5))
                    retry_cmd = [
                        *cmd,
                        "--scan-batch-size",
                        str(new_sbs),
                    ]
                    print(f"Retrying random_search with --scan-batch-size {new_sbs}...")
                    retry_res = subprocess.run(retry_cmd, capture_output=True, text=True, cwd=os.getcwd())
                    retry_stdout = retry_res.stdout or ""
                    retry_stderr = retry_res.stderr or ""
                    retry_metrics: Dict[str, Optional[float]] = {}
                    retry_acc: Optional[float] = None
                    
                    try:
                        m2 = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", retry_stdout.lower())
                        if m2:
                            retry_acc = float(m2.group(1))
                            retry_metrics["overall_accuracy"] = retry_acc
                    except Exception:
                        retry_acc = None
                        retry_metrics["overall_accuracy"] = None
                    
                    for metric_name, pattern in metric_patterns.items():
                        try:
                            m2 = re.search(pattern, retry_stdout.lower())
                            if m2:
                                retry_metrics[metric_name] = float(m2.group(1))
                            else:
                                retry_metrics[metric_name] = None
                        except Exception:
                            retry_metrics[metric_name] = None
                    
                    if retry_res.returncode == 0:
                        print(
                            f"✅ {method} evaluation (retry) completed successfully"
                            + (f" | accuracy={retry_acc}" if retry_acc is not None else "")
                            + (f" | shape_acc={retry_metrics.get('top_1_shape_accuracy', 'N/A')}" if retry_metrics.get('top_1_shape_accuracy') is not None else "")
                            + (f" | pixel_acc={retry_metrics.get('top_1_pixel_correctness', 'N/A')}" if retry_metrics.get('top_1_pixel_correctness') is not None else "")
                            + (f" | correct_shapes={retry_metrics.get('correct_shapes')}" if retry_metrics.get('correct_shapes') is not None else "")
                            + (f" | pixel_correctness={retry_metrics.get('pixel_correctness')}" if retry_metrics.get('pixel_correctness') is not None else "")
                        )
                        return True, retry_acc, retry_metrics, retry_stdout, execution_time
                    else:
                        print(f"❌ {method} evaluation failed with return code {result.returncode}")
                        if stderr.strip():
                            print(f"Error output:\n{stderr}")
                        if retry_stderr.strip():
                            print(f"Retry error output:\n{retry_stderr}")
                        return False, acc, metrics, stdout, execution_time
                except Exception:
                    print(f"❌ {method} evaluation failed with return code {result.returncode}")
                    if stderr.strip():
                        print(f"Error output:\n{stderr}")
                    return False, acc, metrics, stdout, execution_time
            else:
                print(f"❌ {method} evaluation failed with return code {result.returncode}")
                if stderr.strip():
                    print(f"Error output:\n{stderr}")
                return False, acc, metrics, stdout, execution_time
            
    except Exception as e:
        print(f"❌ Error running {method} evaluation: {e}")
        return False, None, {}, "", 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints from a W&B run")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the W&B run")
    parser.add_argument("--json_challenges", type=str, default=None, help="Path to JSON challenges")
    parser.add_argument("--json_solutions", type=str, default=None, help="Path to JSON solutions")
    parser.add_argument("--only_n_tasks", type=int, default=None, help="Limit number of tasks evaluated")
    # Dataset evaluation
    parser.add_argument(
        "-d",
        "--dataset_folder",
        type=str,
        default=None,
        help="Dataset folder under 'src/datasets' (e.g., 'pattern2d_eval')",
    )
    parser.add_argument("--dataset_length", type=int, default=None, help="Max examples to eval")
    parser.add_argument("--dataset_batch_size", type=int, default=None, help="Batch size for dataset eval")
    parser.add_argument("--dataset_use_hf", type=str, default="true", help="Use HF hub (true/false)")
    parser.add_argument("--dataset_seed", type=int, default=0, help="Seed for dataset subsampling")
    parser.add_argument("--project", type=str, default="LPN-ARC", help="W&B project name")
    parser.add_argument("--entity", type=str, default="ga624-imperial-college-london", help="W&B entity")
    parser.add_argument("--use_all_gpus", action="store_true", 
                   help="Use all available GPUs instead of just one")
    parser.add_argument("--gpu_ids", type=str, default=None,
                   help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--batch_size", type=int, default=1, 
                   help="Batch size for evaluation (larger = faster but more memory)")
    parser.add_argument("--parallel_tasks", type=int, default=1, 
                   help="Number of tasks to process in parallel")
    
    # Checkpoint selection options
    parser.add_argument("--max_checkpoints", type=int, default=None,
                       help="Maximum number of checkpoints to evaluate (default: all)")
    parser.add_argument("--checkpoint_strategy", type=str, default="even", 
                       choices=["even", "first", "last", "random"],
                       help="Strategy for selecting checkpoints: 'even'=evenly spaced, 'first'=first N, 'last'=last N, 'random'=random N (default: even)")
    
    # Method selection for plotting
    parser.add_argument("--plot_methods", type=str, nargs="+", 
                       choices=["gradient_ascent", "random_search", "evolutionary_search"],
                       default=["gradient_ascent", "random_search"],
                       help="Methods to include in plots (default: gradient_ascent, random_search)")
    
    # Budget configuration options
    parser.add_argument("--budget_start", type=int, default=1, 
                       help="Starting budget value (default: 1)")
    parser.add_argument("--budget_end", type=int, default=100, 
                       help="Ending budget value (default: 100)")
    parser.add_argument("--budget_period", type=int, default=25, 
                       help="Period between budget values (default: 25)")
    
    args = parser.parse_args()
    
    # Shared budget configuration
    BUDGET_CONFIG = {
        "start": args.budget_start,           # Start value (inclusive)
        "end": args.budget_end,               # End value (inclusive) 
        "period": args.budget_period,         # Step size between values
        "include_start": True,                # Whether to include the start value
    }
    
    # Generate budgets based on configuration
    def generate_budgets(config):
        budgets = []
        if config["include_start"]:
            budgets.append(config["start"])
        current = config["start"]
        while current <= config["end"]:
            if current not in budgets:
                budgets.append(current)
            current += config["period"]
        return sorted(budgets)
    
    # Generate shared budgets
    shared_budgets = generate_budgets(BUDGET_CONFIG)
    
    # Use the same budgets for all methods
    ga_steps = shared_budgets      # Gradient ascent uses num_steps
    rs_samples = shared_budgets    # Random search uses num_samples
    es_generations = shared_budgets # Evolutionary search uses num_generations
    
    print(f"📊 Using shared budgets: {shared_budgets}")
    print(f"   - Start: {BUDGET_CONFIG['start']}")
    print(f"   - End: {BUDGET_CONFIG['end']}")
    print(f"   - Period: {BUDGET_CONFIG['period']}")
    print(f"   - Total budget points: {len(shared_budgets)}")

    print(f"🔍 Checking checkpoints for run: {args.run_name}")
    print(f"📁 Project: {args.project}")
    print(f"👤 Entity: {args.entity}")
    if args.json_challenges and args.json_solutions:
        print(f"🧩 JSON Challenges: {args.json_challenges}")
        print(f"🎯 JSON Solutions: {args.json_solutions}")
        if args.only_n_tasks:
            print(f"📝 Only evaluating {args.only_n_tasks} tasks")
    if args.dataset_folder:
        print(f"📦 Dataset folder: {args.dataset_folder}")
        if args.dataset_length:
            print(f"🔢 Dataset length: {args.dataset_length}")
        if args.dataset_batch_size:
            print(f"📏 Dataset batch size: {args.dataset_batch_size}")
        print(f"☁️ Use HF: {args.dataset_use_hf}")
        print(f"🌱 Dataset seed: {args.dataset_seed}")

    # Validate eval source selection
    using_json = bool(args.json_challenges and args.json_solutions)
    using_dataset = args.dataset_folder is not None
    if not (using_json or using_dataset) or (using_json and using_dataset):
        print("❌ Provide either both JSON files or a dataset folder (but not both).")
        return

    # Start a single W&B run for this sweep
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=f"evaluate_all_checkpoints::{args.run_name}",
        settings=wandb.Settings(console="off"),
        config={
            "run_name": args.run_name,
            "using_json": using_json,
            "dataset_folder": args.dataset_folder,
        },
    )

    # Fetch checkpoints
    all_checkpoints = get_all_checkpoints(args.run_name, args.project, args.entity)
    if not all_checkpoints:
        print("❌ No checkpoints found. Exiting.")
        try:
            run.finish()
        except Exception:
            pass
        return
    
    # Apply checkpoint selection strategy
    if args.max_checkpoints is not None and args.max_checkpoints < len(all_checkpoints):
        if args.checkpoint_strategy == "even":
            # Select evenly spaced checkpoints
            step = len(all_checkpoints) // args.max_checkpoints
            indices = list(range(0, len(all_checkpoints), step))[:args.max_checkpoints]
            checkpoints = [all_checkpoints[i] for i in indices]
            print(f"📊 Selected {len(checkpoints)} evenly spaced checkpoints from {len(all_checkpoints)} total")
        elif args.checkpoint_strategy == "first":
            # Select first N checkpoints
            checkpoints = all_checkpoints[:args.max_checkpoints]
            print(f"📊 Selected first {len(checkpoints)} checkpoints from {len(all_checkpoints)} total")
        elif args.checkpoint_strategy == "last":
            # Select last N checkpoints
            checkpoints = all_checkpoints[-args.max_checkpoints:]
            print(f"📊 Selected last {len(checkpoints)} checkpoints from {len(all_checkpoints)} total")
        elif args.checkpoint_strategy == "random":
            # Select random N checkpoints
            import random
            random.seed(42)  # Fixed seed for reproducibility
            checkpoints = random.sample(all_checkpoints, args.max_checkpoints)
            checkpoints.sort(key=lambda x: x["step"] if x["step"] is not None else -1)  # Keep sorted
            print(f"📊 Selected {len(checkpoints)} random checkpoints from {len(all_checkpoints)} total")
        else:
            print(f"⚠️  Unknown checkpoint strategy: {args.checkpoint_strategy}. Using all checkpoints.")
            checkpoints = all_checkpoints
    else:
        checkpoints = all_checkpoints
        if args.max_checkpoints is not None:
            print(f"📊 Using all {len(checkpoints)} checkpoints (max_checkpoints={args.max_checkpoints} >= total)")
        else:
            print(f"📊 Using all {len(checkpoints)} checkpoints (no max_checkpoints specified)")
        
    # Budgets (already built)
    ga_steps = shared_budgets
    rs_samples = shared_budgets
    
    # Base method configs
    base_methods = {
        "gradient_ascent": {
            "lr": 0.5,
            "optimizer": "adam",
            "lr_schedule": False,
            "lr_schedule_exponent": 0.5,
        },
        "random_search": {
            "scale": 1.0,
            "scan_batch_size": 10,
            "random_search_seed": 0,
        },
        "evolutionary_search": {
            "population_size": 32,
            "num_generations": 25,
            "mutation_std": 0.5,
        },
    }

    # Result counters
    results = {
        "total_checkpoints": len(checkpoints),
        "successful_evals": 0,
        "failed_evals": 0,
        "method_results": {
            "gradient_ascent": {"success": 0, "failed": 0},
            "random_search": {"success": 0, "failed": 0},
            "evolutionary_search": {"success": 0, "failed": 0},
        },
    }

    print(f"\n🚀 Starting evaluation of {len(checkpoints)} checkpoints...")

    # CSV logging
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"eval_{args.run_name}.csv"
    write_header = not out_csv.exists()

    with out_csv.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(
                ["run_name", "checkpoint_name", "checkpoint_step", "method", "budget_type", "budget", 
                 "overall_accuracy", "top_1_shape_accuracy", "top_1_accuracy", "top_1_pixel_correctness",
                 "top_2_shape_accuracy", "top_2_accuracy", "top_2_pixel_correctness"]
            )

        # Iterate checkpoints
        for i, checkpoint in enumerate(checkpoints, 1):
            step = checkpoint["step"]
            if step is None:
                print(f"⚠️  Skipping checkpoint {checkpoint['name']} (no step info)")
                continue
            
            # Extract training progress from checkpoint version (like plot_from_csv.py)
            checkpoint_name = checkpoint["name"]
            training_progress = 0  # Default to v0
            
            if "--checkpoint:" in checkpoint_name:
                version_part = checkpoint_name.split("--checkpoint:")[1]
                try:
                    version_num = int(version_part[1:])  # Remove 'v' and convert to int
                    training_progress = version_num
                except ValueError:
                    training_progress = 0
            
            denom = max(len(checkpoints) - 1, 1)
            pct = int((training_progress / denom) * 100)

            print("\n" + "=" * 60)
            print(f"📊 Checkpoint {i}/{len(checkpoints)}: Step {step} (v{training_progress})")
            print(f"📁 Artifact: {checkpoint['name']}")
            print(f"🎯 Training Progress: {training_progress}/{denom} ({pct}%)")
            print("=" * 60)

            # Build artifact path for evaluate_checkpoint.py
            artifact_path = f"{args.entity}/{args.project}/{checkpoint['name']}"

            # Gradient Ascent sweeps
            print("\n🔧 Testing gradient_ascent across budgets...")
            for num_steps in ga_steps:
                method_kwargs = dict(base_methods["gradient_ascent"])
                method_kwargs["num_steps"] = num_steps

                ok, acc, metrics, _, execution_time = run_evaluation(
                    artifact_path=artifact_path,
                    method="gradient_ascent",
                    method_kwargs=method_kwargs,
                    json_challenges=args.json_challenges,
                    json_solutions=args.json_solutions,
                    only_n_tasks=args.only_n_tasks,
                    dataset_folder=args.dataset_folder,
                    dataset_length=args.dataset_length,
                    dataset_batch_size=args.dataset_batch_size,
                    dataset_use_hf=(str(args.dataset_use_hf).lower() == "true"),
                    dataset_seed=args.dataset_seed,
                )

                if ok:
                    results["method_results"]["gradient_ascent"]["success"] += 1
                    results["successful_evals"] += 1
                    
                    # Log to W&B immediately
                    try:
                        wandb.log({
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/overall_accuracy": acc or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/gradient_ascent/num_steps_{num_steps}/execution_time": execution_time,
                        })
                    except Exception as e:
                        print(f"⚠️  Failed to log to W&B: {e}")
                else:
                    results["method_results"]["gradient_ascent"]["failed"] += 1
                    results["failed_evals"] += 1

                writer.writerow(
                    [args.run_name, checkpoint["name"], training_progress, "gradient_ascent", "num_steps", num_steps, 
                     acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                     metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                     metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                )

            # Random Search sweeps
            print("\n🔧 Testing random_search across budgets...")
            for num_samples in rs_samples:
                method_kwargs = dict(base_methods["random_search"])
                method_kwargs["num_samples"] = num_samples

                ok, acc, metrics, _, execution_time = run_evaluation(
                    artifact_path=artifact_path,
                    method="random_search",
                    method_kwargs=method_kwargs,
                    json_challenges=args.json_challenges,
                    json_solutions=args.json_solutions,
                    only_n_tasks=args.only_n_tasks,
                    dataset_folder=args.dataset_folder,
                    dataset_length=args.dataset_length,
                    dataset_batch_size=args.dataset_batch_size,
                    dataset_use_hf=(str(args.dataset_use_hf).lower() == "true"),
                    dataset_seed=args.dataset_seed,
                )

                if ok:
                    results["method_results"]["random_search"]["success"] += 1
                    results["successful_evals"] += 1
                    
                    # Log to W&B immediately
                    try:
                        wandb.log({
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/overall_accuracy": acc or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/random_search/num_samples_{num_samples}/execution_time": execution_time,
                        })
                    except Exception as e:
                        print(f"⚠️  Failed to log to W&B: {e}")
                else:
                    results["method_results"]["random_search"]["failed"] += 1
                    results["failed_evals"] += 1

                writer.writerow(
                    [args.run_name, checkpoint["name"], training_progress, "random_search", "num_samples", num_samples, 
                     acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                     metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                     metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                )

            # Evolutionary Search sweeps
            print("\n🔧 Testing evolutionary_search across budgets...")
            for num_generations in rs_samples:  # Use same budget range
                method_kwargs = dict(base_methods["evolutionary_search"])
                method_kwargs["num_generations"] = num_generations

                ok, acc, metrics, _, execution_time = run_evaluation(
                    artifact_path=artifact_path,
                    method="evolutionary_search",
                    method_kwargs=method_kwargs,
                    json_challenges=args.json_challenges,
                    json_solutions=args.json_solutions,
                    only_n_tasks=args.only_n_tasks,
                    dataset_folder=args.dataset_folder,
                    dataset_length=args.dataset_length,
                    dataset_batch_size=args.dataset_batch_size,
                    dataset_use_hf=(str(args.dataset_use_hf).lower() == "true"),
                    dataset_seed=args.dataset_seed,
                )

                if ok:
                    results["method_results"]["evolutionary_search"]["success"] += 1
                    results["successful_evals"] += 1
                    
                    # Log to W&B immediately
                    try:
                        wandb.log({
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/overall_accuracy": acc or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                            f"checkpoint_{step}/evolutionary_search/num_generations_{num_generations}/execution_time": execution_time,
                        })
                    except Exception as e:
                        print(f"⚠️  Failed to log to W&B: {e}")
                else:
                    results["method_results"]["evolutionary_search"]["failed"] += 1
                    results["failed_evals"] += 1

                writer.writerow(
                    [args.run_name, checkpoint["name"], training_progress, "evolutionary_search", "num_generations", num_generations, 
                     acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                     metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                     metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                )
            
            # Progress update after each checkpoint
            total_evals = results["successful_evals"] + results["failed_evals"]
            total_expected = len(ga_steps) + len(rs_samples) + len(es_generations)
            print(f"\n📊 Checkpoint {i}/{len(checkpoints)} complete. Total evaluations: {total_evals}/{total_expected * i}")
            print(f"   ⏱️  Timing info available in W&B logs for each method and budget")
            
            # Generate and upload comparison plot for this step
            try:
                # Accumulate data from CSV for selected methods only
                method_to_step_to_budget: Dict[str, Dict[int, Dict[int, float]]] = {}
                for method in args.plot_methods:
                    method_to_step_to_budget[method] = {}
                
                if out_csv.exists():
                    with out_csv.open("r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                row_step = int(row["checkpoint_step"]) if row["checkpoint_step"] else None
                                if row_step is None:
                                    continue
                                method = row["method"]
                                budget = int(row["budget"]) if row["budget"] else None
                                if budget is None:
                                    continue
                                try:
                                    acc_val = float(row["overall_accuracy"]) if row["overall_accuracy"] not in ("", None) else np.nan
                                except Exception:
                                    acc_val = np.nan
                                
                                method_to_step_to_budget[method].setdefault(row_step, {})[budget] = acc_val
                            except Exception:
                                continue
                
                # Check if we have data for any selected methods
                has_data = any(len(method_data) > 0 for method_data in method_to_step_to_budget.values())
                
                if has_data:
                    # Collect all steps and budgets from selected methods
                    all_steps = set()
                    for method_data in method_to_step_to_budget.values():
                        all_steps.update(method_data.keys())
                    all_steps = sorted(all_steps)
                    all_budgets = sorted(shared_budgets)
                    
                    if all_steps and all_budgets:
                        # Create data arrays for selected methods
                        method_arrays = {}
                        for method in args.plot_methods:
                            method_arrays[method] = np.full((len(all_budgets), len(all_steps)), np.nan)
                        
                        # Fill data arrays
                        for j, s in enumerate(all_steps):
                            for k, b in enumerate(all_budgets):
                                for method in args.plot_methods:
                                    if method in method_to_step_to_budget:
                                        method_arrays[method][k, j] = method_to_step_to_budget[method].get(s, {}).get(b, np.nan)
                        
                        # Create plot with selected methods
                        if len(args.plot_methods) == 2:
                            # Two-method comparison
                            fig = visualize_optimization_comparison(
                                steps=np.array(all_steps),
                                budgets=np.array(all_budgets),
                                acc_A=method_arrays[args.plot_methods[0]],
                                acc_B=method_arrays[args.plot_methods[1]],
                                method_A_name=args.plot_methods[0].replace("_", " ").title(),
                                method_B_name=args.plot_methods[1].replace("_", " ").title(),
                            )
                        else:
                            # Single method or more than 2 methods - create simple heatmap for first method
                            fig = visualize_optimization_comparison(
                                steps=np.array(all_steps),
                                budgets=np.array(all_budgets),
                                acc_A=method_arrays[args.plot_methods[0]],
                                acc_B=np.full_like(method_arrays[args.plot_methods[0]], np.nan),
                                method_A_name=args.plot_methods[0].replace("_", " ").title(),
                                method_B_name="",
                            )
                        
                        fig.suptitle(
                            f"Optimization Comparison - Accumulated Data\n"
                            f"Current Training Progress: {training_progress}/{denom} ({pct}%)\n"
                            f"Checkpoint {i}/{len(checkpoints)} | Total Steps: {len(all_steps)}, Budgets: {len(all_budgets)}",
                            fontsize=14, y=0.98
                        )
                        
                        step_plot_path = out_dir / f"optim_comparison_accumulated_progress_{training_progress}.png"
                        fig.savefig(step_plot_path, dpi=200, bbox_inches='tight')
                        plt.close(fig)
                        
                        # Count available data points without shadowing names
                        data_point_count = 0
                        for method_name, step_map in method_to_step_to_budget.items():
                            for _step, budget_map in step_map.items():
                                for v in budget_map.values():
                                    if not np.isnan(v):
                                        data_point_count += 1
                        
                        wandb.log({
                            f"checkpoint_{training_progress}/optimization_comparison": wandb.Image(str(step_plot_path)),
                            f"checkpoint_{training_progress}/plot_step": training_progress,
                            f"checkpoint_{training_progress}/plot_checkpoint_number": i,
                            f"checkpoint_{training_progress}/plot_total_checkpoints": len(checkpoints),
                            f"checkpoint_{training_progress}/plot_available_steps": len(all_steps),
                            f"checkpoint_{training_progress}/plot_available_budgets": len(all_budgets),
                            f"checkpoint_{training_progress}/plot_accumulated_data": True,
                        })
                        
                        wandb.log({
                            "plot_progression/current_step": training_progress,
                            "plot_progression/checkpoint_number": i,
                            "plot_progression/total_checkpoints": len(checkpoints),
                            "plot_progression/comparison_plot": wandb.Image(str(step_plot_path)),
                            "plot_progression/available_data_points": data_point_count,
                            "plot_progression/accumulated_steps": len(all_steps),
                            "plot_progression/accumulated_budgets": len(all_budgets),
                        })
                        
                        print(f"📊 Generated and uploaded accumulated comparison plot for training progress {training_progress}/{denom} ({pct}%)")
                        print(f"   📈 Available steps: {all_steps}")
                        print(f"   💰 Available budgets: {all_budgets}")
                        print(f"   🔍 Data coverage: {data_point_count} data points")
                        
            except Exception as e:
                print(f"⚠️  Failed to generate comparison plot for training progress {training_progress}: {e}")
            
            # Log checkpoint completion to W&B
            try:
                wandb.log({
                    f"checkpoint_{training_progress}/completion": 1.0,
                    f"checkpoint_{training_progress}/total_evaluations": total_evals,
                    f"checkpoint_{training_progress}/successful_evaluations": results["successful_evals"],
                    f"checkpoint_{training_progress}/failed_evaluations": results["failed_evals"],
                })
                
                overall_progress = i / len(checkpoints)
                wandb.log({
                    "overall/progress": overall_progress,
                    "overall/checkpoints_completed": i,
                    "overall/total_checkpoints": len(checkpoints),
                    "overall/total_evaluations": total_evals,
                    "overall/successful_evaluations": results["successful_evals"],
                    "overall/failed_evaluations": results["failed_evals"],
                })
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint completion to W&B: {e}")

    # Upload CSV artifact
    try:
        artifact = wandb.Artifact(f"{args.run_name}--budgets-eval", type="evaluation")
        artifact.add_file(str(out_csv))
        run.log_artifact(artifact)
    except Exception as e:
        print(f"⚠️  Failed to upload CSV artifact: {e}")

    # Build final optimization comparison plot from CSV (overall summary)
    try:
        steps_list: List[int] = []
        method_maps: Dict[str, Dict[int, Dict[int, float]]] = {}
        for method in args.plot_methods:
            method_maps[method] = {}
            
        with out_csv.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row["checkpoint_step"]) if row["checkpoint_step"] else None
                except Exception:
                    step = None
                if step is None:
                    continue
                steps_list.append(step)
                method = row["method"]
                try:
                    budget = int(row["budget"]) if row["budget"] else None
                except Exception:
                    budget = None
                try:
                    acc_val = float(row["overall_accuracy"]) if row["overall_accuracy"] not in ("", None) else np.nan
                except Exception:
                    acc_val = np.nan
                if budget is None:
                    continue
                if method in args.plot_methods:
                    method_maps[method].setdefault(step, {})[budget] = acc_val

        steps_sorted = sorted(set(steps_list))
        actual_budgets = shared_budgets
        
        # Create data arrays for selected methods
        method_arrays = {}
        for method in args.plot_methods:
            method_arrays[method] = np.full((len(actual_budgets), len(steps_sorted)), np.nan)
            
        for j, s in enumerate(steps_sorted):
            for k, b in enumerate(actual_budgets):
                for method in args.plot_methods:
                    method_arrays[method][k, j] = method_maps[method].get(s, {}).get(b, np.nan)

        # Create plot with selected methods
        if len(args.plot_methods) == 2:
            # Two-method comparison
            fig = visualize_optimization_comparison(
                steps=np.array(steps_sorted),
                budgets=np.array(actual_budgets),
                acc_A=method_arrays[args.plot_methods[0]],
                acc_B=method_arrays[args.plot_methods[1]],
                method_A_name=args.plot_methods[0].replace("_", " ").title(),
                method_B_name=args.plot_methods[1].replace("_", " ").title(),
            )
        else:
            # Single method or more than 2 methods - create simple heatmap for first method
            fig = visualize_optimization_comparison(
                steps=np.array(steps_sorted),
                budgets=np.array(actual_budgets),
                acc_A=method_arrays[args.plot_methods[0]],
                acc_B=np.full_like(method_arrays[args.plot_methods[0]], np.nan),
                method_A_name=args.plot_methods[0].replace("_", " ").title(),
                method_B_name="",
            )
        
        max_progress = max(steps_sorted) if steps_sorted else 0
        denom_final = max(len(checkpoints) - 1, 1)
        progress_percentage = int((max_progress / denom_final) * 100) if steps_sorted else 0
        
        fig.suptitle(
            f"Final Optimization Comparison - {args.run_name}\n"
            f"Training Progress: {len(steps_sorted)} steps (0% → {progress_percentage}%), Budgets: {len(actual_budgets)}", 
            fontsize=14, y=0.98
        )
        
        plot_path = out_dir / f"optim_comparison_final_{args.run_name}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        wandb.log({
            "final/optimization_comparison": wandb.Image(str(plot_path)),
            "final/total_checkpoints": len(steps_sorted),
            "final/total_budgets": len(actual_budgets),
            "final/checkpoint_steps": steps_sorted,
            "final/budget_values": actual_budgets,
            "final/training_progress_percentage": progress_percentage,
        })
        
        plot_art = wandb.Artifact(f"{args.run_name}--final-optim-comparison", type="evaluation")
        plot_art.add_file(str(plot_path))
        run.log_artifact(plot_art)
        
        print(f"📊 Generated and uploaded final comparison plot with {len(steps_sorted)} training progress steps (0% → {progress_percentage}%) and {len(actual_budgets)} budgets")
        print(f"   📈 Methods: {', '.join(args.plot_methods).replace('_', ' ').title()}")
            
    except Exception as e:
        print(f"⚠️  Failed to generate or upload final comparison plot: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total checkpoints: {results['total_checkpoints']}")
    print(f"Successful evaluations: {results['successful_evals']}")
    print(f"Failed evaluations: {results['failed_evals']}")

    for method, stats in results["method_results"].items():
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  ✅ Success: {stats['success']}")
        print(f"  ❌ Failed: {stats['failed']}")

    print(f"\n📊 CSV saved to: {out_csv}")
    print("📈 Available metrics in CSV:")
    print("   - overall_accuracy")
    print("   - top_1_shape_accuracy") 
    print("   - top_1_accuracy")
    print("   - top_1_pixel_correctness")
    print("   - top_2_shape_accuracy")
    print("   - top_2_accuracy")
    print("   - top_2_pixel_correctness")
    print(f"\n🔬 Methods evaluated:")
    for method in ["gradient_ascent", "random_search", "evolutionary_search"]:
        if method in args.plot_methods:
            print(f"   - {method} ({'num_steps' if method == 'gradient_ascent' else 'num_samples' if method == 'random_search' else 'num_generations'})")
        else:
            print(f"   - {method} (skipped - not in plot_methods)")

    try:
        run.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
