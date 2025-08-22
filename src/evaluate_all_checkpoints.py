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
import json
import time

import numpy as np
from matplotlib import pyplot as plt
import subprocess
import wandb
from visualization import visualize_optimization_comparison

# Dataset functionality imports
import jax
from jax.tree_util import tree_map
from data_utils import make_leave_one_out
from train import load_datasets


def log_evaluation_start(method: str, budget_info: Dict[str, Any], method_kwargs: Dict[str, Any], 
                        checkpoint_name: str, checkpoint_step: int) -> None:
    """Log the start of an evaluation with all settings."""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING EVALUATION")
    print(f"{'='*80}")
    print(f"📊 Method: {method}")
    print(f"📁 Checkpoint: {checkpoint_name} (Step: {checkpoint_step})")
    
    if method == "gradient_ascent":
        print(f"⚙️  Settings:")
        print(f"   • Learning Rate: {method_kwargs.get('lr', 'N/A')}")
        print(f"   • Optimizer: {method_kwargs.get('optimizer', 'N/A')}")
        print(f"   • Num Steps: {method_kwargs.get('num_steps', 'N/A')}")
        print(f"   • LR Schedule: {method_kwargs.get('lr_schedule', 'N/A')}")
        print(f"   • LR Schedule Exponent: {method_kwargs.get('lr_schedule_exponent', 'N/A')}")
        print(f"   • Accumulate Gradients Decoder Pairs: {method_kwargs.get('accumulate_gradients_decoder_pairs', 'N/A')}")
        print(f"   • Scan Gradients Latents: {method_kwargs.get('scan_gradients_latents', 'N/A')}")
        print(f"   • Include Mean Latent: {method_kwargs.get('include_mean_latent', 'N/A')}")
        print(f"   • Include All Latents: {method_kwargs.get('include_all_latents', 'N/A')}")
        if method_kwargs.get('random_perturbation'):
            print(f"   • Random Perturbation: {method_kwargs.get('random_perturbation')}")
    
    elif method == "random_search":
        print(f"⚙️  Settings:")
        print(f"   • Num Samples: {method_kwargs.get('num_samples', 'N/A')}")
        print(f"   • Scale: {method_kwargs.get('scale', 'N/A')}")
        print(f"   • Scan Batch Size: {method_kwargs.get('scan_batch_size', 'N/A')}")
        print(f"   • Random Search Seed: {method_kwargs.get('random_search_seed', 'N/A')}")
        print(f"   • Include Mean Latent: {method_kwargs.get('include_mean_latent', 'N/A')}")
        print(f"   • Include All Latents: {method_kwargs.get('include_all_latents', 'N/A')}")
        if method_kwargs.get('random_perturbation'):
            print(f"   • Random Perturbation: {method_kwargs.get('random_perturbation')}")
    
    elif method == "evolutionary_search":
        print(f"⚙️  Settings:")
        print(f"   • Population Size: {method_kwargs.get('population_size', 'N/A')}")
        print(f"   • Num Generations: {method_kwargs.get('num_generations', 'N/A')}")
        print(f"   • Mutation Std: {method_kwargs.get('mutation_std', 'N/A')}")
        print(f"   • Include Mean Latent: {method_kwargs.get('include_mean_latent', 'N/A')}")
        print(f"   • Include All Latents: {method_kwargs.get('include_all_latents', 'N/A')}")
        if method_kwargs.get('random_perturbation'):
            print(f"   • Random Perturbation: {method_kwargs.get('random_perturbation')}")
    
    print(f"💰 Budget Info: {budget_info}")
    if "scaled_budget" in budget_info and budget_info["scaled_budget"] != budget_info.get("value", budget_info["scaled_budget"]):
        print(f"   📊 Scaled Budget: {budget_info['scaled_budget']:.1f} (raw: {budget_info['value']})")
    print(f"{'='*80}")


def log_evaluation_results(method: str, results: Dict[str, Any], execution_time: float, 
                          success: bool, error_msg: str = None) -> None:
    """Log the results of an evaluation."""
    print(f"\n{'='*80}")
    if success:
        print(f"✅ EVALUATION COMPLETED SUCCESSFULLY")
    else:
        print(f"❌ EVALUATION FAILED")
    print(f"{'='*80}")
    print(f"📊 Method: {method}")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    
    if success and results:
        print(f"📈 Results:")
        for key, value in results.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    print(f"   • {key}: {value:.6f}")
                else:
                    print(f"   • {key}: {value}")
            else:
                print(f"   • {key}: None/N/A")
    else:
        print(f"📈 Results: None available")
    
    if not success and error_msg:
        print(f"❌ Error: {error_msg}")
    
    print(f"{'='*80}")


def log_evaluation_summary(checkpoint_name: str, checkpoint_step: int, 
                          method: str, budget_info: Dict[str, Any], 
                          success: bool, execution_time: float) -> Dict[str, Any]:
    """Create a summary log entry for the evaluation."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "method": method,
        "budget_info": budget_info,
        "success": success,
        "execution_time": execution_time,
        "status": "SUCCESS" if success else "FAILED"
    }
    
    print(f"📋 Summary: {checkpoint_name} | {method} | Budget: {budget_info} | "
          f"Status: {summary['status']} | Time: {execution_time:.2f}s")
    
    return summary


def generate_checkpoint_figure(checkpoint_name: str, checkpoint_step: int, training_progress: int, 
                              total_checkpoints: int, results_data: List[Dict[str, Any]], 
                              shared_budgets: List[int], plot_methods: List[str]) -> str:
    """Generate a figure for the current checkpoint and return the file path."""
    try:
        # Create a simple progress visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Training Progress
        progress_pct = (training_progress / max(total_checkpoints - 1, 1)) * 100
        ax1.bar(['Training Progress'], [progress_pct], color='skyblue', alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Progress (%)')
        ax1.set_title(f'Checkpoint Progress: {training_progress}/{total_checkpoints-1}')
        ax1.text(0, progress_pct + 2, f'{progress_pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Method Performance (if we have data)
        if results_data:
            methods = list(set([r['method'] for r in results_data]))
            method_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, method in enumerate(methods):
                method_data = [r for r in results_data if r['method'] == method]
                if method_data:
                    accuracies = [r.get('overall_accuracy', 0) for r in method_data if r.get('overall_accuracy') is not None]
                    if accuracies:
                        ax2.scatter([method] * len(accuracies), accuracies, 
                                   c=method_colors[i % len(method_colors)], alpha=0.7, s=50)
                        ax2.scatter([method], [np.mean(accuracies)], 
                                   c=method_colors[i % len(method_colors)], s=200, marker='*', 
                                   edgecolors='black', linewidth=1)
        
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Method Performance (Current Checkpoint)')
        ax2.set_ylim(0, 1)
        
        # Overall title
        fig.suptitle(f'Checkpoint {checkpoint_name} - Step {checkpoint_step}', fontsize=16, y=0.95)
        
        # Save figure
        out_dir = Path("results")
        fig_path = out_dir / f"checkpoint_{checkpoint_step}_progress_{training_progress}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return str(fig_path)
        
    except Exception as e:
        print(f"⚠️  Failed to generate checkpoint figure: {e}")
        return None


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
    parser.add_argument("--inprocess", action="store_true",
                       help="Run dataset evaluations in-process to reuse a single dataset load (faster)")
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
    # Method hyperparameter overrides
    parser.add_argument("--ga_lr", type=float, default=None,
                   help="Override learning rate (step size) for gradient_ascent")
    parser.add_argument("--es_mutation_std", type=float, default=None,
                   help="Override mutation standard deviation for evolutionary_search")
    
    # Budget multiplier flags
    parser.add_argument("--ga_budget_multiplier", type=float, default=1.0,
                   help="Multiply gradient_ascent num_steps by this factor (keeps raw budget 0-100)")
    parser.add_argument("--es_budget_multiplier", type=float, default=1.0,
                   help="Multiply evolutionary_search population_size and num_generations by this factor (keeps raw budget 0-100)")
    
    # Checkpoint selection options
    parser.add_argument("--max_checkpoints", type=int, default=None,
                       help="Maximum number of checkpoints to evaluate (default: all)")
    parser.add_argument("--checkpoint_strategy", type=str, default="even", 
                       choices=["even", "first", "last", "random"],
                       help="Strategy for selecting checkpoints: 'even'=evenly spaced, 'first'=first N, 'last'=last N, 'random'=random N (default: even)")
    
    # Method selection for plotting
    parser.add_argument("--plot_methods", type=str, nargs="+", 
                       choices=["gradient_ascent", "random_search", "evolutionary_search"],
                       default=["gradient_ascent", "evolutionary_search"],
                       help="Methods to include in plots (default: gradient_ascent, evolutionary_search)")
    
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
    
    # Use the same target compute budgets for all methods
    ga_budgets = shared_budgets    # GA compute budget = 2 * num_steps
    rs_samples = shared_budgets    # Random search uses num_samples
    
    print(f"📊 Using shared budgets: {shared_budgets}")
    print(f"   - Start: {BUDGET_CONFIG['start']}")
    print(f"   - End: {BUDGET_CONFIG['end']}")
    print(f"   - Period: {BUDGET_CONFIG['period']}")
    print(f"   - Total budget points: {len(shared_budgets)}")
    
    # Apply budget multipliers
    if args.ga_budget_multiplier != 1.0:
        print(f"⚙️  GA Budget Multiplier: {args.ga_budget_multiplier}x (num_steps will be scaled)")
    if args.es_budget_multiplier != 1.0:
        print(f"⚙️  ES Budget Multiplier: {args.es_budget_multiplier}x (population_size and num_generations will be scaled)")

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

    # Announce in-process mode selection
    if args.inprocess:
        if using_dataset:
            print("⚡ In-process mode: dataset path selected; dataset will be loaded once and reused.")
        if using_json:
            print("⚡ In-process mode: JSON path selected; no subprocess will be launched for evaluations.")

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
    ga_budgets = shared_budgets
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

    # Apply CLI overrides if provided
    if args.ga_lr is not None:
        try:
            base_methods["gradient_ascent"]["lr"] = float(args.ga_lr)
            print(f"⚙️  Overriding gradient_ascent lr -> {base_methods['gradient_ascent']['lr']}")
        except Exception:
            pass
    if args.es_mutation_std is not None:
        try:
            base_methods["evolutionary_search"]["mutation_std"] = float(args.es_mutation_std)
            print(f"⚙️  Overriding evolutionary_search mutation_std -> {base_methods['evolutionary_search']['mutation_std']}")
        except Exception:
            pass
    
    # Evolutionary search budget: balance population and generations first.
    # Choose population ≈ sqrt(budget), enforce at least 3 and cap at 32, then set generations = ceil(budget / population)
    # Apply budget multiplier to scale both population and generations
    es_configs = []  # list of {budget, population_size, num_generations}
    max_pop = base_methods["evolutionary_search"]["population_size"]
    for b in shared_budgets:
        # Apply budget multiplier
        scaled_budget = b * args.es_budget_multiplier
        proposed_pop = int(round(np.sqrt(scaled_budget)))
        proposed_pop = max(3, min(max_pop, proposed_pop))
        gens = int(max(1, int(np.ceil(scaled_budget / proposed_pop))))
        es_configs.append({
            "budget": int(b), 
            "scaled_budget": scaled_budget,
            "population_size": int(proposed_pop), 
            "num_generations": int(gens)
        })
    try:
        cfg_summary = ", ".join([f"{c['budget']}->{c['population_size']}x{c['num_generations']} (scaled:{c['scaled_budget']:.1f})" for c in es_configs])
        print(f"🧬 Evolutionary configs (budget -> pop x gens): [{cfg_summary}]")
    except Exception:
        pass

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

    # Preload dataset once if requested (dataset mode only)
    preloaded = None
    precomputed_splits = None
    if using_dataset and args.inprocess:
        try:
            # Load data once
            grids, shapes, _ = load_datasets([args.dataset_folder], use_hf=(str(args.dataset_use_hf).lower() == "true"))[0]
            if args.dataset_length is not None:
                key = jax.random.PRNGKey(args.dataset_seed)
                indices = jax.random.permutation(key, len(grids))[: args.dataset_length]
                grids, shapes = grids[indices], shapes[indices]

            # Determine batch size; default to full length if not provided
            dataset_batch_size = args.dataset_batch_size if args.dataset_batch_size is not None else int(grids.shape[0])
            num_devices = max(1, jax.local_device_count())

            # Make batch size divisible by number of devices
            if dataset_batch_size % num_devices != 0:
                # Round down to nearest multiple of num_devices, minimum num_devices
                dataset_batch_size = max(num_devices, (dataset_batch_size // num_devices) * num_devices)

            # Drop last incomplete batch
            num_batches_total = grids.shape[0] // dataset_batch_size
            grids = grids[: num_batches_total * dataset_batch_size]
            shapes = shapes[: num_batches_total * dataset_batch_size]

            # Precompute leave-one-out once
            leave_one_out_grids = make_leave_one_out(grids, axis=-4)
            leave_one_out_shapes = make_leave_one_out(shapes, axis=-3)

            # Split across devices
            def split_devices(x):
                return x.reshape((num_devices, x.shape[0] // num_devices, *x.shape[1:]))

            leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
                split_devices, (leave_one_out_grids, leave_one_out_shapes, grids, shapes)
            )

            # Split into batches per device
            batch_size_per_device = dataset_batch_size // num_devices
            def split_batches(x):
                return x.reshape((x.shape[0], x.shape[1] // batch_size_per_device, batch_size_per_device, *x.shape[2:]))

            leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
                split_batches, (leave_one_out_grids, leave_one_out_shapes, grids, shapes)
            )

            preloaded = {
                "leave_one_out_grids": leave_one_out_grids,
                "leave_one_out_shapes": leave_one_out_shapes,
                "grids": grids,
                "shapes": shapes,
                "num_devices": num_devices,
                "num_batches": grids.shape[1],
            }
            print(f"⚡ In-process dataset loaded once: {grids.shape} examples across {num_devices} devices, {preloaded['num_batches']} batches")
        except Exception as e:
            print(f"⚠️  In-process dataset preload failed, falling back to subprocess mode: {e}")
            args.inprocess = False

    with out_csv.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(
                ["timestamp", "run_name", "checkpoint_name", "checkpoint_step", "method", "budget_type", "budget", 
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

            # Run selected methods in the order provided by --plot_methods
            for method in args.plot_methods:
                if method == "gradient_ascent":
                    print("\n🔧 Testing gradient_ascent across budgets...")
                    for compute_budget in ga_budgets:
                        # Budget = 2x steps => num_steps = ceil(budget / 2)
                        # Apply budget multiplier
                        scaled_budget = compute_budget * args.ga_budget_multiplier
                        num_steps = int(np.ceil(scaled_budget / 2))
                        method_kwargs = dict(base_methods["gradient_ascent"])
                        method_kwargs["num_steps"] = num_steps

                        # Log evaluation start
                        budget_info = {"type": "budget", "value": compute_budget, "num_steps": num_steps, "scaled_budget": scaled_budget}
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step)

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

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

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
                            [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], training_progress, "gradient_ascent", "budget", compute_budget, 
                             acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                             metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                             metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                        )
                elif method == "random_search":
                    print("\n🔧 Testing random_search across budgets...")
                    for num_samples in rs_samples:
                        method_kwargs = dict(base_methods["random_search"])
                        method_kwargs["num_samples"] = num_samples

                        # Log evaluation start
                        budget_info = {"type": "num_samples", "value": num_samples}
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step)

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

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

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
                            [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], training_progress, "random_search", "num_samples", num_samples, 
                             acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                             metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                             metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                        )
                elif method == "evolutionary_search":
                    print("\n🔧 Testing evolutionary_search across budgets...")
                    for es_cfg in es_configs:
                        method_kwargs = dict(base_methods["evolutionary_search"])
                        method_kwargs["population_size"] = es_cfg["population_size"]
                        method_kwargs["num_generations"] = es_cfg["num_generations"]

                        # Log evaluation start
                        budget_info = {
                            "type": "budget", 
                            "value": es_cfg["budget"], 
                            "scaled_budget": es_cfg["scaled_budget"],
                            "population_size": es_cfg["population_size"],
                            "num_generations": es_cfg["num_generations"]
                        }
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step)

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

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

                        if ok:
                            results["method_results"]["evolutionary_search"]["success"] += 1
                            results["successful_evals"] += 1
                            
                            # Log to W&B immediately
                            try:
                                wandb.log({
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/overall_accuracy": acc or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/execution_time": execution_time,
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/population_size": es_cfg["population_size"],
                                    f"checkpoint_{step}/evolutionary_search/budget_{es_cfg['budget']}/num_generations": es_cfg["num_generations"],
                                })
                            except Exception as e:
                                print(f"⚠️  Failed to log to W&B: {e}")
                        else:
                            results["method_results"]["evolutionary_search"]["failed"] += 1
                            results["failed_evals"] += 1

                        writer.writerow(
                            [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], training_progress, "evolutionary_search", "budget", es_cfg["budget"], 
                             acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                             metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                             metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", "")]
                        )
            
            # Progress update after each checkpoint
            total_evals = results["successful_evals"] + results["failed_evals"]
            selected_counts = []
            if "gradient_ascent" in args.plot_methods: selected_counts.append(len(ga_budgets))
            if "random_search" in args.plot_methods: selected_counts.append(len(rs_samples))
            if "evolutionary_search" in args.plot_methods: selected_counts.append(len(es_configs))
            total_expected = sum(selected_counts)
            print(f"\n📊 Checkpoint {i}/{len(checkpoints)} complete. Total evaluations: {total_evals}/{total_expected * i}")
            print(f"   ⏱️  Timing info available in W&B logs for each method and budget")
            
            # Generate and upload checkpoint figure
            try:
                # Collect results data for this checkpoint
                checkpoint_results = []
                for method in args.plot_methods:
                    if method == "gradient_ascent":
                        for compute_budget in ga_budgets:
                            checkpoint_results.append({
                                "method": method,
                                "budget": compute_budget,
                                "budget_type": "budget"
                            })
                    elif method == "random_search":
                        for num_samples in rs_samples:
                            checkpoint_results.append({
                                "method": method,
                                "budget": num_samples,
                                "budget_type": "num_samples"
                            })
                    elif method == "evolutionary_search":
                        for es_cfg in es_configs:
                            checkpoint_results.append({
                                "method": method,
                                "budget": es_cfg["budget"],
                                "budget_type": "budget"
                            })
                
                # Generate checkpoint figure
                fig_path = generate_checkpoint_figure(
                    checkpoint_name=checkpoint["name"],
                    checkpoint_step=step,
                    training_progress=training_progress,
                    total_checkpoints=len(checkpoints),
                    results_data=checkpoint_results,
                    shared_budgets=shared_budgets,
                    plot_methods=args.plot_methods
                )
                
                if fig_path:
                    # Upload to wandb under plots/ panel
                    try:
                        wandb.log({
                            f"plots/checkpoint_{training_progress}_progress": wandb.Image(fig_path),
                            f"plots/checkpoint_{training_progress}_step": step,
                            f"plots/checkpoint_{training_progress}_training_progress": training_progress,
                            f"plots/checkpoint_{training_progress}_total_checkpoints": len(checkpoints),
                            f"plots/checkpoint_{training_progress}_evaluations_completed": total_evals,
                        })
                        print(f"📊 Generated and uploaded checkpoint figure: {fig_path}")
                    except Exception as e:
                        print(f"⚠️  Failed to upload checkpoint figure to W&B: {e}")
                else:
                    print(f"⚠️  Failed to generate checkpoint figure")
                    
            except Exception as e:
                print(f"⚠️  Failed to generate or upload checkpoint figure: {e}")
            
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

    # Comprehensive logging summary
    print(f"\n{'='*80}")
    print("📋 COMPREHENSIVE EVALUATION LOG")
    print(f"{'='*80}")
    print(f"🕐 Run completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Run name: {args.run_name}")
    print(f"🎯 Evaluation source: {'JSON' if using_json else 'Dataset'}")
    if using_json:
        print(f"   • Challenges: {args.json_challenges}")
        print(f"   • Solutions: {args.json_solutions}")
        print(f"   • Tasks limited to: {args.only_n_tasks}")
    if args.dataset_folder:
        print(f"   • Dataset: {args.dataset_folder}")
        print(f"   • Length: {args.dataset_length}")
        print(f"   • Batch size: {args.dataset_batch_size}")
    
    print(f"\n⚙️  Method configurations:")
    for method in args.plot_methods:
        if method == "gradient_ascent":
            print(f"   • {method}: lr={base_methods[method].get('lr')}, optimizer={base_methods[method].get('optimizer')}")
        elif method == "random_search":
            print(f"   • {method}: scale={base_methods[method].get('scale')}, scan_batch_size={base_methods[method].get('scan_batch_size')}")
        elif method == "evolutionary_search":
            print(f"   • {method}: mutation_std={base_methods[method].get('mutation_std')}")
    
    print(f"\n💰 Budget configuration:")
    print(f"   • Start: {args.budget_start}")
    print(f"   • End: {args.budget_end}")
    print(f"   • Period: {args.budget_period}")
    print(f"   • Budgets: {shared_budgets}")
    if args.ga_budget_multiplier != 1.0 or args.es_budget_multiplier != 1.0:
        print(f"   • Budget Multipliers:")
        if args.ga_budget_multiplier != 1.0:
            print(f"     - Gradient Ascent: {args.ga_budget_multiplier}x")
        if args.es_budget_multiplier != 1.0:
            print(f"     - Evolutionary Search: {args.es_budget_multiplier}x")

    print(f"\n📊 Checkpoints evaluated:")
    for cp in checkpoints:
        print(f"   • {cp['name']} (Step: {cp['step']})")
    
    print(f"{'='*80}")

    try:
        run.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
