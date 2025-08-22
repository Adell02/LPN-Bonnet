#!/usr/bin/env python3
"""
Evaluate evolutionary search with different mutation standard deviations using a fixed checkpoint.
Creates a sweep over mutation_std values and plots the results as a line graph.

USAGE EXAMPLES:
==============

1. BASIC USAGE (sweep mutation_std from 0.01 to 1.0):
   python3 src/evaluate_mutation_std.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --mutation_std_start 0.01 \
     --mutation_std_end 1.0 \
     --mutation_std_steps 20

2. QUICK TEST (fewer mutation_std values for fast testing):
   python3 src/evaluate_mutation_std.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --only_n_tasks 5 \
     --mutation_std_start 0.1 \
     --mutation_std_end 0.5 \
     --mutation_std_steps 5

3. FIXED POPULATION AND GENERATIONS:
   python3 src/evaluate_mutation_std.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --fixed_population 10 \
     --fixed_generations 10 \
     --mutation_std_start 0.01 \
     --mutation_std_end 1.0 \
     --mutation_std_steps 20
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

# Dataset functionality imports
import jax
from jax.tree_util import tree_map
from data_utils import make_leave_one_out
from train import load_datasets


def log_evaluation_start(mutation_std: float, population_size: int, num_generations: int, 
                        checkpoint_name: str, checkpoint_step: int) -> None:
    """Log the start of an evaluation with all settings."""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING EVALUATION")
    print(f"{'='*80}")
    print(f"📊 Method: evolutionary_search")
    print(f"📁 Checkpoint: {checkpoint_name} (Step: {checkpoint_step})")
    print(f"⚙️  Settings:")
    print(f"   • Population Size: {population_size}")
    print(f"   • Num Generations: {num_generations}")
    print(f"   • Mutation Std: {mutation_std}")
    print(f"{'='*80}")


def log_evaluation_results(mutation_std: float, results: Dict[str, Any], execution_time: float, 
                          success: bool, error_msg: str = None) -> None:
    """Log the results of an evaluation."""
    print(f"\n{'='*80}")
    if success:
        print(f"✅ EVALUATION COMPLETED SUCCESSFULLY")
    else:
        print(f"❌ EVALUATION FAILED")
    print(f"{'='*80}")
    print(f"📊 Mutation Std: {mutation_std}")
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
                          mutation_std: float, success: bool, execution_time: float) -> Dict[str, Any]:
    """Create a summary log entry for the evaluation."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "mutation_std": mutation_std,
        "success": success,
        "execution_time": execution_time,
        "status": "SUCCESS" if success else "FAILED"
    }
    
    print(f"📋 Summary: {checkpoint_name} | mutation_std={mutation_std} | "
          f"Status: {summary['status']} | Time: {execution_time:.2f}s")
    
    return summary


def generate_mutation_std_plot(mutation_stds: List[float], results_data: List[Dict[str, Any]], 
                               checkpoint_name: str, checkpoint_step: int) -> str:
    """Generate a line plot showing how mutation_std affects different accuracy metrics."""
    try:
        # Extract data for plotting
        metrics = ["overall_accuracy", "top_1_shape_accuracy", "top_1_pixel_correctness"]
        metric_names = ["Overall Accuracy", "Shape Accuracy", "Pixel Correctness"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each metric
        for i, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
            values = []
            for mutation_std in mutation_stds:
                # Find the result for this mutation_std
                result = next((r for r in results_data if r['mutation_std'] == mutation_std), None)
                if result and result.get('success') and result.get('results'):
                    val = result['results'].get(metric)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            # Plot the line
            ax.plot(mutation_stds, values, marker='o', linewidth=2, markersize=8, 
                   color=color, label=metric_name, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel("Mutation Standard Deviation", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"Evolutionary Search Performance vs Mutation Std\n"
                    f"Checkpoint: {checkpoint_name} (Step: {checkpoint_step})", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xscale('log')  # Log scale for mutation_std
        ax.set_ylim(0, 1)
        
        # Add grid lines
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        # Save figure
        out_dir = Path("results")
        fig_path = out_dir / f"mutation_std_sweep_{checkpoint_step}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return str(fig_path)
        
    except Exception as e:
        print(f"⚠️  Failed to generate mutation_std plot: {e}")
        return None


def get_checkpoint(
    run_name: str,
    project_name: str = "LPN-ARC",
    entity: str = "ga624-imperial-college-london",
    checkpoint_strategy: str = "last",
    max_checkpoints: int = 1,
) -> Optional[Dict[str, Any]]:
    """Return a single checkpoint artifact from the specified W&B run."""
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
        
        if not checkpoints:
            print("❌ No checkpoints found.")
            return None
        
        # Select checkpoint based on strategy
        if checkpoint_strategy == "last":
            selected_checkpoint = checkpoints[-1]
        elif checkpoint_strategy == "first":
            selected_checkpoint = checkpoints[0]
        elif checkpoint_strategy == "even":
            # Select middle checkpoint
            selected_checkpoint = checkpoints[len(checkpoints) // 2]
        else:
            selected_checkpoint = checkpoints[0]
        
        print(f"📊 Selected checkpoint: {selected_checkpoint['name']} (Step: {selected_checkpoint['step']})")
        return selected_checkpoint
        
    except Exception as e:
        print(f"Error accessing run: {e}")
        return None
    

def run_evaluation(
    artifact_path: str,
    mutation_std: float,
    population_size: int,
    num_generations: int,
    json_challenges: Optional[str] = None,
    json_solutions: Optional[str] = None,
    only_n_tasks: Optional[int] = None,
    dataset_folder: Optional[str] = None,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: bool = True,
    dataset_seed: int = 0,
) -> Tuple[bool, Optional[float], Dict[str, Optional[float]], str, float]:
    """Invoke evaluate_checkpoint.py for evolutionary search with specific mutation_std."""
    cmd = [sys.executable, "src/evaluate_checkpoint.py", "-w", artifact_path, "-i", "evolutionary_search"]

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

    # Evolutionary search specific args
    cmd.extend([
        "--population-size", str(population_size),
        "--num-generations", str(num_generations),
        "--mutation-std", str(mutation_std),
    ])

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
                f"✅ evolutionary_search evaluation completed successfully"
                + (f" | accuracy={acc}" if acc is not None else "")
                + (f" | shape_acc={metrics.get('top_1_shape_accuracy', 'N/A')}" if metrics.get('top_1_shape_accuracy') is not None else "")
                + (f" | pixel_acc={metrics.get('top_1_pixel_correctness', 'N/A')}" if metrics.get('top_1_pixel_correctness') is not None else "")
                + f" | time={execution_time:.2f}s"
            )
            return True, acc, metrics, stdout, execution_time
        else:
            print(f"❌ evolutionary_search evaluation failed with return code {result.returncode}")
            if stderr.strip():
                print(f"Error output:\n{stderr}")
            return False, acc, metrics, stdout, execution_time
            
    except Exception as e:
        print(f"❌ Error running evolutionary_search evaluation: {e}")
        return False, None, {}, "", 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate evolutionary search with different mutation standard deviations")
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
    
    # Checkpoint selection
    parser.add_argument("--checkpoint_strategy", type=str, default="last", 
                       choices=["even", "first", "last"],
                       help="Strategy for selecting checkpoint: 'first'=first, 'last'=last, 'even'=middle (default: last)")
    
    # Mutation std sweep configuration
    parser.add_argument("--mutation_std_start", type=float, default=0.01, 
                       help="Starting mutation standard deviation (default: 0.01)")
    parser.add_argument("--mutation_std_end", type=float, default=1.0, 
                       help="Ending mutation standard deviation (default: 1.0)")
    parser.add_argument("--mutation_std_steps", type=int, default=20, 
                       help="Number of mutation standard deviation values to test (default: 20)")
    
    # Fixed evolutionary search parameters
    parser.add_argument("--fixed_population", type=int, default=10,
                       help="Fixed population size for all evaluations (default: 10)")
    parser.add_argument("--fixed_generations", type=int, default=10,
                       help="Fixed number of generations for all evaluations (default: 10)")
    
    # Dynamic population/generations calculation (like evaluate_all_checkpoints.py)
    parser.add_argument("--use_dynamic_population", action="store_true",
                       help="Calculate population and generations dynamically based on budget (like evaluate_all_checkpoints.py)")
    parser.add_argument("--budget_for_dynamic", type=int, default=50,
                       help="Budget value to use when calculating dynamic population/generations (default: 50)")
    
    args = parser.parse_args()
    
    # Generate mutation_std values (logarithmically spaced)
    mutation_stds = np.logspace(np.log10(args.mutation_std_start), 
                               np.log10(args.mutation_std_end), 
                               args.mutation_std_steps)
    
    # Calculate population and generations (either fixed or dynamic)
    if args.use_dynamic_population:
        # Use the same logic as evaluate_all_checkpoints.py
        budget = args.budget_for_dynamic
        proposed_pop = int(round(np.sqrt(budget)))
        proposed_pop = max(3, min(32, proposed_pop))  # Cap at 32 like the original
        gens = int(max(1, int(np.ceil(budget / proposed_pop))))
        population_size = proposed_pop
        num_generations = gens
        print(f"🧬 Dynamic population calculation (budget {budget}): population={population_size}, generations={num_generations}")
    else:
        population_size = args.fixed_population
        num_generations = args.fixed_generations
        print(f"🧬 Fixed parameters: population={population_size}, generations={num_generations}")
    
    print(f"🔬 Mutation Standard Deviation Sweep Configuration:")
    print(f"   - Start: {args.mutation_std_start}")
    print(f"   - End: {args.mutation_std_end}")
    print(f"   - Steps: {args.mutation_std_steps}")
    print(f"   - Values: {mutation_stds}")
    print(f"🧬 Evolutionary Search Parameters:")
    print(f"   - Population Size: {population_size}")
    print(f"   - Num Generations: {num_generations}")

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
        name=f"mutation_std_sweep::{args.run_name}",
        settings=wandb.Settings(console="off"),
        config={
            "run_name": args.run_name,
            "using_json": using_json,
            "dataset_folder": args.dataset_folder,
            "mutation_std_start": args.mutation_std_start,
            "mutation_std_end": args.mutation_std_end,
            "mutation_std_steps": args.mutation_std_steps,
            "fixed_population": args.fixed_population,
            "fixed_generations": args.fixed_generations,
        },
    )

    # Fetch single checkpoint
    checkpoint = get_checkpoint(args.run_name, args.project, args.entity, 
                               args.checkpoint_strategy, 1)
    if not checkpoint:
        print("❌ No checkpoint found. Exiting.")
        try:
            run.finish()
        except Exception:
            pass
        return
    
    step = checkpoint["step"]
    if step is None:
        print(f"⚠️  Skipping checkpoint {checkpoint['name']} (no step info)")
        try:
            run.finish()
        except Exception:
            pass
        return
    
    print(f"\n🚀 Starting mutation_std sweep on checkpoint: {checkpoint['name']} (Step: {step})")

    # CSV logging
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"mutation_std_sweep_{args.run_name}_{timestamp}.csv"
    write_header = not out_csv.exists()

    # Preload dataset once if requested (dataset mode only)
    preloaded = None
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

    # Build artifact path for evaluate_checkpoint.py
    artifact_path = f"{args.entity}/{args.project}/{checkpoint['name']}"

    # Result tracking
    results_data = []
    successful_evals = 0
    failed_evals = 0

    with out_csv.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(
                ["timestamp", "run_name", "checkpoint_name", "checkpoint_step", "mutation_std", 
                 "population_size", "num_generations", "overall_accuracy", "top_1_shape_accuracy", 
                 "top_1_accuracy", "top_1_pixel_correctness", "top_2_shape_accuracy", 
                 "top_2_accuracy", "top_2_pixel_correctness", "execution_time"]
            )

        # Run evolutionary search for each mutation_std value
        for i, mutation_std in enumerate(mutation_stds, 1):
            print(f"\n🔬 Testing mutation_std = {mutation_std:.6f} ({i}/{len(mutation_stds)})")

            # Log evaluation start
            log_evaluation_start(mutation_std, population_size, num_generations, 
                               checkpoint["name"], step)

            # Run evaluation
            ok, acc, metrics, _, execution_time = run_evaluation(
                artifact_path=artifact_path,
                mutation_std=mutation_std,
                population_size=population_size,
                num_generations=num_generations,
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
            log_evaluation_results(mutation_std, metrics, execution_time, ok)
            summary = log_evaluation_summary(checkpoint["name"], step, mutation_std, ok, execution_time)

            # Store results for plotting
            result_entry = {
                "mutation_std": mutation_std,
                "success": ok,
                "results": metrics,
                "execution_time": execution_time
            }
            results_data.append(result_entry)

            if ok:
                successful_evals += 1
                
                # Log to W&B immediately
                try:
                    wandb.log({
                        f"mutation_std_{mutation_std:.6f}/overall_accuracy": acc or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                        f"mutation_std_{mutation_std:.6f}/execution_time": execution_time,
                        f"mutation_std_{mutation_std:.6f}/population_size": population_size,
                        f"mutation_std_{mutation_std:.6f}/num_generations": num_generations,
                    })
                except Exception as e:
                    print(f"⚠️  Failed to log to W&B: {e}")
            else:
                failed_evals += 1

            # Write to CSV
            writer.writerow(
                [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], step, 
                 mutation_std, population_size, num_generations,
                 acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                 metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                 metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", ""),
                 execution_time]
            )

    # Generate and upload mutation_std sweep plot
    try:
        fig_path = generate_mutation_std_plot(mutation_stds, results_data, 
                                            checkpoint["name"], step)
        
        if fig_path:
            # Upload to wandb
            try:
                wandb.log({
                    "plots/mutation_std_sweep": wandb.Image(fig_path),
                    "plots/checkpoint_name": checkpoint["name"],
                    "plots/checkpoint_step": step,
                    "plots/mutation_std_values": mutation_stds.tolist(),
                    "plots/successful_evaluations": successful_evals,
                    "plots/failed_evaluations": failed_evals,
                })
                print(f"📊 Generated and uploaded mutation_std sweep plot: {fig_path}")
            except Exception as e:
                print(f"⚠️  Failed to upload plot to W&B: {e}")
        else:
            print(f"⚠️  Failed to generate mutation_std sweep plot")
            
    except Exception as e:
        print(f"⚠️  Failed to generate or upload mutation_std sweep plot: {e}")

    # Upload CSV artifact
    try:
        artifact = wandb.Artifact(f"{args.run_name}--mutation-std-sweep", type="evaluation")
        artifact.add_file(str(out_csv))
        run.log_artifact(artifact)
    except Exception as e:
        print(f"⚠️  Failed to upload CSV artifact: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📈 MUTATION_STD SWEEP SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint['name']} (Step: {step})")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Failed evaluations: {failed_evals}")
    print(f"Total mutation_std values tested: {len(mutation_stds)}")
    print(f"Mutation_std range: {args.mutation_std_start} to {args.mutation_std_end}")
    print(f"Population size: {population_size}")
    print(f"Number of generations: {num_generations}")
    if args.use_dynamic_population:
        print(f"Dynamic calculation used (budget: {args.budget_for_dynamic})")
    else:
        print(f"Fixed parameters used")

    print(f"\n📊 CSV saved to: {out_csv}")
    print(f"📅 Timestamp: {timestamp}")
    print("📈 Available metrics in CSV:")
    print("   - overall_accuracy")
    print("   - top_1_shape_accuracy") 
    print("   - top_1_accuracy")
    print("   - top_1_pixel_correctness")
    print("   - top_2_shape_accuracy")
    print("   - top_2_accuracy")
    print("   - top_2_pixel_correctness")

    # Comprehensive logging summary
    print(f"\n{'='*80}")
    print("📋 COMPREHENSIVE MUTATION_STD SWEEP LOG")
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
    
    print(f"\n🧬 Evolutionary Search Configuration:")
    print(f"   • Population Size: {population_size}")
    print(f"   • Num Generations: {num_generations}")
    print(f"   • Mutation Std Range: {args.mutation_std_start} to {args.mutation_std_end}")
    print(f"   • Mutation Std Steps: {args.mutation_std_steps}")
    print(f"   • Values Tested: {mutation_stds}")
    if args.use_dynamic_population:
        print(f"   • Dynamic Calculation: Yes (budget: {args.budget_for_dynamic})")
    else:
        print(f"   • Dynamic Calculation: No (fixed parameters)")
    
    print(f"\n📊 Checkpoint evaluated:")
    print(f"   • {checkpoint['name']} (Step: {checkpoint['step']})")
    
    print(f"{'='*80}")

    try:
        run.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
