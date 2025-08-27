#!/usr/bin/env python3
"""
Evaluate gradient ascent with different learning rates using a fixed checkpoint.
Creates a sweep over lr values and plots the results as a line graph.

USAGE EXAMPLES:
==============

1. BASIC USAGE (sweep lr from 1e-3 to 1.0):
   python3 src/evaluate_ga_lr.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --lr_start 1e-3 \
     --lr_end 1.0 \
     --lr_steps 15

2. QUICK TEST (fewer lr values for fast testing):
   python3 src/evaluate_ga_lr.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --only_n_tasks 5 \
     --lr_start 1e-2 \
     --lr_end 0.5 \
     --lr_steps 6

3. CUSTOM GA PARAMS (set num_steps/optimizer/schedule):
   python3 src/evaluate_ga_lr.py \
     --run_name "winter-fire-132" \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --num_steps 50 \
     --optimizer adam \
     --lr_schedule false \
     --lr_schedule_exponent 0.5
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

import numpy as np
from matplotlib import pyplot as plt
import subprocess
import wandb


def log_evaluation_start(lr: float, num_steps: int, optimizer: str, lr_schedule: bool,
                        lr_schedule_exponent: float, checkpoint_name: str, checkpoint_step: int) -> None:
    """Log the start of an evaluation with all settings."""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING EVALUATION")
    print(f"{'='*80}")
    print(f"📊 Method: gradient_ascent")
    print(f"📁 Checkpoint: {checkpoint_name} (Step: {checkpoint_step})")
    print(f"⚙️  Settings:")
    print(f"   • Num Steps: {num_steps}")
    print(f"   • Learning Rate: {lr}")
    print(f"   • Optimizer: {optimizer}")
    print(f"   • LR Schedule: {lr_schedule}")
    print(f"   • LR Schedule Exponent: {lr_schedule_exponent}")
    print(f"{'='*80}")


def log_evaluation_results(lr: float, results: Dict[str, Any], execution_time: float,
                          success: bool, error_msg: str = None) -> None:
    """Log the results of an evaluation."""
    print(f"\n{'='*80}")
    if success:
        print(f"✅ EVALUATION COMPLETED SUCCESSFULLY")
    else:
        print(f"❌ EVALUATION FAILED")
    print(f"{'='*80}")
    print(f"📊 Learning Rate: {lr}")
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
                          lr: float, success: bool, execution_time: float) -> Dict[str, Any]:
    """Create a summary log entry for the evaluation."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "lr": lr,
        "success": success,
        "execution_time": execution_time,
        "status": "SUCCESS" if success else "FAILED",
    }
    print(
        f"📋 Summary: {checkpoint_name} | lr={lr} | "
        f"Status: {summary['status']} | Time: {execution_time:.2f}s"
    )
    return summary


def generate_lr_plot(lrs: List[float], results_data: List[Dict[str, Any]],
                     checkpoint_name: str, checkpoint_step: int) -> str:
    """Generate a line plot showing how learning rate affects loss and accuracy metrics."""
    try:
        # Create subplots: one for loss (log scale), one for accuracy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Loss vs Learning Rate (log y-axis) - Note: Loss metric not available yet
        # For now, we'll show a placeholder message
        ax1.text(0.5, 0.5, 'Loss metric not available\nin current evaluation output', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        ax1.set_xlabel("Learning Rate", fontsize=14)
        ax1.set_ylabel("Total Final Loss (log scale)", fontsize=14)
        ax1.set_title(
            f"Gradient Ascent Loss vs Learning Rate\n"
            f"Checkpoint: {checkpoint_name} (Step: {checkpoint_step})\n"
            f"⚠️  Loss metric not available in evaluation output",
            fontsize=16,
        )
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.grid(True, which="minor", ls=":", alpha=0.2)
        
        # Plot 2: Accuracy vs Learning Rate (original plot)
        metrics = ["overall_accuracy", "top_1_shape_accuracy", "top_1_pixel_correctness"]
        metric_names = ["Overall Accuracy", "Shape Accuracy", "Pixel Correctness"]
        colors = ['#FBB998', '#DB74DB', '#5361E5']
        
        for metric, metric_name, color in zip(metrics, metric_names, colors):
            values = []
            for lr in lrs:
                result = next((r for r in results_data if r['lr'] == lr), None)
                if result and result.get('success') and result.get('results'):
                    val = result['results'].get(metric)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            ax2.plot(lrs, values, marker='o', linewidth=2, markersize=8,
                     color=color, label=metric_name, alpha=0.8)

        ax2.set_xlabel("Learning Rate", fontsize=14)
        ax2.set_ylabel("Accuracy", fontsize=14)
        ax2.set_title(
            f"Gradient Ascent Accuracy vs Learning Rate\n"
            f"Checkpoint: {checkpoint_name} (Step: {checkpoint_step})",
            fontsize=16,
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_xscale('log')
        ax2.set_ylim(0, 1)
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.grid(True, which="minor", ls=":", alpha=0.2)

        plt.tight_layout()
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"ga_lr_sweep_{checkpoint_step}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return str(fig_path)
    except Exception as e:
        print(f"⚠️  Failed to generate lr plot: {e}")
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
            if "checkpoint" not in artifact.name.lower():
                continue
            step_match: Optional[int] = None
            if "--checkpoint" in artifact.name:
                name_part = artifact.name.split("--checkpoint")[0]
                nums = re.findall(r"\d+", name_part)
                if nums:
                    step_match = int(nums[-1])
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
                    "name": artifact.name,
                    "step": step_match,
                    "aliases": artifact.aliases,
                }
            )

        checkpoints.sort(key=lambda x: x["step"] if x["step"] is not None else -1)
        if not checkpoints:
            print("❌ No checkpoints found.")
            return None
        if checkpoint_strategy == "last":
            selected_checkpoint = checkpoints[-1]
        elif checkpoint_strategy == "first":
            selected_checkpoint = checkpoints[0]
        elif checkpoint_strategy == "even":
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
    lr: float,
    num_steps: int,
    optimizer: str,
    lr_schedule: bool,
    lr_schedule_exponent: float,
    json_challenges: Optional[str] = None,
    json_solutions: Optional[str] = None,
    only_n_tasks: Optional[int] = None,
    dataset_folder: Optional[str] = None,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: bool = True,
    dataset_seed: int = 0,
) -> Tuple[bool, Optional[float], Dict[str, Optional[float]], str, float]:
    """Invoke evaluate_checkpoint.py for gradient_ascent with specific learning rate."""
    cmd = [sys.executable, "src/evaluate_checkpoint.py", "-w", artifact_path, "-i", "gradient_ascent"]

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
        return False, None, {}, "", 0.0

    cmd.extend([
        "--num-steps", str(num_steps),
        "--lr", str(lr),
        "--optimizer", optimizer,
        "--lr-schedule", str(lr_schedule).lower(),
        "--lr-schedule-exponent", str(lr_schedule_exponent),
    ])

    cmd.extend(["--no-wandb-run", "true"])
    print(f"\nRunning: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        execution_time = time.time() - start_time
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        metrics: Dict[str, Optional[float]] = {}
        acc: Optional[float] = None
        try:
            m = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", stdout.lower())
            if m:
                acc = float(m.group(1))
                metrics["overall_accuracy"] = acc
        except Exception:
            acc = None
            metrics["overall_accuracy"] = None

        metric_patterns = {
            "top_1_shape_accuracy": r"top_1_shape_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_1_accuracy": r"top_1_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_1_pixel_correctness": r"top_1_pixel_correctness:\s*([0-9]*\.?[0-9]+)",
            "top_2_shape_accuracy": r"top_2_shape_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_2_accuracy": r"top_2_accuracy:\s*([0-9]*\.?[0-9]+)",
            "top_2_pixel_correctness": r"top_2_pixel_correctness:\s*([0-9]*\.?[0-9]+)",
            "correct_shapes": r"correct_shapes:\s*([0-9]*\.?[0-9]+)",
            "pixel_correctness": r"pixel_correctness:\s*([0-9]*\.?[0-9]+)",
            # Loss metrics
            "total_final_loss": r"total_final_loss:\s*([0-9]*\.?[0-9]+)",
        }
        for metric_name, pattern in metric_patterns.items():
            try:
                m2 = re.search(pattern, stdout.lower())
                if m2:
                    metrics[metric_name] = float(m2.group(1))
                else:
                    metrics[metric_name] = None
            except Exception:
                metrics[metric_name] = None

        if result.returncode == 0:
            print(
                f"✅ gradient_ascent evaluation completed successfully"
                + (f" | accuracy={acc}" if acc is not None else "")
                + (f" | shape_acc={metrics.get('top_1_shape_accuracy', 'N/A')}" if metrics.get('top_1_shape_accuracy') is not None else "")
                + (f" | pixel_acc={metrics.get('top_1_pixel_correctness', 'N/A')}" if metrics.get('top_1_pixel_correctness') is not None else "")
                + (f" | loss={metrics.get('total_final_loss', 'N/A')}" if metrics.get('total_final_loss') is not None else "")
                + f" | time={execution_time:.2f}s"
            )
            return True, acc, metrics, stdout, execution_time
        else:
            print(f"❌ gradient_ascent evaluation failed with return code {result.returncode}")
            if stderr.strip():
                print(f"Error output:\n{stderr}")
            return False, acc, metrics, stdout, execution_time
    except Exception as e:
        print(f"❌ Error running gradient_ascent evaluation: {e}")
        return False, None, {}, "", 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate gradient ascent with different learning rates")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the W&B run")
    parser.add_argument("--json_challenges", type=str, default=None, help="Path to JSON challenges")
    parser.add_argument("--json_solutions", type=str, default=None, help="Path to JSON solutions")
    parser.add_argument("--only_n_tasks", type=int, default=None, help="Limit number of tasks evaluated")

    # Dataset evaluation
    parser.add_argument("-d", "--dataset_folder", type=str, default=None,
                        help="Dataset folder under 'src/datasets' (e.g., 'pattern2d_eval')")
    parser.add_argument("--dataset_length", type=int, default=None, help="Max examples to eval")
    parser.add_argument("--dataset_batch_size", type=int, default=None, help="Batch size for dataset eval")
    parser.add_argument("--dataset_use_hf", type=str, default="true", help="Use HF hub (true/false)")
    parser.add_argument("--dataset_seed", type=int, default=0, help="Seed for dataset subsampling")

    parser.add_argument("--project", type=str, default="LPN-ARC", help="W&B project name")
    parser.add_argument("--entity", type=str, default="ga624-imperial-college-london", help="W&B entity")

    # Checkpoint selection
    parser.add_argument("--checkpoint_strategy", type=str, default="last",
                        choices=["even", "first", "last"],
                        help="Strategy for selecting checkpoint: 'first'=first, 'last'=last, 'even'=middle (default: last)")

    # GA sweep configuration
    parser.add_argument("--lr_start", type=float, default=1e-3, help="Starting learning rate (default: 1e-3)")
    parser.add_argument("--lr_end", type=float, default=1.0, help="Ending learning rate (default: 1.0)")
    parser.add_argument("--lr_steps", type=int, default=15, help="Number of learning rate values to test (default: 15)")

    # GA fixed parameters
    parser.add_argument("--num_steps", type=int, default=50, help="Number of GA steps for all evaluations (default: 50)")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for GA (default: adam)")
    parser.add_argument("--lr_schedule", type=str, default="false", help="Use LR schedule (true/false) (default: false)")
    parser.add_argument("--lr_schedule_exponent", type=float, default=0.5, help="LR schedule exponent (default: 0.5)")

    args = parser.parse_args()

    # Generate LR values (logarithmically spaced)
    lrs = np.logspace(np.log10(args.lr_start), np.log10(args.lr_end), args.lr_steps)

    # Parse lr_schedule string to bool
    lr_schedule_bool = str(args.lr_schedule).lower() == "true"

    print(f"🔬 Learning Rate Sweep Configuration:")
    print(f"   - Start: {args.lr_start}")
    print(f"   - End: {args.lr_end}")
    print(f"   - Steps: {args.lr_steps}")
    print(f"   - Values: {lrs}")
    print(f"🛠️  Gradient Ascent Parameters:")
    print(f"   - Num Steps: {args.num_steps}")
    print(f"   - Optimizer: {args.optimizer}")
    print(f"   - LR Schedule: {lr_schedule_bool}")
    print(f"   - LR Schedule Exponent: {args.lr_schedule_exponent}")

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
        name=f"ga_lr_sweep::{args.run_name}",
        settings=wandb.Settings(console="off"),
        config={
            "run_name": args.run_name,
            "using_json": using_json,
            "dataset_folder": args.dataset_folder,
            "lr_start": args.lr_start,
            "lr_end": args.lr_end,
            "lr_steps": args.lr_steps,
            "num_steps": args.num_steps,
            "optimizer": args.optimizer,
            "lr_schedule": lr_schedule_bool,
            "lr_schedule_exponent": args.lr_schedule_exponent,
        },
    )

    # Fetch single checkpoint
    checkpoint = get_checkpoint(args.run_name, args.project, args.entity, args.checkpoint_strategy, 1)
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

    print(f"\n🚀 Starting lr sweep on checkpoint: {checkpoint['name']} (Step: {step})")

    # CSV logging
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"ga_lr_sweep_{args.run_name}_{timestamp}.csv"
    write_header = not out_csv.exists()

    # Results
    results_data: List[Dict[str, Any]] = []
    successful_evals = 0
    failed_evals = 0

    artifact_path = f"{args.entity}/{args.project}/{checkpoint['name']}"

    with out_csv.open("a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow([
                "timestamp", "run_name", "checkpoint_name", "checkpoint_step", "lr",
                "num_steps", "optimizer", "lr_schedule", "lr_schedule_exponent",
                "overall_accuracy", "top_1_shape_accuracy", "top_1_accuracy",
                "top_1_pixel_correctness", "top_2_shape_accuracy", "top_2_accuracy",
                "top_2_pixel_correctness", "total_final_loss", "execution_time",
            ])

        for i, lr in enumerate(lrs, 1):
            print(f"\n🔬 Testing lr = {lr:.6f} ({i}/{len(lrs)})")

            log_evaluation_start(lr, args.num_steps, args.optimizer, lr_schedule_bool,
                                 args.lr_schedule_exponent, checkpoint["name"], step)

            ok, acc, metrics, _, execution_time = run_evaluation(
                artifact_path=artifact_path,
                lr=lr,
                num_steps=args.num_steps,
                optimizer=args.optimizer,
                lr_schedule=lr_schedule_bool,
                lr_schedule_exponent=args.lr_schedule_exponent,
                json_challenges=args.json_challenges,
                json_solutions=args.json_solutions,
                only_n_tasks=args.only_n_tasks,
                dataset_folder=args.dataset_folder,
                dataset_length=args.dataset_length,
                dataset_batch_size=args.dataset_batch_size,
                dataset_use_hf=(str(args.dataset_use_hf).lower() == "true"),
                dataset_seed=args.dataset_seed,
            )

            log_evaluation_results(lr, metrics, execution_time, ok)
            _ = log_evaluation_summary(checkpoint["name"], step, lr, ok, execution_time)

            results_data.append({
                "lr": lr,
                "success": ok,
                "results": metrics,
                "execution_time": execution_time,
            })

            if ok:
                successful_evals += 1
                try:
                    wandb.log({
                        f"lr_{lr:.6f}/overall_accuracy": acc or 0.0,
                        f"lr_{lr:.6f}/top_1_shape_accuracy": metrics.get("top_1_shape_accuracy", 0.0) or 0.0,
                        f"lr_{lr:.6f}/top_1_accuracy": metrics.get("top_1_accuracy", 0.0) or 0.0,
                        f"lr_{lr:.6f}/top_1_pixel_correctness": metrics.get("top_1_pixel_correctness", 0.0) or 0.0,
                        f"lr_{lr:.6f}/top_2_shape_accuracy": metrics.get("top_2_shape_accuracy", 0.0) or 0.0,
                        f"lr_{lr:.6f}/top_2_accuracy": metrics.get("top_2_accuracy", 0.0) or 0.0,
                        f"lr_{lr:.6f}/top_2_pixel_correctness": metrics.get("top_2_pixel_correctness", 0.0) or 0.0,
                        f"lr_{lr:.6f}/total_final_loss": metrics.get("total_final_loss", 0.0) or 0.0,
                        f"lr_{lr:.6f}/execution_time": execution_time,
                        f"lr_{lr:.6f}/num_steps": args.num_steps,
                        f"lr_{lr:.6f}/optimizer": args.optimizer,
                        f"lr_{lr:.6f}/lr_schedule": lr_schedule_bool,
                        f"lr_{lr:.6f}/lr_schedule_exponent": args.lr_schedule_exponent,
                    })
                except Exception as e:
                    print(f"⚠️  Failed to log to W&B: {e}")
            else:
                failed_evals += 1

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], step,
                lr, args.num_steps, args.optimizer, lr_schedule_bool, args.lr_schedule_exponent,
                acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", ""),
                metrics.get("total_final_loss", ""), execution_time,
            ])

    try:
        fig_path = generate_lr_plot(lrs, results_data, checkpoint["name"], step)
        if fig_path:
            try:
                wandb.log({
                    "plots/ga_lr_sweep": wandb.Image(fig_path),
                    "plots/checkpoint_name": checkpoint["name"],
                    "plots/checkpoint_step": step,
                    "plots/lr_values": lrs.tolist(),
                    "plots/successful_evaluations": successful_evals,
                    "plots/failed_evaluations": failed_evals,
                })
                print(f"📊 Generated and uploaded GA lr sweep plot: {fig_path}")
            except Exception as e:
                print(f"⚠️  Failed to upload plot to W&B: {e}")
        else:
            print(f"⚠️  Failed to generate GA lr sweep plot")
    except Exception as e:
        print(f"⚠️  Failed to generate or upload GA lr sweep plot: {e}")

    try:
        artifact = wandb.Artifact(f"{args.run_name}--ga-lr-sweep", type="evaluation")
        artifact.add_file(str(out_csv))
        run.log_artifact(artifact)
    except Exception as e:
        print(f"⚠️  Failed to upload CSV artifact: {e}")

    print("\n" + "=" * 60)
    print("📈 GA LR SWEEP SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint['name']} (Step: {step})")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Failed evaluations: {failed_evals}")
    print(f"Total lr values tested: {len(lrs)}")
    print(f"LR range: {args.lr_start} to {args.lr_end}")
    print(f"Num steps: {args.num_steps}")
    print(f"Optimizer: {args.optimizer}")
    print(f"LR schedule: {lr_schedule_bool}")
    print(f"LR schedule exponent: {args.lr_schedule_exponent}")
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
    print("   - total_final_loss (⚠️  Not available in current evaluation output)")

    try:
        run.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()


