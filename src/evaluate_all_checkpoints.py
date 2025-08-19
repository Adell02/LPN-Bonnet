#!/usr/bin/env python3
"""
Evaluate all checkpoints from a specific Weights & Biases run using src/evaluate_checkpoint.py.
Runs both gradient_ascent and random_search for each checkpoint and logs results to CSV.
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import subprocess
import wandb


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
) -> Tuple[bool, Optional[float], str]:
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
        return False, None, ""

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
    else:
        print(f"❌ Unknown method: {method}")
        return False, None, ""

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Parse accuracy from stdout (case-insensitive)
        acc: Optional[float] = None
        try:
            m = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", stdout.lower())
            if m:
                acc = float(m.group(1))
        except Exception:
            acc = None

        if result.returncode == 0:
            print(
                f"✅ {method} evaluation completed successfully"
                + (f" | accuracy={acc}" if acc is not None else "")
            )
            return True, acc, stdout
        else:
            print(f"❌ {method} evaluation failed with return code {result.returncode}")
            if stderr.strip():
                print(f"Error output:\n{stderr}")
            return False, acc, stdout

    except Exception as e:
        print(f"❌ Error running {method} evaluation: {e}")
        return False, None, ""


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

    args = parser.parse_args()

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

    # Fetch checkpoints
    checkpoints = get_all_checkpoints(args.run_name, args.project, args.entity)
    if not checkpoints:
        print("❌ No checkpoints found. Exiting.")
        return

    # Budgets
    ga_steps = [1] + list(range(5, 101, 5))    # 1,5,10,...,100
    rs_samples = [1] + list(range(5, 101, 5))  # 1,5,10,...,100

    # Base method configs
    base_methods = {
        "gradient_ascent": {
            "lr": 0.1,
            "optimizer": "adam",
            "lr_schedule": False,
            "lr_schedule_exponent": 0.5,
        },
        "random_search": {
            "scale": 1.0,
            "scan_batch_size": 10,
            "random_search_seed": 0,
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
                ["run_name", "checkpoint_name", "checkpoint_step", "method", "budget_type", "budget", "accuracy"]
            )

        # Iterate checkpoints
        for i, checkpoint in enumerate(checkpoints, 1):
            step = checkpoint["step"]
            if step is None:
                print(f"⚠️  Skipping checkpoint {checkpoint['name']} (no step info)")
                continue

            print("\n" + "=" * 60)
            print(f"📊 Checkpoint {i}/{len(checkpoints)}: Step {step}")
            print(f"📁 Artifact: {checkpoint['name']}")
            print("=" * 60)

            # Build artifact path for evaluate_checkpoint.py
            # W&B expects '{entity}/{project}/{artifact_name}' where artifact_name includes ':version' or alias
            artifact_path = f"{args.entity}/{args.project}/{checkpoint['name']}"

            # Gradient Ascent sweeps
            print("\n🔧 Testing gradient_ascent across budgets...")
            for num_steps in ga_steps:
                method_kwargs = dict(base_methods["gradient_ascent"])
                method_kwargs["num_steps"] = num_steps

                ok, acc, _ = run_evaluation(
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
                else:
                    results["method_results"]["gradient_ascent"]["failed"] += 1
                    results["failed_evals"] += 1

                writer.writerow(
                    [args.run_name, checkpoint["name"], step, "gradient_ascent", "num_steps", num_steps, acc or ""]
                )

            # Random Search sweeps
            print("\n🔧 Testing random_search across budgets...")
            for num_samples in rs_samples:
                method_kwargs = dict(base_methods["random_search"])
                method_kwargs["num_samples"] = num_samples

                ok, acc, _ = run_evaluation(
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
                else:
                    results["method_results"]["random_search"]["failed"] += 1
                    results["failed_evals"] += 1

                writer.writerow(
                    [args.run_name, checkpoint["name"], step, "random_search", "num_samples", num_samples, acc or ""]
                )

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

    print("\n🎉 Evaluation complete!")
    if results["failed_evals"] == 0:
        print("🎊 All evaluations completed successfully!")
    else:
        print(f"⚠️  {results['failed_evals']} evaluations failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
