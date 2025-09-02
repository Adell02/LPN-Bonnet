#!/usr/bin/env python3
"""
Compare search methods (GA vs ES) across different checkpoints and create heatmaps.

This script evaluates multiple checkpoints from a W&B run, runs both gradient ascent
and evolutionary search with a specified budget, extracts intermediate metrics,
and creates individual and differential heatmaps using visualize_loss_difference_heatmap.

Example usage:
python src/compare_search_heatmap.py \
  --run_name 8ejrpt3n \
  --project LPN-eval-heatmap \
  --dataset_folder tetro_pattern \
  --dataset_length 500 \
  --dataset_batch_size 20 \
  --dataset_seed 0 \
  --dataset_use_hf false \
  --max_checkpoints 10 \
  --checkpoint_strategy even \
  --budget_start 10 \
  --budget_end 500 \
  --ga_lr 0.2 \
  --es_mutation_std 0.05 \
  --es_mutation_decay 0.95
"""

import argparse
import os
import sys
import tempfile
import subprocess
import numpy as np
import wandb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from visualization import visualize_loss_difference_heatmap


def get_checkpoints_from_run(run_name: str, project: str, max_checkpoints: int, strategy: str) -> List[str]:
    """Get checkpoint artifact paths from a W&B run."""
    api = wandb.Api()
    run = api.run(f"{project}/{run_name}")
    
    # Get all checkpoint artifacts
    artifacts = []
    for artifact in run.logged_artifacts():
        if "checkpoint" in artifact.name:
            artifacts.append(artifact)
    
    # Sort by creation time
    artifacts.sort(key=lambda x: x.created_at)
    
    if strategy == "even":
        # Select evenly spaced checkpoints
        if len(artifacts) <= max_checkpoints:
            selected = artifacts
        else:
            indices = np.linspace(0, len(artifacts) - 1, max_checkpoints, dtype=int)
            selected = [artifacts[i] for i in indices]
    elif strategy == "latest":
        # Select latest checkpoints
        selected = artifacts[-max_checkpoints:]
    else:
        raise ValueError(f"Unknown checkpoint strategy: {strategy}")
    
    # Return artifact paths
    checkpoint_paths = []
    for artifact in selected:
        path = f"{artifact.entity}/{artifact.project}/{artifact.name}:{artifact.version}"
        checkpoint_paths.append(path)
    
    print(f"Selected {len(checkpoint_paths)} checkpoints from {len(artifacts)} total")
    return checkpoint_paths


def run_evaluation_with_budget(
    artifact_path: str,
    method: str,
    budget: int,
    ga_lr: float,
    es_mutation_std: float,
    es_mutation_decay: float,
    dataset_folder: str,
    dataset_length: int,
    dataset_batch_size: int,
    dataset_use_hf: bool,
    dataset_seed: int,
    temp_dir: str
) -> Tuple[bool, Dict[str, Any]]:
    """Run evaluation with a specific budget and extract intermediate metrics."""
    
    # Calculate method-specific parameters
    if method == "gradient_ascent":
        ga_steps = budget // 2  # Each step = 2 evaluations (forward + backward)
        cmd = [
            sys.executable, "src/evaluate_checkpoint.py",
            "-w", artifact_path,
            "-d", dataset_folder,
            "--dataset-length", str(dataset_length),
            "--dataset-batch-size", str(dataset_batch_size),
            "--dataset-use-hf", str(dataset_use_hf).lower(),
            "--dataset-seed", str(dataset_seed),
            "-i", "gradient_ascent",
            "--num-steps", str(ga_steps),
            "--lr", str(ga_lr),
            "--no-wandb-run", "true",
            "--store-latents", os.path.join(temp_dir, f"ga_latents_{budget}.npz")
        ]
    elif method == "evolutionary_search":
        # ES: population * generations = budget
        pop = int(np.sqrt(budget))
        gens = budget // pop
        
        cmd = [
            sys.executable, "src/evaluate_checkpoint.py",
            "-w", artifact_path,
            "-d", dataset_folder,
            "--dataset-length", str(dataset_length),
            "--dataset-batch-size", str(dataset_batch_size),
            "--dataset-use-hf", str(dataset_use_hf).lower(),
            "--dataset-seed", str(dataset_seed),
            "-i", "evolutionary_search",
            "--population-size", str(pop),
            "--num-generations", str(gens),
            "--mutation-std", str(es_mutation_std),
            "--mutation-decay", str(es_mutation_decay),
            "--no-wandb-run", "true",
            "--store-latents", os.path.join(temp_dir, f"es_latents_{budget}.npz")
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    try:
        print(f"Running {method} with budget {budget}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error running {method} with budget {budget}:")
            print(result.stderr)
            return False, {}
        
        # Parse output to extract metrics
        output_lines = result.stdout.split('\n')
        metrics = {}
        
        for line in output_lines:
            if "Final accuracy:" in line:
                try:
                    accuracy = float(line.split(":")[1].strip())
                    metrics["accuracy"] = accuracy
                except:
                    pass
            elif "Final loss:" in line:
                try:
                    loss = float(line.split(":")[1].strip())
                    metrics["loss"] = loss
                except:
                    pass
        
        # Try to load trajectory data if available
        latents_file = os.path.join(temp_dir, f"{method}_latents_{budget}.npz")
        if os.path.exists(latents_file):
            try:
                data = np.load(latents_file)
                if "losses" in data:
                    metrics["losses"] = data["losses"]
                if "accuracies" in data:
                    metrics["accuracies"] = data["accuracies"]
                if "steps" in data:
                    metrics["steps"] = data["steps"]
            except Exception as e:
                print(f"Warning: Could not load trajectory data: {e}")
        
        return True, metrics
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running {method} with budget {budget}")
        return False, {}
    except Exception as e:
        print(f"Error running {method} with budget {budget}: {e}")
        return False, {}


def create_heatmaps(
    checkpoint_results: List[Dict[str, Any]],
    budgets: List[int],
    output_dir: str
) -> List[str]:
    """Create individual and differential heatmaps from checkpoint results."""
    
    heatmap_files = []
    
    for i, checkpoint_data in enumerate(checkpoint_results):
        checkpoint_name = checkpoint_data["checkpoint_name"]
        ga_results = checkpoint_data["ga_results"]
        es_results = checkpoint_data["es_results"]
        
        # Extract losses for both methods
        ga_losses = []
        es_losses = []
        
        for budget in budgets:
            if budget in ga_results and "losses" in ga_results[budget]:
                ga_losses.append(ga_results[budget]["losses"])
            else:
                ga_losses.append(np.array([0.0]))  # Placeholder
                
            if budget in es_results and "losses" in es_results[budget]:
                es_losses.append(es_results[budget]["losses"])
            else:
                es_losses.append(np.array([0.0]))  # Placeholder
        
        # Create individual heatmaps
        if ga_losses and any(len(losses) > 0 for losses in ga_losses):
            # GA individual heatmap
            max_steps = max(len(losses) for losses in ga_losses if len(losses) > 0)
            ga_loss_matrix = np.zeros((len(budgets), max_steps))
            
            for j, losses in enumerate(ga_losses):
                if len(losses) > 0:
                    ga_loss_matrix[j, :len(losses)] = losses
                    if len(losses) < max_steps:
                        # Pad with last value
                        ga_loss_matrix[j, len(losses):] = losses[-1]
            
            steps = np.arange(max_steps)
            fig = visualize_loss_difference_heatmap(
                steps, budgets, ga_loss_matrix,
                method_A_name="GA", method_B_name="GA"
            )
            ga_file = os.path.join(output_dir, f"ga_individual_{checkpoint_name}.png")
            fig.savefig(ga_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(ga_file)
        
        if es_losses and any(len(losses) > 0 for losses in es_losses):
            # ES individual heatmap
            max_steps = max(len(losses) for losses in es_losses if len(losses) > 0)
            es_loss_matrix = np.zeros((len(budgets), max_steps))
            
            for j, losses in enumerate(es_losses):
                if len(losses) > 0:
                    es_loss_matrix[j, :len(losses)] = losses
                    if len(losses) < max_steps:
                        # Pad with last value
                        es_loss_matrix[j, len(losses):] = losses[-1]
            
            steps = np.arange(max_steps)
            fig = visualize_loss_difference_heatmap(
                steps, budgets, es_loss_matrix,
                method_A_name="ES", method_B_name="ES"
            )
            es_file = os.path.join(output_dir, f"es_individual_{checkpoint_name}.png")
            fig.savefig(es_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(es_file)
        
        # Create differential heatmap (ES - GA)
        if ga_losses and es_losses:
            # Align the loss trajectories
            max_steps = max(
                max(len(losses) for losses in ga_losses if len(losses) > 0),
                max(len(losses) for losses in es_losses if len(losses) > 0)
            )
            
            ga_loss_matrix = np.zeros((len(budgets), max_steps))
            es_loss_matrix = np.zeros((len(budgets), max_steps))
            
            for j, (ga_loss, es_loss) in enumerate(zip(ga_losses, es_losses)):
                if len(ga_loss) > 0:
                    ga_loss_matrix[j, :len(ga_loss)] = ga_loss
                    if len(ga_loss) < max_steps:
                        ga_loss_matrix[j, len(ga_loss):] = ga_loss[-1]
                
                if len(es_loss) > 0:
                    es_loss_matrix[j, :len(es_loss)] = es_loss
                    if len(es_loss) < max_steps:
                        es_loss_matrix[j, len(es_loss):] = es_loss[-1]
            
            # Calculate difference (ES - GA)
            loss_diff = es_loss_matrix - ga_loss_matrix
            
            steps = np.arange(max_steps)
            fig = visualize_loss_difference_heatmap(
                steps, budgets, loss_diff,
                method_A_name="GA", method_B_name="ES"
            )
            diff_file = os.path.join(output_dir, f"differential_{checkpoint_name}.png")
            fig.savefig(diff_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(diff_file)
    
    return heatmap_files


def main():
    parser = argparse.ArgumentParser(description="Compare search methods across checkpoints and create heatmaps")
    
    # W&B configuration
    parser.add_argument("--run_name", type=str, required=True, help="Name of the W&B run")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity")
    
    # Dataset configuration
    parser.add_argument("--dataset_folder", type=str, required=True, help="Dataset folder name")
    parser.add_argument("--dataset_length", type=int, default=500, help="Dataset length")
    parser.add_argument("--dataset_batch_size", type=int, default=20, help="Dataset batch size")
    parser.add_argument("--dataset_seed", type=int, default=0, help="Dataset seed")
    parser.add_argument("--dataset_use_hf", type=str, default="false", help="Use HF hub (true/false)")
    
    # Checkpoint configuration
    parser.add_argument("--max_checkpoints", type=int, default=10, help="Maximum number of checkpoints to evaluate")
    parser.add_argument("--checkpoint_strategy", type=str, default="even", choices=["even", "latest"], 
                       help="Strategy for selecting checkpoints")
    
    # Budget configuration
    parser.add_argument("--budget_start", type=int, default=10, help="Starting budget")
    parser.add_argument("--budget_end", type=int, default=500, help="Ending budget")
    parser.add_argument("--budget_step", type=int, default=10, help="Budget step size")
    
    # Method parameters
    parser.add_argument("--ga_lr", type=float, default=0.2, help="Gradient ascent learning rate")
    parser.add_argument("--es_mutation_std", type=float, default=0.05, help="ES mutation standard deviation")
    parser.add_argument("--es_mutation_decay", type=float, default=0.95, help="ES mutation decay")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="results/heatmaps", help="Output directory for heatmaps")
    parser.add_argument("--wandb_project", type=str, default="LPN-eval-heatmap", help="W&B project for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.entity,
        name=f"heatmap_comparison_{args.run_name}",
        config=vars(args)
    )
    
    # Get checkpoints
    print(f"Getting checkpoints from run {args.run_name}...")
    checkpoint_paths = get_checkpoints_from_run(
        args.run_name, args.project, args.max_checkpoints, args.checkpoint_strategy
    )
    
    # Generate budget range
    budgets = list(range(args.budget_start, args.budget_end + 1, args.budget_step))
    print(f"Evaluating budgets: {budgets}")
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_results = []
        
        # Evaluate each checkpoint
        for i, checkpoint_path in enumerate(checkpoint_paths):
            checkpoint_name = checkpoint_path.split("/")[-1].split(":")[0]
            print(f"\nEvaluating checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_name}")
            
            ga_results = {}
            es_results = {}
            
            # Test with max budget first (as requested)
            max_budget = max(budgets)
            print(f"Testing with max budget {max_budget}...")
            
            # Run GA
            success, metrics = run_evaluation_with_budget(
                checkpoint_path, "gradient_ascent", max_budget,
                args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                args.dataset_use_hf == "true", args.dataset_seed, temp_dir
            )
            if success:
                ga_results[max_budget] = metrics
                print(f"GA max budget test: accuracy={metrics.get('accuracy', 'N/A')}, loss={metrics.get('loss', 'N/A')}")
            else:
                print("GA max budget test failed")
            
            # Run ES
            success, metrics = run_evaluation_with_budget(
                checkpoint_path, "evolutionary_search", max_budget,
                args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                args.dataset_use_hf == "true", args.dataset_seed, temp_dir
            )
            if success:
                es_results[max_budget] = metrics
                print(f"ES max budget test: accuracy={metrics.get('accuracy', 'N/A')}, loss={metrics.get('loss', 'N/A')}")
            else:
                print("ES max budget test failed")
            
            # If max budget test works, run all budgets
            if ga_results or es_results:
                print(f"Max budget test successful, running all budgets...")
                
                for budget in budgets:
                    if budget == max_budget:
                        continue  # Already tested
                    
                    # Run GA
                    success, metrics = run_evaluation_with_budget(
                        checkpoint_path, "gradient_ascent", budget,
                        args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                        args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                        args.dataset_use_hf == "true", args.dataset_seed, temp_dir
                    )
                    if success:
                        ga_results[budget] = metrics
                    
                    # Run ES
                    success, metrics = run_evaluation_with_budget(
                        checkpoint_path, "evolutionary_search", budget,
                        args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                        args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                        args.dataset_use_hf == "true", args.dataset_seed, temp_dir
                    )
                    if success:
                        es_results[budget] = metrics
                
                checkpoint_results.append({
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_path": checkpoint_path,
                    "ga_results": ga_results,
                    "es_results": es_results
                })
                
                # Create heatmaps for this checkpoint
                print(f"Creating heatmaps for checkpoint {checkpoint_name}...")
                heatmap_files = create_heatmaps(
                    [checkpoint_results[-1]], budgets, args.output_dir
                )
                
                # Upload to W&B
                for heatmap_file in heatmap_files:
                    wandb.log({
                        f"heatmap_{checkpoint_name}": wandb.Image(heatmap_file)
                    })
                
                print(f"Uploaded {len(heatmap_files)} heatmaps to W&B")
            else:
                print(f"Skipping checkpoint {checkpoint_name} due to max budget test failure")
    
    # Create final summary heatmaps
    if checkpoint_results:
        print("Creating summary heatmaps...")
        summary_heatmaps = create_heatmaps(checkpoint_results, budgets, args.output_dir)
        
        for heatmap_file in summary_heatmaps:
            wandb.log({
                "summary_heatmap": wandb.Image(heatmap_file)
            })
    
    print(f"Completed evaluation of {len(checkpoint_results)} checkpoints")
    wandb.finish()


if __name__ == "__main__":
    main()
