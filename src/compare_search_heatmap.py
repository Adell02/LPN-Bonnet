#!/usr/bin/env python3
"""
Compare search methods (GA vs ES) across different checkpoints and create comprehensive heatmaps.

This script evaluates multiple checkpoints from a W&B run, runs both gradient ascent
and evolutionary search with a specified budget, extracts intermediate metrics,
and creates comprehensive heatmaps with:
- Y-axis: Budget steps (0 to max_budget)
- X-axis: Training checkpoints (1, 2, 3, ... N_checkpoints)
- Color: Loss/accuracy/difference values

The comprehensive heatmaps show the complete evolution across both budget and checkpoint dimensions,
creating matrices of shape (N_budget_steps, N_checkpoints) for each metric.

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
import math

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from visualization import visualize_loss_difference_heatmap


def get_checkpoints_from_run(run_name: str, project: str, max_checkpoints: int, strategy: str, max_checkpoint: Optional[str] = None) -> List[str]:
    """Get checkpoint artifact paths from a W&B run."""
    api = wandb.Api()
    
    # Try to find the run in the specified project first
    try:
        run = api.run(f"{project}/{run_name}")
    except Exception as e:
        print(f"Could not find run {run_name} in project {project}: {e}")
        print("Trying to find run in other projects...")
        
        # Search for the run across all accessible projects
        found_run = None
        for proj in api.projects():
            try:
                potential_run = api.run(f"{proj.entity}/{proj.name}/{run_name}")
                found_run = potential_run
                print(f"Found run {run_name} in project {proj.entity}/{proj.name}")
                break
            except:
                continue
        
        if found_run is None:
            raise ValueError(f"Could not find run {run_name} in any accessible project")
        run = found_run
    
    # Helper to parse wandb artifact version like 'v15' -> 15
    def _parse_version(ver: Optional[str]) -> Optional[int]:
        if ver is None:
            return None
        try:
            if isinstance(ver, str):
                if ver.startswith('v'):
                    return int(ver[1:])
                # try name suffix like '...:v12'
                if ':' in ver:
                    part = ver.split(':')[-1]
                    if part.startswith('v'):
                        return int(part[1:])
                return int(ver)
            return int(ver)
        except Exception:
            return None

    # Get all checkpoint artifacts
    artifacts = []
    for artifact in run.logged_artifacts():
        if "checkpoint" in artifact.name:
            artifacts.append(artifact)
    
    if not artifacts:
        raise ValueError(f"No checkpoint artifacts found in run {run_name}")
    
    # Optional filter by max_checkpoint version (e.g., 'v15')
    if max_checkpoint is not None:
        max_ver = _parse_version(max_checkpoint)
        if max_ver is not None:
            filtered = []
            for a in artifacts:
                a_ver = None
                # Prefer artifact.version if present; fallback to parse from full name
                try:
                    a_ver = _parse_version(getattr(a, 'version', None))
                except Exception:
                    a_ver = None
                if a_ver is None:
                    try:
                        a_ver = _parse_version(getattr(a, 'name', None))
                    except Exception:
                        a_ver = None
                if a_ver is None or a_ver <= max_ver:
                    filtered.append(a)
            artifacts = filtered

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
        # Format: entity/project/checkpoint_name (without version for evaluate_checkpoint.py)
        # The artifact name should already be in the correct format
        path = f"{artifact.entity}/{artifact.project}/{artifact.name}"
        checkpoint_paths.append(path)
    
    print(f"Selected {len(checkpoint_paths)} checkpoints from {len(artifacts)} total")
    print(f"Sample checkpoint path: {checkpoint_paths[0] if checkpoint_paths else 'None'}")
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
    temp_dir: str,
    granularity_mode: str = "auto"
) -> Tuple[bool, Dict[str, Any]]:
    """Run evaluation with a specific budget and extract intermediate metrics from trajectory."""
    
    # Calculate method-specific parameters (same as store_latent_search.py)
    if method == "gradient_ascent":
        ga_steps = int(np.ceil(budget / 2))  # Each step = 2 evaluations (forward + backward)
        # Ensure batch size doesn't exceed dataset length
        effective_batch_size = min(dataset_batch_size, dataset_length)
        cmd = [
            sys.executable, "src/evaluate_checkpoint.py",
            "-w", artifact_path,
            "-d", dataset_folder,
            "--dataset-length", str(dataset_length),
            "--dataset-batch-size", str(effective_batch_size),
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
        
        # Ensure batch size doesn't exceed dataset length
        effective_batch_size = min(dataset_batch_size, dataset_length)
        cmd = [
            sys.executable, "src/evaluate_checkpoint.py",
            "-w", artifact_path,
            "-d", dataset_folder,
            "--dataset-length", str(dataset_length),
            "--dataset-batch-size", str(effective_batch_size),
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
        if method == "gradient_ascent":
            print(f"Running {method} with budget {budget} (steps: {ga_steps})...")
            print(f"Using effective batch size: {effective_batch_size} (dataset_length: {dataset_length}, requested_batch_size: {dataset_batch_size})")
        else:
            print(f"Running {method} with budget {budget} (population: {pop}, generations: {gens})...")
            print(f"Using effective batch size: {effective_batch_size} (dataset_length: {dataset_length}, requested_batch_size: {dataset_batch_size})")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Always print the output for debugging
        print(f"Subprocess return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"Error running {method} with budget {budget}")
            return False, {}
        
        # Parse output to extract final metrics
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
        
        # Load trajectory data and extract intermediate metrics
        if method == "gradient_ascent":
            latents_file = os.path.join(temp_dir, f"ga_latents_{budget}.npz")
        else:
            latents_file = os.path.join(temp_dir, f"es_latents_{budget}.npz")
        print(f"Looking for trajectory file: {latents_file}")
        
        # Debug: list all files in temp directory
        print(f"Files in temp directory: {os.listdir(temp_dir)}")
        
        if os.path.exists(latents_file):
            try:
                data = np.load(latents_file)
                print(f"Loaded trajectory data with keys: {list(data.keys())}")
                
                # Extract intermediate losses (same logic as store_latent_search.py)
                if method == "gradient_ascent":
                    # For GA, look for losses_per_sample or trajectory_losses
                    if "ga_losses_per_sample" in data:
                        losses_per_sample = np.array(data["ga_losses_per_sample"])
                        print(f"Found ga_losses_per_sample with shape: {losses_per_sample.shape}")
                        if losses_per_sample.ndim >= 2:
                            # Take mean across samples to get trajectory values
                            trajectory_losses = np.mean(losses_per_sample, axis=0)
                            metrics["losses"] = trajectory_losses
                            print(f"Extracted GA trajectory losses: {trajectory_losses.shape}")
                    elif "ga_trajectory_losses" in data:
                        metrics["losses"] = np.array(data["ga_trajectory_losses"])
                        print(f"Extracted GA trajectory losses: {metrics['losses'].shape}")
                    elif "ga_log_probs" in data:
                        # Convert log_probs to losses
                        log_probs = np.array(data["ga_log_probs"])
                        print(f"Found ga_log_probs with shape: {log_probs.shape}")
                        if log_probs.ndim == 4:  # (B, C, T, S)
                            simple_scores = log_probs.mean(axis=(0, 1))
                            metrics["losses"] = -simple_scores
                            print(f"Extracted GA losses from log_probs: {metrics['losses'].shape}")
                    else:
                        print("No GA trajectory data found in expected keys")
                
                elif method == "evolutionary_search":
                    # For ES, look for generation_losses or best_losses_per_generation
                    if "es_generation_losses" in data:
                        metrics["losses"] = np.array(data["es_generation_losses"]).reshape(-1)
                        print(f"Extracted ES generation losses: {metrics['losses'].shape}")
                    elif "es_best_losses_per_generation" in data:
                        metrics["losses"] = np.array(data["es_best_losses_per_generation"]).reshape(-1)
                        print(f"Extracted ES best losses per generation: {metrics['losses'].shape}")
                    elif "es_all_losses" in data:
                        metrics["losses"] = np.array(data["es_all_losses"]).reshape(-1)
                        print(f"Extracted ES all losses: {metrics['losses'].shape}")
                    else:
                        print("No ES trajectory data found in expected keys")
                
                # Extract accuracies if available
                if f"{method}_accuracies" in data:
                    metrics["accuracies"] = np.array(data[f"{method}_accuracies"])
                elif f"{method}_scores" in data:
                    # Convert scores to accuracies if needed
                    scores = np.array(data[f"{method}_scores"])
                    metrics["accuracies"] = scores  # Assuming scores are already accuracies
                
                # Extract steps/budget information
                if f"{method}_budget" in data:
                    metrics["budget"] = np.array(data[f"{method}_budget"])
                    print(f"Extracted {method} budget trajectory: {metrics['budget'].shape}")
                elif f"{method}_steps" in data:
                    metrics["budget"] = np.array(data[f"{method}_steps"])
                    print(f"Extracted {method} steps as budget: {metrics['budget'].shape}")
                else:
                    # Create budget steps based on method and granularity mode
                    num_steps = len(metrics.get("losses", []))
                    if num_steps > 0:
                        if method == "gradient_ascent":
                            if granularity_mode == "match_es":
                                # Reduce GA granularity to match ES (sample every few steps)
                                # Calculate ES granularity first
                                pop = int(np.sqrt(budget))
                                gens = budget // pop
                                es_granularity = max(1, gens)  # Number of ES data points
                                
                                # Sample GA data to match ES granularity
                                if num_steps > es_granularity:
                                    step_indices = np.linspace(0, num_steps - 1, es_granularity, dtype=int)
                                    metrics["budget"] = step_indices * 2  # Each GA step = 2 evaluations
                                else:
                                    metrics["budget"] = np.arange(0, num_steps, 1) * 2
                            else:
                                # Default GA: each step = 2 evaluations, so budget steps are [0, 2, 4, 6, ...]
                                metrics["budget"] = np.arange(0, num_steps, 1) * 2
                        else:  # evolutionary_search
                            if granularity_mode == "match_ga":
                                # Reduce ES granularity to match GA (sample fewer points)
                                ga_granularity = num_steps  # GA has this many data points
                                if num_steps > ga_granularity:
                                    step_indices = np.linspace(0, num_steps - 1, ga_granularity, dtype=int)
                                    metrics["budget"] = np.linspace(0, budget, len(step_indices), dtype=int)
                                else:
                                    metrics["budget"] = np.linspace(0, budget, num_steps, dtype=int)
                            elif granularity_mode == "high_res":
                                # Maximum granularity: create budget steps for every evaluation
                                metrics["budget"] = np.linspace(0, budget, num_steps, dtype=int)
                            else:  # auto or match_es
                                # Create more granular budget steps to match GA granularity
                                # Calculate population size from budget and generations
                                pop = int(np.sqrt(budget))
                                gens = budget // pop
                                
                                # Create per-evaluation budget steps for more granularity
                                if gens > 1:
                                    # Create budget steps that go from 0 to budget with pop_size increments
                                    # This gives us budget steps like [0, pop, 2*pop, 3*pop, ..., budget]
                                    budget_steps = np.arange(0, budget + 1, pop)
                                    # If we have more loss values than budget steps, interpolate
                                    if num_steps > len(budget_steps):
                                        # Interpolate to match the number of loss values
                                        metrics["budget"] = np.linspace(0, budget, num_steps, dtype=int)
                                    else:
                                        # Use the calculated budget steps, truncating if necessary
                                        metrics["budget"] = budget_steps[:num_steps]
                                else:
                                    # Single generation case - create uniform distribution
                                    metrics["budget"] = np.linspace(0, budget, num_steps, dtype=int)
                    else:
                        metrics["budget"] = np.array([])
                    print(f"Created {method} budget trajectory: {metrics['budget'].shape}")
                    
                    # Ensure budget values are properly aligned and monotonic
                    if len(metrics.get("budget", [])) > 0:
                        budget_array = np.array(metrics["budget"])
                        # Ensure budgets are non-negative and monotonic
                        budget_array = np.maximum(budget_array, 0)
                        # Sort to ensure monotonicity
                        sort_idx = np.argsort(budget_array)
                        metrics["budget"] = budget_array[sort_idx]
                        # Also sort the corresponding losses
                        if "losses" in metrics and len(metrics["losses"]) == len(budget_array):
                            metrics["losses"] = np.array(metrics["losses"])[sort_idx]
                        if "accuracies" in metrics and len(metrics["accuracies"]) == len(budget_array):
                            metrics["accuracies"] = np.array(metrics["accuracies"])[sort_idx]
                
            except Exception as e:
                print(f"Warning: Could not load trajectory data: {e}")
        else:
            print(f"Trajectory file not found: {latents_file}")
        
        return True, metrics
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running {method} with budget {budget}")
        return False, {}
    except Exception as e:
        print(f"Error running {method} with budget {budget}: {e}")
        return False, {}


def create_heatmaps(
    checkpoint_results: List[Dict[str, Any]],
    max_budget: int,
    output_dir: str,
    progressive: bool = True
) -> List[str]:
    """Create comprehensive 2D heatmaps from checkpoint results.
    
    Creates heatmaps with:
    - Y-axis: Budget steps (0 to max_budget)
    - X-axis: Training checkpoints (1, 2, 3, ... N_checkpoints)
    - Color: Loss/accuracy/difference values
    
    Args:
        checkpoint_results: List of checkpoint data with GA and ES results
        max_budget: Maximum budget used for evaluation
        output_dir: Directory to save heatmap files
        progressive: If True, create comprehensive heatmaps; if False, create individual heatmaps
    """
    
    heatmap_files = []
    
    def _two_sided_sign_test_pvalue(diffs: np.ndarray) -> float:
        # Remove zeros (ties)
        clean = diffs[np.isfinite(diffs) & (diffs != 0.0)]
        n = int(clean.size)
        if n == 0:
            return float('nan')
        k = int(np.sum(clean > 0))
        # Binomial tail with p=0.5
        # Compute CDF up to k and upper tail from k to n
        # two-sided p = 2 * min( P(X<=k), P(X>=k) )
        denom = 2.0 ** n
        cdf_le_k = sum(math.comb(n, i) for i in range(0, k + 1)) / denom
        cdf_ge_k = sum(math.comb(n, i) for i in range(k, n + 1)) / denom
        p = 2.0 * min(cdf_le_k, cdf_ge_k)
        return min(1.0, max(0.0, p))
    
    def _compute_metric_pvalues(ga_losses: np.ndarray, ga_budget: np.ndarray, 
                               es_losses: np.ndarray, es_budget: np.ndarray,
                               checkpoint_name: str, checkpoint_idx: int) -> dict:
        """Compute p-values for all metrics (GA, ES, differential) and upload to W&B."""
        pvalues = {}
        
        # Find common budget points for comparison
        common_b = np.intersect1d(np.asarray(ga_budget, dtype=int), np.asarray(es_budget, dtype=int))
        if common_b.size == 0:
            print(f"No common budget points found for checkpoint {checkpoint_name}")
            return pvalues
        
        # Create mapping from budget to loss values
        ga_map = {int(b): float(ga_losses[idx]) for idx, b in enumerate(ga_budget)}
        es_map = {int(b): float(es_losses[idx]) for idx, b in enumerate(es_budget)}
        
        # Get loss values at common budget points
        ga_common = [ga_map[int(b)] for b in common_b if int(b) in ga_map]
        es_common = [es_map[int(b)] for b in common_b if int(b) in es_map]
        
        if len(ga_common) == 0 or len(es_common) == 0:
            print(f"No valid loss values at common budget points for checkpoint {checkpoint_name}")
            return pvalues
        
        # 1. Differential p-value (ES - GA)
        diffs = [es_common[i] - ga_common[i] for i in range(len(ga_common))]
        if len(diffs) > 0:
            diff_pval = _two_sided_sign_test_pvalue(np.asarray(diffs, dtype=float))
            pvalues["ga_es_differential_p_value"] = diff_pval
            wandb.log({
                "ga_es_differential_p_value": diff_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        # 2. GA improvement p-value (comparing GA losses to baseline/mean)
        # Test if GA losses are significantly different from their mean
        ga_array = np.array(ga_common)
        ga_mean = np.mean(ga_array)
        ga_diffs = ga_array - ga_mean
        if len(ga_diffs) > 0:
            ga_pval = _two_sided_sign_test_pvalue(ga_diffs)
            pvalues["ga_improvement_p_value"] = ga_pval
            wandb.log({
                "ga_improvement_p_value": ga_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        # 3. ES improvement p-value (comparing ES losses to baseline/mean)
        # Test if ES losses are significantly different from their mean
        es_array = np.array(es_common)
        es_mean = np.mean(es_array)
        es_diffs = es_array - es_mean
        if len(es_diffs) > 0:
            es_pval = _two_sided_sign_test_pvalue(es_diffs)
            pvalues["es_improvement_p_value"] = es_pval
            wandb.log({
                "es_improvement_p_value": es_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        # 4. GA vs ES relative performance p-value (alternative formulation)
        # Test if GA is consistently better/worse than ES
        ga_vs_es_diffs = [ga_common[i] - es_common[i] for i in range(len(ga_common))]
        if len(ga_vs_es_diffs) > 0:
            ga_vs_es_pval = _two_sided_sign_test_pvalue(np.asarray(ga_vs_es_diffs, dtype=float))
            pvalues["ga_vs_es_relative_p_value"] = ga_vs_es_pval
            wandb.log({
                "ga_vs_es_relative_p_value": ga_vs_es_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        print(f"Computed p-values for checkpoint {checkpoint_name}: {pvalues}")
        return pvalues
    
    def _compute_accuracy_pvalues(ga_accuracies: np.ndarray, ga_budget: np.ndarray, 
                                  es_accuracies: np.ndarray, es_budget: np.ndarray,
                                  checkpoint_name: str, checkpoint_idx: int) -> dict:
        """Compute p-values for accuracy metrics and upload to W&B."""
        pvalues = {}
        
        # Find common budget points for comparison
        common_b = np.intersect1d(np.asarray(ga_budget, dtype=int), np.asarray(es_budget, dtype=int))
        if common_b.size == 0:
            print(f"No common budget points found for accuracy in checkpoint {checkpoint_name}")
            return pvalues
        
        # Create mapping from budget to accuracy values
        ga_map = {int(b): float(ga_accuracies[idx]) for idx, b in enumerate(ga_budget)}
        es_map = {int(b): float(es_accuracies[idx]) for idx, b in enumerate(es_budget)}
        
        # Get accuracy values at common budget points
        ga_common = [ga_map[int(b)] for b in common_b if int(b) in ga_map]
        es_common = [es_map[int(b)] for b in common_b if int(b) in es_map]
        
        if len(ga_common) == 0 or len(es_common) == 0:
            print(f"No valid accuracy values at common budget points for checkpoint {checkpoint_name}")
            return pvalues
        
        # 1. Differential accuracy p-value (ES - GA)
        acc_diffs = [es_common[i] - ga_common[i] for i in range(len(ga_common))]
        if len(acc_diffs) > 0:
            acc_diff_pval = _two_sided_sign_test_pvalue(np.asarray(acc_diffs, dtype=float))
            pvalues["ga_es_accuracy_differential_p_value"] = acc_diff_pval
            wandb.log({
                "ga_es_accuracy_differential_p_value": acc_diff_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        # 2. GA accuracy improvement p-value
        ga_acc_array = np.array(ga_common)
        ga_acc_mean = np.mean(ga_acc_array)
        ga_acc_diffs = ga_acc_array - ga_acc_mean
        if len(ga_acc_diffs) > 0:
            ga_acc_pval = _two_sided_sign_test_pvalue(ga_acc_diffs)
            pvalues["ga_accuracy_improvement_p_value"] = ga_acc_pval
            wandb.log({
                "ga_accuracy_improvement_p_value": ga_acc_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        # 3. ES accuracy improvement p-value
        es_acc_array = np.array(es_common)
        es_acc_mean = np.mean(es_acc_array)
        es_acc_diffs = es_acc_array - es_acc_mean
        if len(es_acc_diffs) > 0:
            es_acc_pval = _two_sided_sign_test_pvalue(es_acc_diffs)
            pvalues["es_accuracy_improvement_p_value"] = es_acc_pval
            wandb.log({
                "es_accuracy_improvement_p_value": es_acc_pval,
                "checkpoint_name": checkpoint_name
            }, step=checkpoint_idx)
        
        print(f"Computed accuracy p-values for checkpoint {checkpoint_name}: {pvalues}")
        return pvalues
    
    def _create_unified_budget_grid(ga_budgets: List[np.ndarray], es_budgets: List[np.ndarray], 
                                  max_budget: int, target_granularity: int = 100) -> np.ndarray:
        """
        Create a unified budget grid that ensures both GA and ES have the same granularity.
        
        Args:
            ga_budgets: List of GA budget arrays
            es_budgets: List of ES budget arrays  
            max_budget: Maximum budget value
            target_granularity: Target number of budget points
            
        Returns:
            Unified budget grid with consistent granularity
        """
        # Collect all unique budget values from both methods
        all_budgets = set()
        
        for ga_budget in ga_budgets:
            if len(ga_budget) > 0:
                all_budgets.update(ga_budget)
        
        for es_budget in es_budgets:
            if len(es_budget) > 0:
                all_budgets.update(es_budget)
        
        if not all_budgets:
            # Fallback to simple linear grid
            return np.linspace(0, max_budget, target_granularity, dtype=int)
        
        # Convert to sorted array
        unique_budgets = np.array(sorted(all_budgets))
        
        # If we have too many points, sample them
        if len(unique_budgets) > target_granularity:
            # Sample to target granularity while preserving key points (0, max_budget)
            indices = np.linspace(0, len(unique_budgets) - 1, target_granularity, dtype=int)
            unified_grid = unique_budgets[indices]
        else:
            # Use all unique budget points
            unified_grid = unique_budgets
        
        # Ensure 0 and max_budget are included
        if 0 not in unified_grid:
            unified_grid = np.concatenate([[0], unified_grid])
        if max_budget not in unified_grid:
            unified_grid = np.concatenate([unified_grid, [max_budget]])
        
        # Sort and remove duplicates
        unified_grid = np.unique(unified_grid)
        
        print(f"Created unified budget grid with {len(unified_grid)} points: {unified_grid[:5]}...{unified_grid[-5:]}")
        return unified_grid
    
    # Aggregate data across all checkpoints
    all_checkpoints = []
    all_ga_losses = []
    all_es_losses = []
    all_ga_accuracies = []
    all_es_accuracies = []
    all_ga_budgets = []
    all_es_budgets = []
    
    for i, checkpoint_data in enumerate(checkpoint_results):
        checkpoint_name = checkpoint_data["checkpoint_name"]
        ga_results = checkpoint_data["ga_results"]
        es_results = checkpoint_data["es_results"]
        
        # Extract losses, accuracies and budget trajectories for both methods
        ga_losses = None
        es_losses = None
        ga_accuracies = None
        es_accuracies = None
        ga_budget = None
        es_budget = None
        
        if max_budget in ga_results and "losses" in ga_results[max_budget]:
            ga_losses = ga_results[max_budget]["losses"]
            ga_budget = ga_results[max_budget].get("budget", np.arange(len(ga_losses)) * 2)
            ga_accuracies = ga_results[max_budget].get("accuracies", None)
            print(f"GA losses shape: {ga_losses.shape}, budget shape: {ga_budget.shape}")
                
        if max_budget in es_results and "losses" in es_results[max_budget]:
            es_losses = es_results[max_budget]["losses"]
            es_budget = es_results[max_budget].get("budget", np.arange(len(es_losses)) * 4)
            es_accuracies = es_results[max_budget].get("accuracies", None)
            print(f"ES losses shape: {es_losses.shape}, budget shape: {es_budget.shape}")
        
        # Store data for aggregation
        all_checkpoints.append(checkpoint_name)
        if ga_losses is not None:
            all_ga_losses.append(ga_losses)
            all_ga_budgets.append(ga_budget)
            if ga_accuracies is not None:
                all_ga_accuracies.append(ga_accuracies)
        if es_losses is not None:
            all_es_losses.append(es_losses)
            all_es_budgets.append(es_budget)
            if es_accuracies is not None:
                all_es_accuracies.append(es_accuracies)
    
    if progressive:
        # Create comprehensive heatmaps with budget on Y-axis, checkpoints on X-axis
        print(f"Creating comprehensive heatmaps for {len(all_checkpoints)} checkpoints...")
        
        # Create unified budget grid that ensures both methods have the same granularity
        # Find the maximum budget across all methods and checkpoints
        max_budget_found = 0
        for ga_budget in all_ga_budgets:
            if ga_budget is not None and len(ga_budget) > 0:
                max_budget_found = max(max_budget_found, int(ga_budget[-1]))
        for es_budget in all_es_budgets:
            if es_budget is not None and len(es_budget) > 0:
                max_budget_found = max(max_budget_found, int(es_budget[-1]))
        
        # Create unified budget grid with consistent granularity
        uniform_budget_grid = _create_unified_budget_grid(all_ga_budgets, all_es_budgets, max_budget_found)
        
        # Create comprehensive GA loss heatmap
        if all_ga_losses:
            ga_loss_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)

            for i, (ga_losses, ga_budget) in enumerate(zip(all_ga_losses, all_ga_budgets)):
                # Use linear interpolation for smooth transitions
                if len(ga_losses) > 0 and len(ga_budget) > 0:
                    # Ensure budgets are sorted for interpolation
                    sort_idx = np.argsort(ga_budget)
                    ga_budget_sorted = ga_budget[sort_idx]
                    ga_losses_sorted = ga_losses[sort_idx]
                    
                    # Remove duplicates and ensure monotonic budgets
                    unique_mask = np.concatenate(([True], np.diff(ga_budget_sorted) > 0))
                    ga_budget_unique = ga_budget_sorted[unique_mask]
                    ga_losses_unique = ga_losses_sorted[unique_mask]
                    
                    if len(ga_budget_unique) > 1:
                        # Linear interpolation for smooth transitions
                        ga_interpolated = np.interp(uniform_budget_grid, ga_budget_unique, ga_losses_unique)
                        ga_loss_matrix[:, i] = ga_interpolated
                    elif len(ga_budget_unique) == 1:
                        # Single point - fill with constant value
                        ga_loss_matrix[:, i] = ga_losses_unique[0]

            # Diagnostics: report GA loss range
            ga_finite = np.isfinite(ga_loss_matrix)
            if np.any(ga_finite):
                ga_min = float(np.nanmin(ga_loss_matrix[ga_finite]))
                ga_max = float(np.nanmax(ga_loss_matrix[ga_finite]))
                print(f"GA comprehensive loss range: min={ga_min:.4f}, max={ga_max:.4f}")

            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, ga_loss_matrix,
                method_A_name="GA", method_B_name="GA",
                symmetric=False, descending_colorbar=False
            )
            ga_file = os.path.join(output_dir, "ga_comprehensive_loss_heatmap.png")
            fig.savefig(ga_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(ga_file)

        # Create comprehensive ES loss heatmap
        if all_es_losses:
            es_loss_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)

            for i, (es_losses, es_budget) in enumerate(zip(all_es_losses, all_es_budgets)):
                # Use linear interpolation for smooth transitions
                if len(es_losses) > 0 and len(es_budget) > 0:
                    # Ensure budgets are sorted for interpolation
                    sort_idx = np.argsort(es_budget)
                    es_budget_sorted = es_budget[sort_idx]
                    es_losses_sorted = es_losses[sort_idx]
                    
                    # Remove duplicates and ensure monotonic budgets
                    unique_mask = np.concatenate(([True], np.diff(es_budget_sorted) > 0))
                    es_budget_unique = es_budget_sorted[unique_mask]
                    es_losses_unique = es_losses_sorted[unique_mask]
                    
                    if len(es_budget_unique) > 1:
                        # Linear interpolation for smooth transitions
                        es_interpolated = np.interp(uniform_budget_grid, es_budget_unique, es_losses_unique)
                        es_loss_matrix[:, i] = es_interpolated
                    elif len(es_budget_unique) == 1:
                        # Single point - fill with constant value
                        es_loss_matrix[:, i] = es_losses_unique[0]

            # Diagnostics: report ES loss range
            es_finite = np.isfinite(es_loss_matrix)
            if np.any(es_finite):
                es_min = float(np.nanmin(es_loss_matrix[es_finite]))
                es_max = float(np.nanmax(es_loss_matrix[es_finite]))
                print(f"ES comprehensive loss range: min={es_min:.4f}, max={es_max:.4f}")

            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, es_loss_matrix,
                method_A_name="ES", method_B_name="ES",
                symmetric=False, descending_colorbar=False
            )
            es_file = os.path.join(output_dir, "es_comprehensive_loss_heatmap.png")
            fig.savefig(es_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(es_file)
        
        # Create comprehensive differential heatmap (ES - GA)
        if (all_ga_losses and all_es_losses and 
            len(all_ga_losses) == len(all_es_losses)):
            # Use the already interpolated matrices for consistent granularity
            if 'ga_loss_matrix' in locals() and 'es_loss_matrix' in locals():
                # Use the already interpolated matrices
                ga_matrix = ga_loss_matrix
                es_matrix = es_loss_matrix
            else:
                # Fallback: create aligned matrices using linear interpolation
                ga_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)
                es_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)

                # Fill GA matrix with linear interpolation
                for i, (ga_losses, ga_budget) in enumerate(zip(all_ga_losses, all_ga_budgets)):
                    if len(ga_losses) > 0 and len(ga_budget) > 0:
                        sort_idx = np.argsort(ga_budget)
                        ga_budget_sorted = ga_budget[sort_idx]
                        ga_losses_sorted = ga_losses[sort_idx]
                        
                        unique_mask = np.concatenate(([True], np.diff(ga_budget_sorted) > 0))
                        ga_budget_unique = ga_budget_sorted[unique_mask]
                        ga_losses_unique = ga_losses_sorted[unique_mask]
                        
                        if len(ga_budget_unique) > 1:
                            ga_interpolated = np.interp(uniform_budget_grid, ga_budget_unique, ga_losses_unique)
                            ga_matrix[:, i] = ga_interpolated
                        elif len(ga_budget_unique) == 1:
                            ga_matrix[:, i] = ga_losses_unique[0]

                # Fill ES matrix with linear interpolation
                for i, (es_losses, es_budget) in enumerate(zip(all_es_losses, all_es_budgets)):
                    if len(es_losses) > 0 and len(es_budget) > 0:
                        sort_idx = np.argsort(es_budget)
                        es_budget_sorted = es_budget[sort_idx]
                        es_losses_sorted = es_losses[sort_idx]
                        
                        unique_mask = np.concatenate(([True], np.diff(es_budget_sorted) > 0))
                        es_budget_unique = es_budget_sorted[unique_mask]
                        es_losses_unique = es_losses_sorted[unique_mask]
                        
                        if len(es_budget_unique) > 1:
                            es_interpolated = np.interp(uniform_budget_grid, es_budget_unique, es_losses_unique)
                            es_matrix[:, i] = es_interpolated
                        elif len(es_budget_unique) == 1:
                            es_matrix[:, i] = es_losses_unique[0]

            # Calculate difference (GA - ES) with proper budget matching (aligns with binary convention)
            diff_matrix = ga_matrix - es_matrix

            # Diagnostics: report DIFF range
            diff_finite = np.isfinite(diff_matrix)
            if np.any(diff_finite):
                diff_min = float(np.nanmin(diff_matrix[diff_finite]))
                diff_max = float(np.nanmax(diff_matrix[diff_finite]))
                print(f"Differential (ES-GA) loss range: min={diff_min:.4f}, max={diff_max:.4f}")

            # Compute and log comprehensive p-values for all metrics
            for i, (ga_losses, ga_budget, es_losses, es_budget) in enumerate(
                zip(all_ga_losses, all_ga_budgets, all_es_losses, all_es_budgets)
            ):
                _compute_metric_pvalues(
                    ga_losses, ga_budget, es_losses, es_budget,
                    all_checkpoints[i], i
                )

            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, diff_matrix,
                method_A_name="GA", method_B_name="ES",
                symmetric=True
            )
            diff_file = os.path.join(output_dir, "differential_comprehensive_loss_heatmap.png")
            fig.savefig(diff_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(diff_file)
            
            # Create binary differential heatmap (1: GA better, 0: same, -1: ES better)
            binary_diff_matrix = np.full_like(diff_matrix, np.nan)
            finite_mask = np.isfinite(diff_matrix)
            binary_diff_matrix[finite_mask] = np.where(
                diff_matrix[finite_mask] > 0, 1,    # GA has lower loss (GA - ES > 0)
                np.where(diff_matrix[finite_mask] < 0, -1, 0)  # ES has lower loss (GA - ES < 0), or same (== 0)
            )
            
            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, binary_diff_matrix,
                method_A_name="GA", method_B_name="ES",
                symmetric=True
            )
            binary_diff_file = os.path.join(output_dir, "binary_differential_loss_heatmap.png")
            fig.savefig(binary_diff_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(binary_diff_file)
        
        # Create comprehensive accuracy heatmaps if available
        if all_ga_accuracies and all_es_accuracies:
            # GA accuracy heatmap
            ga_acc_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)

            for i, (ga_acc, ga_budget) in enumerate(zip(all_ga_accuracies, all_ga_budgets)):
                # Use linear interpolation for smooth transitions
                if len(ga_acc) > 0 and len(ga_budget) > 0:
                    sort_idx = np.argsort(ga_budget)
                    ga_budget_sorted = ga_budget[sort_idx]
                    ga_acc_sorted = ga_acc[sort_idx]
                    
                    unique_mask = np.concatenate(([True], np.diff(ga_budget_sorted) > 0))
                    ga_budget_unique = ga_budget_sorted[unique_mask]
                    ga_acc_unique = ga_acc_sorted[unique_mask]
                    
                    if len(ga_budget_unique) > 1:
                        ga_acc_interpolated = np.interp(uniform_budget_grid, ga_budget_unique, ga_acc_unique)
                        ga_acc_matrix[:, i] = ga_acc_interpolated
                    elif len(ga_budget_unique) == 1:
                        ga_acc_matrix[:, i] = ga_acc_unique[0]

            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, ga_acc_matrix,
                method_A_name="GA", method_B_name="GA"
            )
            ga_acc_file = os.path.join(output_dir, "ga_comprehensive_accuracy_heatmap.png")
            fig.savefig(ga_acc_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(ga_acc_file)
            
            # ES accuracy heatmap
            es_acc_matrix = np.full((len(uniform_budget_grid), len(all_checkpoints)), np.nan)

            for i, (es_acc, es_budget) in enumerate(zip(all_es_accuracies, all_es_budgets)):
                # Use linear interpolation for smooth transitions
                if len(es_acc) > 0 and len(es_budget) > 0:
                    sort_idx = np.argsort(es_budget)
                    es_budget_sorted = es_budget[sort_idx]
                    es_acc_sorted = es_acc[sort_idx]
                    
                    unique_mask = np.concatenate(([True], np.diff(es_budget_sorted) > 0))
                    es_budget_unique = es_budget_sorted[unique_mask]
                    es_acc_unique = es_acc_sorted[unique_mask]
                    
                    if len(es_budget_unique) > 1:
                        es_acc_interpolated = np.interp(uniform_budget_grid, es_budget_unique, es_acc_unique)
                        es_acc_matrix[:, i] = es_acc_interpolated
                    elif len(es_budget_unique) == 1:
                        es_acc_matrix[:, i] = es_acc_unique[0]

            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, es_acc_matrix,
                method_A_name="ES", method_B_name="ES"
            )
            es_acc_file = os.path.join(output_dir, "es_comprehensive_accuracy_heatmap.png")
            fig.savefig(es_acc_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(es_acc_file)
            
            # Differential accuracy heatmap (ES - GA)
            # Calculate accuracy difference (ES - GA)
            acc_diff_matrix = es_acc_matrix - ga_acc_matrix
            
            # Compute and log accuracy p-values for all checkpoints
            for i, (ga_acc, ga_budget, es_acc, es_budget) in enumerate(
                zip(all_ga_accuracies, all_ga_budgets, all_es_accuracies, all_es_budgets)
            ):
                _compute_accuracy_pvalues(
                    ga_acc, ga_budget, es_acc, es_budget,
                    all_checkpoints[i], i
                )
            
            checkpoint_indices = np.arange(len(all_checkpoints))

            fig = visualize_loss_difference_heatmap(
                checkpoint_indices, uniform_budget_grid, acc_diff_matrix,
                method_A_name="GA", method_B_name="ES"
            )
            acc_diff_file = os.path.join(output_dir, "differential_comprehensive_accuracy_heatmap.png")
            fig.savefig(acc_diff_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            heatmap_files.append(acc_diff_file)
    
    else:
        # Original behavior: create individual heatmaps for each checkpoint
        for i, checkpoint_name in enumerate(all_checkpoints):
            # GA individual heatmap
            if i < len(all_ga_losses) and len(all_ga_losses[i]) > 0:
                ga_losses = all_ga_losses[i]
                ga_budget = all_ga_budgets[i]

                # Create 2D matrix with budgets on the first axis
                # For a single checkpoint the matrix shape should be [budget_steps, 1]
                ga_loss_matrix = ga_losses.reshape(-1, 1)  # [num_budget_steps, 1]
                checkpoint_indices = np.array([i])  # [1] - single checkpoint
                
                fig = visualize_loss_difference_heatmap(
                    checkpoint_indices, ga_budget, ga_loss_matrix,
                    method_A_name="GA", method_B_name="GA",
                    symmetric=False, descending_colorbar=False
                )
                ga_file = os.path.join(output_dir, f"ga_individual_{checkpoint_name}.png")
                fig.savefig(ga_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                heatmap_files.append(ga_file)
            
            # ES individual heatmap
            if i < len(all_es_losses) and len(all_es_losses[i]) > 0:
                es_losses = all_es_losses[i]
                es_budget = all_es_budgets[i]

                # Create 2D matrix with budgets on the first axis
                es_loss_matrix = es_losses.reshape(-1, 1)  # [num_budget_steps, 1]
                checkpoint_indices = np.array([i])  # [1] - single checkpoint
                
                fig = visualize_loss_difference_heatmap(
                    checkpoint_indices, es_budget, es_loss_matrix,
                    method_A_name="ES", method_B_name="ES",
                    symmetric=False, descending_colorbar=False
                )
                es_file = os.path.join(output_dir, f"es_individual_{checkpoint_name}.png")
                fig.savefig(es_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                heatmap_files.append(es_file)
        
        # Create differential heatmap (GA - ES) for each checkpoint
        for i, checkpoint_name in enumerate(all_checkpoints):
            if (i < len(all_ga_losses) and i < len(all_es_losses) and 
                len(all_ga_losses[i]) > 0 and len(all_es_losses[i]) > 0):
                
                ga_losses = all_ga_losses[i]
                es_losses = all_es_losses[i]
                ga_budget = all_ga_budgets[i]
                es_budget = all_es_budgets[i]
                
                # Align the loss trajectories to the same length
                min_len = min(len(ga_losses), len(es_losses))
                # Truncate to same length
                ga_losses_aligned = ga_losses[:min_len]
                es_losses_aligned = es_losses[:min_len]
                ga_budget_aligned = ga_budget[:min_len]
                es_budget_aligned = es_budget[:min_len]
                # Calculate difference (GA - ES) to match binary convention
                loss_diff = ga_losses_aligned - es_losses_aligned
                # Use the average budget trajectory
                avg_budget = (ga_budget_aligned + es_budget_aligned) / 2

                # Compute and log comprehensive p-values for all metrics
                _compute_metric_pvalues(
                    ga_losses, ga_budget, es_losses, es_budget,
                    checkpoint_name, i
                )
                
                # Create 2D matrix with budgets on the first axis
                loss_diff_matrix = loss_diff.reshape(-1, 1)  # [num_budget_steps, 1]
                checkpoint_indices = np.array([i])  # [1] - single checkpoint
                
                fig = visualize_loss_difference_heatmap(
                    checkpoint_indices, avg_budget, loss_diff_matrix,
                    method_A_name="GA", method_B_name="ES",
                    symmetric=True
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
    parser.add_argument("--project", type=str, required=True, help="W&B project name for finding the run")
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
    parser.add_argument("--max_checkpoint", type=str, default=None, help="Maximum checkpoint version to include (e.g., v15)")
    
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
    
    # Test mode
    parser.add_argument("--test_mode", action="store_true", help="Test mode: only run max budget test on first checkpoint")
    
    # Comprehensive heatmaps
    parser.add_argument("--progressive", action="store_true", default=True, help="Create comprehensive heatmaps with budget on Y-axis, checkpoints on X-axis")
    parser.add_argument("--no_progressive", action="store_true", help="Disable comprehensive heatmaps, use original individual heatmaps")
    
    # Granularity control
    parser.add_argument("--granularity_mode", type=str, default="auto", choices=["auto", "match_ga", "match_es", "high_res"], 
                       help="Control granularity: auto=improve ES granularity, match_ga=reduce GA to match ES, match_es=reduce ES to match GA, high_res=maximum granularity for both")
    
    args = parser.parse_args()
    
    # Handle progressive flag
    use_progressive = args.progressive and not args.no_progressive
    
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
    print(f"Getting checkpoints from run {args.run_name} in project {args.project}...")
    checkpoint_paths = get_checkpoints_from_run(
        args.run_name, args.project, args.max_checkpoints, args.checkpoint_strategy, args.max_checkpoint
    )
    
    # Use max budget only (like store_latent_search.py)
    max_budget = args.budget_end
    print(f"Running with max budget {max_budget} and extracting intermediate steps...")
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_results = []
        
        # Evaluate each checkpoint
        checkpoints_to_evaluate = checkpoint_paths[:1] if args.test_mode else checkpoint_paths
        for i, checkpoint_path in enumerate(checkpoints_to_evaluate):
            # Extract checkpoint name from the full path
            checkpoint_name = checkpoint_path.split("/")[-1]
            print(f"\nEvaluating checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_name}")
            
            ga_results = {}
            es_results = {}
            
            # Run GA with max budget and extract intermediate steps
            print(f"Running GA with budget {max_budget}...")
            success, metrics = run_evaluation_with_budget(
                checkpoint_path, "gradient_ascent", max_budget,
                args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                args.dataset_use_hf == "true", args.dataset_seed, temp_dir,
                args.granularity_mode
            )
            if success:
                ga_results[max_budget] = metrics
                print(f"GA completed: accuracy={metrics.get('accuracy', 'N/A')}, loss={metrics.get('loss', 'N/A')}")
                if "losses" in metrics:
                    print(f"GA trajectory: {len(metrics['losses'])} intermediate steps extracted")
            else:
                print("GA failed")
            
            # Run ES with max budget and extract intermediate steps
            print(f"Running ES with budget {max_budget}...")
            success, metrics = run_evaluation_with_budget(
                checkpoint_path, "evolutionary_search", max_budget,
                args.ga_lr, args.es_mutation_std, args.es_mutation_decay,
                args.dataset_folder, args.dataset_length, args.dataset_batch_size,
                args.dataset_use_hf == "true", args.dataset_seed, temp_dir,
                args.granularity_mode
            )
            if success:
                es_results[max_budget] = metrics
                print(f"ES completed: accuracy={metrics.get('accuracy', 'N/A')}, loss={metrics.get('loss', 'N/A')}")
                if "losses" in metrics:
                    print(f"ES trajectory: {len(metrics['losses'])} intermediate steps extracted")
            else:
                print("ES failed")
            
            # If either method succeeded, create heatmaps
            if ga_results or es_results:
                checkpoint_results.append({
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_path": checkpoint_path,
                    "ga_results": ga_results,
                    "es_results": es_results
                })
                
                # Create heatmaps using all collected checkpoints so far
                print(f"Creating heatmaps for checkpoint {checkpoint_name}...")
                heatmap_files = create_heatmaps(
                    checkpoint_results, max_budget, args.output_dir, progressive=use_progressive
                )

                # Upload to W&B using unique keys based on filename
                for heatmap_file in heatmap_files:
                    wandb.log({
                        Path(heatmap_file).stem: wandb.Image(heatmap_file)
                    })

                print(f"Uploaded {len(heatmap_files)} heatmaps to W&B")
            else:
                print(f"Skipping checkpoint {checkpoint_name} due to both methods failing")
    
    # Create final summary heatmaps
    if checkpoint_results:
        print("Creating summary heatmaps...")
        summary_heatmaps = create_heatmaps(checkpoint_results, max_budget, args.output_dir, progressive=use_progressive)
        
        for heatmap_file in summary_heatmaps:
            # Give binary differential heatmap a specific name in W&B
            if "binary_differential_loss_heatmap.png" in heatmap_file:
                wandb.log({
                    "binary_differential_loss": wandb.Image(heatmap_file)
                })
            else:
                wandb.log({
                    "summary_heatmap": wandb.Image(heatmap_file)
                })
    
    print(f"Completed evaluation of {len(checkpoint_results)} checkpoints")
    wandb.finish()


if __name__ == "__main__":
    main()
