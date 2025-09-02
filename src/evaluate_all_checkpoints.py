#!/usr/bin/env python3
"""
Evaluate all checkpoints from a specific Weights & Biases run using src/evaluate_checkpoint.py.
Runs gradient_ascent, random_search, and evolutionary_search for each checkpoint and logs results to CSV.
Supports subspace evolutionary search with localized mutation in low-dimensional subspaces.

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

7. SUBSPACE EVOLUTIONARY SEARCH (enable localized mutation in low-dimensional subspace):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --plot_methods evolutionary_search \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --es_use_subspace_mutation \
     --es_subspace_dim 32 \
     --es_ga_step_length 0.5 \
     --es_trust_region_radius 2.0 \
     --only_n_tasks 20

8. OPTIMIZATION MECHANISM TUNING (customize decay, elite size, and other parameters):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --plot_methods gradient_ascent evolutionary_search \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --ga_decay 0.95 \
     --ga_elite_size 3 \
     --ga_momentum 0.1 \
     --es_decay 0.9 \
     --es_elite_size 5 \
     --es_crossover_rate 0.7 \
     --es_tournament_size 4 \
     --only_n_tasks 10

9. ADVANCED OPTIMIZATION FEATURES (enable specialized mechanisms):
   python3 src/evaluate_all_checkpoints.py \
     --run_name "winter-fire-132" \
     --plot_methods gradient_ascent \
     --json_challenges json/arc-agi_evaluation_challenges.json \
     --json_solutions json/arc-agi_evaluation_solutions.json \
     --ga_accumulate_gradients_decoder_pairs \
     --ga_scan_gradients_latents \
     --ga_include_mean_latent \
     --ga_random_perturbation 0.01 \
     --only_n_tasks 15

10. COMPREHENSIVE OPTIMIZATION CONTROL (all parameters from store_latent_search.py):
    python3 src/evaluate_all_checkpoints.py \
      --run_name "winter-fire-132" \
      --plot_methods gradient_ascent evolutionary_search \
      --json_challenges json/arc-agi_evaluation_challenges.json \
      --json_solutions json/arc-agi_evaluation_solutions.json \
      --ga_lr 0.2 \
      --ga_steps 1000 \
      --ga_optimizer adam \
      --ga_lr_schedule true \
      --ga_lr_schedule_exponent 0.5 \
      --ga_accumulate_gradients_decoder_pairs \
      --ga_scan_gradients_latents \
      --ga_include_mean_latent \
      --es_mutation_std 0.5 \
      --es_population 100 \
      --es_generations 20 \
      --es_mutation_decay 0.95 \
      --es_elite_size 80 \
      --es_use_subspace_mutation \
      --es_subspace_dim 32 \
      --es_ga_step_length 0.5 \
      --es_trust_region_radius 2.0 \
      --track_progress \
      --background_resolution 400 \
      --background_smoothing \
      --background_knn 5 \
      --background_bandwidth_scale 1.25 \
      --background_global_mix 0.05 \
      --out_dir results/advanced_eval \
      --only_n_tasks 20

11. MULTIPLE RUNS WITH STATISTICAL ANALYSIS:
    python3 src/evaluate_all_checkpoints.py \
      --run_name "winter-fire-132" \
      --plot_methods gradient_ascent evolutionary_search \
      --json_challenges json/arc-agi_evaluation_challenges.json \
      --json_solutions json/arc-agi_evaluation_solutions.json \
      --n_samples 5 \
      --aggregate_statistics \
      --dataset_seed 42 \
      --only_n_tasks 10
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

# Import functions from store_latent_search for trajectory analysis
from store_latent_search import _extract_vals, _extract_best_per_gen, _extract_pop, Trace

def extract_losses_and_accuracies_per_budget(ga_npz_path: str, es_npz_path: str, dataset_length: int) -> Dict[str, Any]:
    """
    Extract losses and accuracies per budget point for both GA and ES methods.
    Similar to store_latent_search.py logic but focused on data extraction.
    
    Args:
        ga_npz_path: Path to GA trajectory NPZ file
        es_npz_path: Path to ES trajectory NPZ file  
        dataset_length: Number of samples evaluated
        
    Returns:
        Dictionary with extracted losses and accuracies per budget for both methods
    """
    import numpy as np
    
    print(f"[extract] Extracting losses and accuracies per budget for {dataset_length} samples...")
    
    def safe_array_to_scalar(arr: np.ndarray, default=None):
        """Safely convert array to scalar, handling various array shapes"""
        if arr is None:
            return default
        if arr.size == 0:
            return default
        if arr.size == 1:
            return float(arr.flat[0])
        return arr
    
    def _extract_method_data(npz_path: str, method_name: str) -> Dict[str, Any]:
        """Extract data for a single method (GA or ES)"""
        data = {}
        
        if not os.path.exists(npz_path):
            print(f"[extract] {method_name}: NPZ file not found: {npz_path}")
            return data
            
        try:
            with np.load(npz_path, allow_pickle=True) as f:
                print(f"[extract] {method_name}: Available keys: {list(f.keys())}")
                
                # Extract budget information
                budget_key = f"{method_name.lower()}_budget"
                if budget_key in f:
                    data['budget'] = np.array(f[budget_key]).reshape(-1)
                    print(f"[extract] {method_name}: Budget loaded: {data['budget'].shape}, range: [{data['budget'].min()}, {data['budget'].max()}]")
                else:
                    print(f"[extract] {method_name}: Budget key '{budget_key}' not found")
                
                # Extract per-sample losses if available
                losses_key = f"{method_name.lower()}_losses_per_sample"
                if losses_key in f:
                    data['losses_per_sample'] = np.array(f[losses_key])  # (N, T)
                    print(f"[extract] {method_name}: Losses per sample: {data['losses_per_sample'].shape}")
                    
                    # Compute statistics
                    data['losses_mean'] = np.mean(data['losses_per_sample'], axis=0)
                    data['losses_std'] = np.std(data['losses_per_sample'], axis=0)
                    data['losses_se'] = data['losses_std'] / np.sqrt(data['losses_per_sample'].shape[0])
                    print(f"[extract] {method_name}: Loss statistics computed")
                else:
                    print(f"[extract] {method_name}: Losses per sample key '{losses_key}' not found")
                
                # Extract per-sample accuracies if available
                accuracy_keys = [
                    f"{method_name.lower()}_accuracy_per_sample_per_step",
                    f"{method_name.lower()}_accuracy_per_sample",
                    f"{method_name.lower()}_overall_accuracy",
                    "overall_accuracy"
                ]
                
                for acc_key in accuracy_keys:
                    if acc_key in f:
                        acc_data = np.array(f[acc_key])
                        if acc_data.ndim == 1 and acc_data.size == dataset_length:
                            # Per-sample final accuracy
                            data['accuracy_per_sample'] = acc_data.reshape(-1, 1)  # (N, 1)
                            print(f"[extract] {method_name}: Final accuracy per sample: {data['accuracy_per_sample'].shape}")
                        elif acc_data.ndim == 2 and acc_data.shape[0] == dataset_length:
                            # Per-sample per-step accuracy
                            data['accuracy_per_sample'] = acc_data  # (N, T)
                            print(f"[extract] {method_name}: Accuracy per sample per step: {data['accuracy_per_sample'].shape}")
                        else:
                            # Overall accuracy
                            overall_acc = safe_array_to_scalar(acc_data)
                            data['overall_accuracy'] = overall_acc
                            print(f"[extract] {method_name}: Overall accuracy: {overall_acc}")
                        break
                
                # Compute accuracy statistics if per-sample data available
                if 'accuracy_per_sample' in data:
                    data['accuracy_mean'] = np.mean(data['accuracy_per_sample'], axis=0)
                    data['accuracy_std'] = np.std(data['accuracy_per_sample'], axis=0)
                    data['accuracy_se'] = data['accuracy_std'] / np.sqrt(data['accuracy_per_sample'].shape[0])
                    print(f"[extract] {method_name}: Accuracy statistics computed")
                
                # Extract trajectory data for single-sample case
                if dataset_length == 1:
                    trajectory_keys = [
                        f"{method_name.lower()}_trajectory_losses",
                        f"{method_name.lower()}_losses",
                        "trajectory_losses",
                        "losses"
                    ]
                    
                    for traj_key in trajectory_keys:
                        if traj_key in f:
                            data['trajectory_losses'] = np.array(f[traj_key]).reshape(-1)
                            print(f"[extract] {method_name}: Trajectory losses: {data['trajectory_losses'].shape}")
                            break
                    
                    # Extract trajectory accuracies
                    traj_acc_keys = [
                        f"{method_name.lower()}_trajectory_accuracy",
                        f"{method_name.lower()}_accuracy",
                        "trajectory_accuracy",
                        "accuracy"
                    ]
                    
                    for traj_acc_key in traj_acc_keys:
                        if traj_acc_key in f:
                            data['trajectory_accuracy'] = np.array(f[traj_acc_key]).reshape(-1)
                            print(f"[extract] {method_name}: Trajectory accuracy: {data['trajectory_accuracy'].shape}")
                            break
                
        except Exception as e:
            print(f"[extract] {method_name}: Error loading NPZ: {e}")
            
        return data
    
    # Extract data for both methods
    ga_data = _extract_method_data(ga_npz_path, "GA")
    es_data = _extract_method_data(es_npz_path, "ES")
    
    return {
        'ga': ga_data,
        'es': es_data,
        'dataset_length': dataset_length
    }


def compute_statistical_analysis_per_budget(ga_npz_path: str, es_npz_path: str, dataset_length: int) -> Dict[str, Any]:
    """
    Compute statistical analysis comparing GA vs ES performance per budget point.
    Returns a dictionary with p-values, test statistics, and effect sizes for each budget.
    
    Args:
        ga_npz_path: Path to GA trajectory NPZ file
        es_npz_path: Path to ES trajectory NPZ file  
        dataset_length: Number of samples evaluated
        
    Returns:
        Dictionary with statistical analysis results per budget point
    """
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    
    if dataset_length <= 1:
        print(f"[stats] Skipping per-budget statistical analysis: dataset_length={dataset_length} (need > 1)")
        return {}
    
    print(f"[stats] Computing per-budget statistical analysis for {dataset_length} samples...")
    
    # Extract data using the new extraction function
    extracted_data = extract_losses_and_accuracies_per_budget(ga_npz_path, es_npz_path, dataset_length)
    ga_data = extracted_data['ga']
    es_data = extracted_data['es']
    
    # Load GA per-sample data
    ga_per_sample_data = {}
    ga_budget = None
    if os.path.exists(ga_npz_path):
        try:
            with np.load(ga_npz_path) as npz:
                # Load budget information
                if "ga_budget" in npz:
                    ga_budget = np.array(npz["ga_budget"]).reshape(-1)
                    print(f"[stats] GA budget loaded: {ga_budget.shape}, range: [{ga_budget.min()}, {ga_budget.max()}]")
                
                # Load per-sample loss trajectories (N, T) where N=samples, T=time steps
                if "ga_losses_per_sample" in npz:
                    ga_per_sample_data['losses'] = np.array(npz["ga_losses_per_sample"])  # (N, T)
                    print(f"[stats] GA losses per sample: {ga_per_sample_data['losses'].shape}")
                
                # Load per-sample accuracy trajectories if available
                if "ga_accuracy_per_sample_per_step" in npz:
                    ga_per_sample_data['accuracy'] = np.array(npz["ga_accuracy_per_sample_per_step"])  # (N, T)
                    print(f"[stats] GA accuracy per sample per step: {ga_per_sample_data['accuracy'].shape}")
                elif "ga_accuracy_per_sample" in npz:
                    # If only final accuracy, replicate across all steps
                    final_acc = np.array(npz["ga_accuracy_per_sample"]).reshape(-1, 1)  # (N, 1)
                    if ga_per_sample_data.get('losses') is not None:
                        T = ga_per_sample_data['losses'].shape[1]
                        ga_per_sample_data['accuracy'] = np.tile(final_acc, (1, T))  # (N, T)
                        print(f"[stats] GA accuracy replicated across {T} steps: {ga_per_sample_data['accuracy'].shape}")
                
                # Load other metrics similarly
                for metric in ['shape_correctness', 'pixel_correctness']:
                    key = f"ga_{metric}_per_sample_per_step"
                    if key in npz:
                        ga_per_sample_data[metric] = np.array(npz[key])
                        print(f"[stats] GA {metric} per sample per step: {ga_per_sample_data[metric].shape}")
                    elif f"ga_{metric}_per_sample" in npz:
                        final_metric = np.array(npz[f"ga_{metric}_per_sample"]).reshape(-1, 1)
                        if ga_per_sample_data.get('losses') is not None:
                            T = ga_per_sample_data['losses'].shape[1]
                            ga_per_sample_data[metric] = np.tile(final_metric, (1, T))
                            print(f"[stats] GA {metric} replicated across {T} steps: {ga_per_sample_data[metric].shape}")
        except Exception as e:
            print(f"[stats] Failed to load GA per-sample data: {e}")
    
    # Load ES per-sample data
    es_per_sample_data = {}
    es_budget = None
    if os.path.exists(es_npz_path):
        try:
            with np.load(es_npz_path) as npz:
                # Load budget information
                if "es_budget" in npz:
                    es_budget = np.array(npz["es_budget"]).reshape(-1)
                    print(f"[stats] ES budget loaded: {es_budget.shape}, range: [{es_budget.min()}, {es_budget.max()}]")
                
                # Load per-sample loss trajectories (N, G) where N=samples, G=generations
                if "es_generation_losses_per_sample" in npz:
                    es_per_sample_data['losses'] = np.array(npz["es_generation_losses_per_sample"])  # (N, G)
                    print(f"[stats] ES losses per sample: {es_per_sample_data['losses'].shape}")
                
                # Load per-sample accuracy trajectories if available
                if "per_sample_accuracy_per_generation" in npz:
                    es_per_sample_data['accuracy'] = np.array(npz["per_sample_accuracy_per_generation"])  # (N, G)
                    print(f"[stats] ES accuracy per sample per generation: {es_per_sample_data['accuracy'].shape}")
                elif "per_sample_accuracy" in npz:
                    # If only final accuracy, replicate across all generations
                    final_acc = np.array(npz["per_sample_accuracy"]).reshape(-1, 1)  # (N, 1)
                    if es_per_sample_data.get('losses') is not None:
                        G = es_per_sample_data['losses'].shape[1]
                        es_per_sample_data['accuracy'] = np.tile(final_acc, (1, G))  # (N, G)
                        print(f"[stats] ES accuracy replicated across {G} generations: {es_per_sample_data['accuracy'].shape}")
                
                # Load other metrics similarly
                for metric in ['shape_correctness', 'pixel_correctness']:
                    key = f"per_sample_{metric}_per_generation"
                    if key in npz:
                        es_per_sample_data[metric] = np.array(npz[key])
                        print(f"[stats] ES {metric} per sample per generation: {es_per_sample_data[metric].shape}")
                    elif f"per_sample_{metric}" in npz:
                        final_metric = np.array(npz[f"per_sample_{metric}"]).reshape(-1, 1)
                        if es_per_sample_data.get('losses') is not None:
                            G = es_per_sample_data['losses'].shape[1]
                            es_per_sample_data[metric] = np.tile(final_metric, (1, G))
                            print(f"[stats] ES {metric} replicated across {G} generations: {es_per_sample_data[metric].shape}")
        except Exception as e:
            print(f"[stats] Failed to load ES per-sample data: {e}")
    
    # Helper function to check if array is binary
    def _is_binary_array(arr: np.ndarray) -> bool:
        u = np.unique(arr)
        return set(u.tolist()).issubset({0, 1})
    
    # Perform statistical tests per budget point
    results = {}
    metrics_for_test = ['accuracy', 'shape_correctness', 'pixel_correctness', 'losses']
    
    for metric in metrics_for_test:
        ga_data = ga_per_sample_data.get(metric, None)
        es_data = es_per_sample_data.get(metric, None)
        
        if ga_data is None or es_data is None:
            print(f"[stats] Skipping {metric}: missing data (GA: {ga_data is not None}, ES: {es_data is not None})")
            continue
        
        # Ensure same number of samples
        n_samples = min(ga_data.shape[0], es_data.shape[0])
        if n_samples < 2:
            print(f"[stats] Skipping {metric}: insufficient samples ({n_samples})")
            continue
        
        ga_data = ga_data[:n_samples]
        es_data = es_data[:n_samples]
        
        # Get the minimum number of time points to compare
        n_time_points = min(ga_data.shape[1], es_data.shape[1])
        if n_time_points < 2:
            print(f"[stats] Skipping {metric}: insufficient time points ({n_time_points})")
            continue
        
        print(f"[stats] Computing per-budget statistics for {metric}: {n_samples} samples × {n_time_points} time points")
        
        # Get budget values for this metric
        if metric == 'losses':
            ga_budget_vals = ga_budget if ga_budget is not None else np.arange(n_time_points)
            es_budget_vals = es_budget if es_budget is not None else np.arange(n_time_points)
        else:
            # For other metrics, use the same budget as losses if available
            ga_budget_vals = ga_budget if ga_budget is not None else np.arange(n_time_points)
            es_budget_vals = es_budget if es_budget is not None else np.arange(n_time_points)
        
        # Ensure budget arrays match the data dimensions
        if len(ga_budget_vals) != n_time_points:
            ga_budget_vals = np.arange(n_time_points)
        if len(es_budget_vals) != n_time_points:
            es_budget_vals = np.arange(n_time_points)
        
        # Compute statistics for each time point
        for t in range(n_time_points):
            ga_t = ga_data[:, t]  # (N,)
            es_t = es_data[:, t]  # (N,)
            
            # Skip if all values are the same (no variance)
            if np.std(ga_t) == 0 and np.std(es_t) == 0:
                continue
            
            # Get the actual budget value for this time point
            budget_val = int((ga_budget_vals[t] + es_budget_vals[t]) / 2)  # Average if different
            
            diff = ga_t - es_t
            
            # Paired effect size: Cohen's dz = mean(diff)/std(diff)
            diff_mean = float(np.mean(diff))
            diff_std = float(np.std(diff, ddof=1)) if n_samples > 1 else np.nan
            dz = float(diff_mean / diff_std) if diff_std > 0 else np.nan
            
            # 95% CI for paired difference
            se = diff_std / np.sqrt(n_samples) if np.isfinite(diff_std) else np.nan
            ci_low = diff_mean - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = diff_mean + 1.96 * se if np.isfinite(se) else np.nan
            
            # Choose appropriate test
            if metric == 'accuracy' and _is_binary_array(ga_t) and _is_binary_array(es_t):
                # McNemar's test for paired binary outcomes
                try:
                    from statsmodels.stats.contingency_tables import mcnemar
                    b = int(np.sum((ga_t == 1) & (es_t == 0)))
                    c = int(np.sum((ga_t == 0) & (es_t == 1)))
                    table = np.array([[0, b], [c, 0]])
                    res = mcnemar(table, exact=False, correction=True)
                    results[f"{metric}_budget_{budget_val}_test"] = "mcnemar"
                    results[f"{metric}_budget_{budget_val}_statistic"] = float(res.statistic)
                    results[f"{metric}_budget_{budget_val}_pvalue"] = float(res.pvalue)
                    results[f"{metric}_budget_{budget_val}_mean_diff"] = diff_mean
                    results[f"{metric}_budget_{budget_val}_cohens_dz"] = dz
                    results[f"{metric}_budget_{budget_val}_ci_low"] = ci_low
                    results[f"{metric}_budget_{budget_val}_ci_high"] = ci_high
                    results[f"{metric}_budget_{budget_val}_discordant_ga1_es0"] = b
                    results[f"{metric}_budget_{budget_val}_discordant_ga0_es1"] = c
                except Exception as e:
                    print(f"[stats] McNemar failed for {metric} at budget {t}: {e}")
                    # Fallback to paired t-test
                    try:
                        stat, p_val = ttest_rel(ga_t, es_t)
                        results[f"{metric}_budget_{budget_val}_test"] = "ttest_rel_fallback"
                        results[f"{metric}_budget_{budget_val}_statistic"] = float(stat)
                        results[f"{metric}_budget_{budget_val}_pvalue"] = float(p_val)
                        results[f"{metric}_budget_{budget_val}_mean_diff"] = diff_mean
                        results[f"{metric}_budget_{budget_val}_cohens_dz"] = dz
                        results[f"{metric}_budget_{budget_val}_ci_low"] = ci_low
                        results[f"{metric}_budget_{budget_val}_ci_high"] = ci_high
                    except Exception as e:
                        print(f"[stats] Paired t-test fallback failed for {metric} at budget {t}: {e}")
            else:
                # Continuous/near-continuous: paired t-test, Wilcoxon fallback
                try:
                    stat, p_val = ttest_rel(ga_t, es_t)
                    results[f"{metric}_budget_{budget_val}_test"] = "ttest_rel"
                    results[f"{metric}_budget_{budget_val}_statistic"] = float(stat)
                    results[f"{metric}_budget_{budget_val}_pvalue"] = float(p_val)
                    results[f"{metric}_budget_{budget_val}_mean_diff"] = diff_mean
                    results[f"{metric}_budget_{budget_val}_cohens_dz"] = dz
                    results[f"{metric}_budget_{budget_val}_ci_low"] = ci_low
                    results[f"{metric}_budget_{budget_val}_ci_high"] = ci_high
                except Exception as e:
                    print(f"[stats] Paired t-test failed for {metric} at budget {t}: {e}")
                    try:
                        stat, p_val = wilcoxon(diff)
                        results[f"{metric}_budget_{budget_val}_test"] = "wilcoxon"
                        results[f"{metric}_budget_{budget_val}_statistic"] = float(stat)
                        results[f"{metric}_budget_{budget_val}_pvalue"] = float(p_val)
                        results[f"{metric}_budget_{budget_val}_mean_diff"] = diff_mean
                        results[f"{metric}_budget_{budget_val}_cohens_dz"] = dz
                        results[f"{metric}_budget_{budget_val}_ci_low"] = ci_low
                        results[f"{metric}_budget_{budget_val}_ci_high"] = ci_high
                    except Exception as e:
                        print(f"[stats] Wilcoxon failed for {metric} at budget {t}: {e}")
    
    print(f"[stats] Computed per-budget statistical analysis with {len(results)} metrics")
    return results

def compute_statistical_analysis(ga_npz_path: str, es_npz_path: str, dataset_length: int) -> Dict[str, Any]:
    """
    Compute statistical analysis comparing GA vs ES performance across multiple metrics.
    Returns a dictionary with p-values, test statistics, and effect sizes.
    
    Args:
        ga_npz_path: Path to GA trajectory NPZ file
        es_npz_path: Path to ES trajectory NPZ file  
        dataset_length: Number of samples evaluated
        
    Returns:
        Dictionary with statistical analysis results
    """
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    
    if dataset_length <= 1:
        print(f"[stats] Skipping statistical analysis: dataset_length={dataset_length} (need > 1)")
        return {}
    
    print(f"[stats] Computing statistical analysis for {dataset_length} samples...")
    
    # Extract per-sample metrics from both NPZ files
    ga_metrics = {}
    es_metrics = {}
    
    # Load GA metrics
    if os.path.exists(ga_npz_path):
        try:
            with np.load(ga_npz_path) as npz:
                # Extract per-sample accuracy
                if "ga_accuracy_per_sample" in npz:
                    ga_metrics['accuracy'] = np.array(npz["ga_accuracy_per_sample"])
                # Extract per-sample shape correctness
                if "ga_shape_correctness_per_sample" in npz:
                    ga_metrics['shape_correctness'] = np.array(npz["ga_shape_correctness_per_sample"])
                # Extract per-sample pixel correctness
                if "ga_pixel_correctness_per_sample" in npz:
                    ga_metrics['pixel_correctness'] = np.array(npz["ga_pixel_correctness_per_sample"])
                # Extract per-sample best loss
                if "ga_losses_per_sample" in npz:
                    ga_losses = np.array(npz["ga_losses_per_sample"])
                    if ga_losses.ndim >= 2:
                        ga_metrics['best_loss'] = np.min(ga_losses, axis=1)  # Best loss per sample
        except Exception as e:
            print(f"[stats] Failed to load GA metrics: {e}")
    
    # Load ES metrics
    if os.path.exists(es_npz_path):
        try:
            with np.load(es_npz_path) as npz:
                # Extract per-sample accuracy
                if "es_accuracy_per_sample" in npz:
                    es_metrics['accuracy'] = np.array(npz["es_accuracy_per_sample"])
                # Extract per-sample shape correctness
                if "es_shape_correctness_per_sample" in npz:
                    es_metrics['shape_correctness'] = np.array(npz["es_shape_correctness_per_sample"])
                # Extract per-sample pixel correctness
                if "es_pixel_correctness_per_sample" in npz:
                    es_metrics['pixel_correctness'] = np.array(npz["es_pixel_correctness_per_sample"])
                # Extract per-sample best loss
                if "es_generation_losses_per_sample" in npz:
                    es_losses = np.array(npz["es_generation_losses_per_sample"])
                    if es_losses.ndim >= 2:
                        es_metrics['best_loss'] = np.min(es_losses, axis=1)  # Best loss per sample
        except Exception as e:
            print(f"[stats] Failed to load ES metrics: {e}")
    
    # Helper function to check if array is binary
    def _is_binary_array(arr: np.ndarray) -> bool:
        u = np.unique(arr)
        return set(u.tolist()).issubset({0, 1})
    
    # Perform statistical tests
    results = {}
    metrics_for_test = ['accuracy', 'shape_correctness', 'pixel_correctness', 'best_loss']
    
    for metric in metrics_for_test:
        ga_data = ga_metrics.get(metric, None)
        es_data = es_metrics.get(metric, None)
        if ga_data is None or es_data is None:
            continue
            
        # Ensure same length and not empty
        n = min(len(ga_data), len(es_data))
        if n < 2:
            continue
            
        ga_arr = np.asarray(ga_data[:n], dtype=float)
        es_arr = np.asarray(es_data[:n], dtype=float)
        diff = ga_arr - es_arr
        
        # Paired effect size: Cohen's dz = mean(diff)/std(diff)
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1)) if n > 1 else np.nan
        dz = float(diff_mean / diff_std) if diff_std > 0 else np.nan
        
        # 95% CI for paired difference (normal approx)
        se = diff_std / np.sqrt(n) if np.isfinite(diff_std) else np.nan
        ci_low = diff_mean - 1.96 * se if np.isfinite(se) else np.nan
        ci_high = diff_mean + 1.96 * se if np.isfinite(se) else np.nan
        
        # Choose appropriate test
        if metric == 'accuracy' and _is_binary_array(ga_arr) and _is_binary_array(es_arr):
            # McNemar's test for paired binary outcomes
            try:
                from statsmodels.stats.contingency_tables import mcnemar
                # Build 2x2: b = GA=1,ES=0; c = GA=0,ES=1
                b = int(np.sum((ga_arr == 1) & (es_arr == 0)))
                c = int(np.sum((ga_arr == 0) & (es_arr == 1)))
                table = np.array([[0, b], [c, 0]])
                res = mcnemar(table, exact=False, correction=True)
                results[f"{metric}_test"] = "mcnemar"
                results[f"{metric}_statistic"] = float(res.statistic)
                results[f"{metric}_pvalue"] = float(res.pvalue)
                results[f"{metric}_mean_diff"] = diff_mean
                results[f"{metric}_cohens_dz"] = dz
                results[f"{metric}_ci_low"] = ci_low
                results[f"{metric}_ci_high"] = ci_high
                results[f"{metric}_discordant_ga1_es0"] = b
                results[f"{metric}_discordant_ga0_es1"] = c
            except Exception as e:
                print(f"[stats] McNemar failed for {metric}: {e}")
                # Fallback to paired t-test
                try:
                    stat, p_val = ttest_rel(ga_arr, es_arr)
                    results[f"{metric}_test"] = "ttest_rel_fallback"
                    results[f"{metric}_statistic"] = float(stat)
                    results[f"{metric}_pvalue"] = float(p_val)
                    results[f"{metric}_mean_diff"] = diff_mean
                    results[f"{metric}_cohens_dz"] = dz
                    results[f"{metric}_ci_low"] = ci_low
                    results[f"{metric}_ci_high"] = ci_high
                except Exception as e:
                    print(f"[stats] Paired t-test fallback failed for {metric}: {e}")
        else:
            # Continuous/near-continuous: paired t-test, Wilcoxon fallback
            try:
                stat, p_val = ttest_rel(ga_arr, es_arr)
                results[f"{metric}_test"] = "ttest_rel"
                results[f"{metric}_statistic"] = float(stat)
                results[f"{metric}_pvalue"] = float(p_val)
                results[f"{metric}_mean_diff"] = diff_mean
                results[f"{metric}_cohens_dz"] = dz
                results[f"{metric}_ci_low"] = ci_low
                results[f"{metric}_ci_high"] = ci_high
            except Exception as e:
                print(f"[stats] Paired t-test failed for {metric}: {e}")
                try:
                    stat, p_val = wilcoxon(diff)
                    results[f"{metric}_test"] = "wilcoxon"
                    results[f"{metric}_statistic"] = float(stat)
                    results[f"{metric}_pvalue"] = float(p_val)
                    results[f"{metric}_mean_diff"] = diff_mean
                    results[f"{metric}_cohens_dz"] = dz
                    results[f"{metric}_ci_low"] = ci_low
                    results[f"{metric}_ci_high"] = ci_high
                except Exception as e:
                    print(f"[stats] Wilcoxon failed for {metric}: {e}")
    
    print(f"[stats] Computed statistical analysis with {len(results)} metrics")
    return results

def generate_budget_based_plots(ga_npz_path: str, es_npz_path: str, out_dir: str, 
                               dataset_length: int, checkpoint_name: str, checkpoint_step: int) -> Dict[str, str]:
    """
    Generate budget-based plots similar to store_latent_search.py.
    Creates loss vs budget and accuracy vs budget plots.
    
    Args:
        ga_npz_path: Path to GA trajectory NPZ file
        es_npz_path: Path to ES trajectory NPZ file
        out_dir: Output directory for plots
        dataset_length: Number of samples evaluated
        checkpoint_name: Name of the checkpoint
        checkpoint_step: Step number of the checkpoint
        
    Returns:
        Dictionary with paths to generated plot files
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"[plots] Generating budget-based plots for {dataset_length} samples...")
    
    # Extract data using the extraction function
    extracted_data = extract_losses_and_accuracies_per_budget(ga_npz_path, es_npz_path, dataset_length)
    ga_data = extracted_data['ga']
    es_data = extracted_data['es']
    
    plot_paths = {}
    
    # Create loss vs budget plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        title = f"Loss vs Budget: {checkpoint_name} (Step {checkpoint_step})"
        ax.set_title(title)
        ax.set_xlabel("Budget (evaluations)")
        ax.set_ylabel("Loss (lower is better)")
        ax.grid(True, alpha=0.3)
        
        # Plot GA data
        if 'losses_mean' in ga_data and 'budget' in ga_data:
            ga_budget = ga_data['budget']
            ga_mean = ga_data['losses_mean']
            ga_se = ga_data.get('losses_se', np.zeros_like(ga_mean))
            
            ax.fill_between(ga_budget, ga_mean - ga_se, ga_mean + ga_se, 
                           color="#FBB998", alpha=0.25, label="GA standard error")
            ax.plot(ga_budget, ga_mean, color="#FBB998", linewidth=3.0, 
                   marker='o', markersize=4, label="GA mean")
            print(f"[plots] GA: {len(ga_budget)} budget points, loss range: [{ga_mean.min():.4f}, {ga_mean.max():.4f}]")
        
        # Plot ES data
        if 'losses_mean' in es_data and 'budget' in es_data:
            es_budget = es_data['budget']
            es_mean = es_data['losses_mean']
            es_se = es_data.get('losses_se', np.zeros_like(es_mean))
            
            ax.fill_between(es_budget, es_mean - es_se, es_mean + es_se, 
                           color="#5361E5", alpha=0.25, label="ES standard error")
            ax.plot(es_budget, es_mean, color="#5361E5", linewidth=3.0, 
                   marker='s', markersize=4, label="ES mean")
            print(f"[plots] ES: {len(es_budget)} budget points, loss range: [{es_mean.min():.4f}, {es_mean.max():.4f}]")
        
        ax.legend()
        ax.set_xlim(left=0)
        
        # Save plot
        loss_plot_path = os.path.join(out_dir, f"loss_vs_budget_{checkpoint_name}_step{checkpoint_step}.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['loss_vs_budget'] = loss_plot_path
        print(f"[plots] Loss vs budget plot saved: {loss_plot_path}")
        
    except Exception as e:
        print(f"[plots] Error creating loss vs budget plot: {e}")
    
    # Create accuracy vs budget plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        title = f"Accuracy vs Budget: {checkpoint_name} (Step {checkpoint_step})"
        ax.set_title(title)
        ax.set_xlabel("Budget (evaluations)")
        ax.set_ylabel("Accuracy (higher is better)")
        ax.grid(True, alpha=0.3)
        
        # Plot GA accuracy data
        if 'accuracy_mean' in ga_data and 'budget' in ga_data:
            ga_budget = ga_data['budget']
            ga_acc_mean = ga_data['accuracy_mean']
            ga_acc_se = ga_data.get('accuracy_se', np.zeros_like(ga_acc_mean))
            
            ax.fill_between(ga_budget, ga_acc_mean - ga_acc_se, ga_acc_mean + ga_acc_se, 
                           color="#FBB998", alpha=0.25, label="GA accuracy standard error")
            ax.plot(ga_budget, ga_acc_mean, color="#FBB998", linewidth=3.0, 
                   marker='o', markersize=4, label="GA accuracy mean")
            print(f"[plots] GA accuracy: {len(ga_budget)} budget points, range: [{ga_acc_mean.min():.4f}, {ga_acc_mean.max():.4f}]")
        
        # Plot ES accuracy data
        if 'accuracy_mean' in es_data and 'budget' in es_data:
            es_budget = es_data['budget']
            es_acc_mean = es_data['accuracy_mean']
            es_acc_se = es_data.get('accuracy_se', np.zeros_like(es_acc_mean))
            
            ax.fill_between(es_budget, es_acc_mean - es_acc_se, es_acc_mean + es_acc_se, 
                           color="#5361E5", alpha=0.25, label="ES accuracy standard error")
            ax.plot(es_budget, es_acc_mean, color="#5361E5", linewidth=3.0, 
                   marker='s', markersize=4, label="ES accuracy mean")
            print(f"[plots] ES accuracy: {len(es_budget)} budget points, range: [{es_acc_mean.min():.4f}, {es_acc_mean.max():.4f}]")
        
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)  # Accuracy is typically 0-1
        
        # Save plot
        acc_plot_path = os.path.join(out_dir, f"accuracy_vs_budget_{checkpoint_name}_step{checkpoint_step}.png")
        plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['accuracy_vs_budget'] = acc_plot_path
        print(f"[plots] Accuracy vs budget plot saved: {acc_plot_path}")
        
    except Exception as e:
        print(f"[plots] Error creating accuracy vs budget plot: {e}")
    
    return plot_paths


def generate_loss_vs_budget_plot(method_arrays: Dict[str, np.ndarray], 
                                budgets: List[int], 
                                method_names: List[str],
                                checkpoint_name: str, 
                                checkpoint_step: int) -> str:
    """Generate a plot showing Loss vs Budget for both methods."""
    try:
        # SAFETY CHECK: Ensure reasonable data dimensions
        if len(budgets) > 1000:
            print(f"⚠️  WARNING: Too many budgets ({len(budgets)}), limiting to 1000 for plotting")
            budget_indices = np.linspace(0, len(budgets)-1, 1000, dtype=int)
            budgets = [budgets[i] for i in budget_indices]
            # Also limit method arrays
            for method in method_names:
                if method in method_arrays:
                    method_arrays[method] = method_arrays[method][budget_indices, :]
        
        # ADDITIONAL SAFETY CHECK: Ensure budget values are reasonable to prevent extremely large images
        if max(budgets) > 10000:
            print(f"⚠️  WARNING: Budget values too large for loss vs budget plot!")
            print(f"   Max budget: {max(budgets)}")
            print(f"   This would create an extremely tall image. Skipping plot generation.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use custom color palette
        colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
        for i, method in enumerate(method_names):
            # Get the loss data for this method (averaged across checkpoints)
            method_data = method_arrays[method]
            # Average across checkpoints (axis=1) to get one value per budget
            avg_losses = np.nanmean(method_data, axis=1)
            
            # Filter out NaN values
            valid_indices = ~np.isnan(avg_losses)
            if np.any(valid_indices):
                valid_budgets = [budgets[j] for j in range(len(budgets)) if valid_indices[j]]
                valid_losses = [avg_losses[j] for j in range(len(avg_losses)) if valid_indices[j]]
                
                ax.plot(valid_budgets, valid_losses, marker='o', linewidth=2, markersize=8,
                       color=colors[i % len(colors)], label=method.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel("Budget", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title(f"Loss vs Budget Comparison\n"
                    f"Checkpoint: {checkpoint_name} (Step: {checkpoint_step})", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_ylim(bottom=0)  # Loss is typically non-negative
        
        # Save figure
        out_dir = Path("results")
        fig_path = out_dir / f"loss_vs_budget_{checkpoint_step}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return str(fig_path)
        
    except Exception as e:
        print(f"⚠️  Failed to generate loss vs budget plot: {e}")
        return None

def generate_loss_vs_training_plot(method_arrays: Dict[str, np.ndarray], 
                                  steps: List[int], 
                                  method_names: List[str],
                                  checkpoint_name: str, 
                                  checkpoint_step: int,
                                  total_checkpoints: int) -> str:
    """Generate a plot showing Loss vs Training Progress for both methods."""
    try:
        # SAFETY CHECK: Ensure reasonable data dimensions
        if len(steps) > 1000:
            print(f"⚠️  WARNING: Too many steps ({len(steps)}), limiting to 1000 for plotting")
            step_indices = np.linspace(0, len(steps)-1, 1000, dtype=int)
            steps = [steps[i] for i in step_indices]
            # Also limit method arrays
            for method in method_names:
                if method in method_arrays:
                    method_arrays[method] = method_arrays[method][:, step_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use custom color palette
        colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
        for i, method in enumerate(method_names):
            # Get the loss data for this method (averaged across budgets)
            method_data = method_arrays[method]
            # Average across budgets (axis=0) to get one value per checkpoint
            avg_losses = np.nanmean(method_data, axis=0)
            
            # Calculate training progress percentage
            training_progress = [(step / max(total_checkpoints - 1, 1)) * 100 for step in steps]
            
            # Filter out NaN values
            valid_indices = ~np.isnan(avg_losses)
            if np.any(valid_indices):
                valid_progress = [training_progress[j] for j in range(len(training_progress)) if valid_indices[j]]
                valid_losses = [avg_losses[j] for j in range(len(avg_losses)) if valid_indices[j]]
                
                ax.plot(valid_progress, valid_losses, marker='o', linewidth=2, markersize=8,
                       color=colors[i % len(colors)], label=method.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel("Training Progress (%)", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title(f"Loss vs Training Progress Comparison\n"
                    f"Checkpoint: {checkpoint_name} (Step: {checkpoint_step})", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_ylim(bottom=0)  # Loss is typically non-negative
        ax.set_xlim(0, 100)  # Training progress is 0-100%
        
        # Save figure
        out_dir = Path("results")
        fig_path = out_dir / f"loss_vs_training_{checkpoint_step}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return str(fig_path)
        
    except Exception as e:
        print(f"⚠️  Failed to generate loss vs training plot: {e}")
        return None

# Dataset functionality imports
import jax
from jax.tree_util import tree_map
from data_utils import make_leave_one_out
from train import load_datasets

def extract_loss_from_trajectory_file(npz_path: str, method: str) -> Optional[float]:
    """
    Extract final loss from saved trajectory NPZ file using store_latent_search functions.
    
    Args:
        npz_path: Path to the saved trajectory file
        method: Method name ('gradient_ascent', 'evolutionary_search', 'random_search')
    
    Returns:
        Final loss value or None if not available
    """
    try:
        import numpy as np
        
        if not os.path.exists(npz_path):
            return None
            
        with np.load(npz_path) as npz:
            if method == "gradient_ascent":
                # Try to extract GA losses using store_latent_search functions
                if "ga_losses_per_sample" in npz:
                    # Get final step losses for all samples
                    ga_losses = np.array(npz["ga_losses_per_sample"])  # (N, steps)
                    if ga_losses.ndim >= 2:
                        final_losses = ga_losses[:, -1]  # Last step for each sample
                        return float(np.mean(final_losses))
                elif "ga_losses" in npz:
                    # Get final loss from trajectory
                    ga_losses = np.array(npz["ga_losses"])
                    if ga_losses.ndim >= 1:
                        return float(ga_losses[-1])  # Last step
                        
            elif method == "evolutionary_search":
                # Try to extract ES losses using store_latent_search functions
                if "es_generation_losses_per_sample" in npz:
                    # Get final generation losses for all samples
                    es_losses = np.array(npz["es_generation_losses_per_sample"])  # (N, generations)
                    if es_losses.ndim >= 2:
                        final_losses = es_losses[:, -1]  # Last generation for each sample
                        return float(np.mean(final_losses))
                elif "es_generation_losses" in npz:
                    # Get final loss from trajectory
                    es_losses = np.array(npz["es_generation_losses"])
                    if es_losses.ndim >= 1:
                        return float(es_losses[-1])  # Last generation
                elif "es_final_best_loss" in npz:
                    return float(np.array(npz["es_final_best_loss"]))
                    
            elif method == "random_search":
                # For random search, we might not have trajectory data
                # But we can try to extract from any available data
                if "per_sample_accuracy" in npz:
                    accuracies = np.array(npz["per_sample_accuracy"])
                    if accuracies.ndim >= 1:
                        # Convert accuracy to loss (1 - accuracy)
                        final_losses = 1.0 - accuracies
                        return float(np.mean(final_losses))
                        
    except Exception as e:
        print(f"⚠️  Failed to extract loss from trajectory file {npz_path} for {method}: {e}")
    
    return None


def extract_full_trajectory_data(npz_path: str, method: str, dataset_length: int) -> Dict[str, Any]:
    """
    Extract full trajectory data from NPZ file for high-granularity heatmap generation.
    Similar to store_latent_search.py but focused on data extraction for heatmaps.
    
    Args:
        npz_path: Path to the saved trajectory file
        method: Method name ('gradient_ascent', 'evolutionary_search', 'random_search')
        dataset_length: Number of samples evaluated
        
    Returns:
        Dictionary with full trajectory data for heatmap generation
    """
    import numpy as np
    
    if not os.path.exists(npz_path):
        print(f"[extract_full] NPZ file not found: {npz_path}")
        return {}
    
    print(f"[extract_full] Extracting full trajectory data from {npz_path} for {method}...")
    
    try:
        with np.load(npz_path, allow_pickle=True) as npz:
            print(f"[extract_full] Available keys: {list(npz.keys())}")
            
            trajectory_data = {}
            
            if method == "gradient_ascent":
                # Extract GA trajectory data
                if "ga_losses_per_sample" in npz:
                    losses_per_sample = np.array(npz["ga_losses_per_sample"])  # (N, T)
                    trajectory_data['losses_per_sample'] = losses_per_sample
                    trajectory_data['losses_mean'] = np.mean(losses_per_sample, axis=0)  # (T,)
                    trajectory_data['losses_std'] = np.std(losses_per_sample, axis=0)  # (T,)
                    print(f"[extract_full] GA losses: {losses_per_sample.shape}, mean range: [{trajectory_data['losses_mean'].min():.4f}, {trajectory_data['losses_mean'].max():.4f}]")
                
                # Extract GA accuracy data if available
                if "ga_accuracy_per_sample_per_step" in npz:
                    acc_per_sample = np.array(npz["ga_accuracy_per_sample_per_step"])  # (N, T)
                    trajectory_data['accuracy_per_sample'] = acc_per_sample
                    trajectory_data['accuracy_mean'] = np.mean(acc_per_sample, axis=0)  # (T,)
                    trajectory_data['accuracy_std'] = np.std(acc_per_sample, axis=0)  # (T,)
                    print(f"[extract_full] GA accuracy: {acc_per_sample.shape}, mean range: [{trajectory_data['accuracy_mean'].min():.4f}, {trajectory_data['accuracy_mean'].max():.4f}]")
                
                # Extract budget information (steps)
                if "ga_budget" in npz:
                    trajectory_data['budget'] = np.array(npz["ga_budget"]).reshape(-1)
                else:
                    # Create budget from number of steps
                    if 'losses_per_sample' in trajectory_data:
                        T = trajectory_data['losses_per_sample'].shape[1]
                        trajectory_data['budget'] = np.arange(1, T + 1)  # 1, 2, 3, ..., T
                
                # Extract other metrics
                for metric in ['shape_correctness', 'pixel_correctness']:
                    key = f"ga_{metric}_per_sample_per_step"
                    if key in npz:
                        metric_data = np.array(npz[key])  # (N, T)
                        trajectory_data[f'{metric}_per_sample'] = metric_data
                        trajectory_data[f'{metric}_mean'] = np.mean(metric_data, axis=0)  # (T,)
                        trajectory_data[f'{metric}_std'] = np.std(metric_data, axis=0)  # (T,)
                        print(f"[extract_full] GA {metric}: {metric_data.shape}")
                
            elif method == "evolutionary_search":
                # Extract ES trajectory data
                if "es_generation_losses_per_sample" in npz:
                    losses_per_sample = np.array(npz["es_generation_losses_per_sample"])  # (N, G)
                    trajectory_data['losses_per_sample'] = losses_per_sample
                    trajectory_data['losses_mean'] = np.mean(losses_per_sample, axis=0)  # (G,)
                    trajectory_data['losses_std'] = np.std(losses_per_sample, axis=0)  # (G,)
                    print(f"[extract_full] ES losses: {losses_per_sample.shape}, mean range: [{trajectory_data['losses_mean'].min():.4f}, {trajectory_data['losses_mean'].max():.4f}]")
                
                # Extract ES accuracy data if available
                if "per_sample_accuracy_per_generation" in npz:
                    acc_per_sample = np.array(npz["per_sample_accuracy_per_generation"])  # (N, G)
                    trajectory_data['accuracy_per_sample'] = acc_per_sample
                    trajectory_data['accuracy_mean'] = np.mean(acc_per_sample, axis=0)  # (G,)
                    trajectory_data['accuracy_std'] = np.std(acc_per_sample, axis=0)  # (G,)
                    print(f"[extract_full] ES accuracy: {acc_per_sample.shape}, mean range: [{trajectory_data['accuracy_mean'].min():.4f}, {trajectory_data['accuracy_mean'].max():.4f}]")
                
                # Extract budget information (generations)
                if "es_budget" in npz:
                    trajectory_data['budget'] = np.array(npz["es_budget"]).reshape(-1)
                else:
                    # Create budget from number of generations
                    if 'losses_per_sample' in trajectory_data:
                        G = trajectory_data['losses_per_sample'].shape[1]
                        trajectory_data['budget'] = np.arange(1, G + 1)  # 1, 2, 3, ..., G
                
                # Extract other metrics
                for metric in ['shape_correctness', 'pixel_correctness']:
                    key = f"per_sample_{metric}_per_generation"
                    if key in npz:
                        metric_data = np.array(npz[key])  # (N, G)
                        trajectory_data[f'{metric}_per_sample'] = metric_data
                        trajectory_data[f'{metric}_mean'] = np.mean(metric_data, axis=0)  # (G,)
                        trajectory_data[f'{metric}_std'] = np.std(metric_data, axis=0)  # (G,)
                        print(f"[extract_full] ES {metric}: {metric_data.shape}")
                
            elif method == "random_search":
                # For random search, extract what's available
                if "per_sample_accuracy" in npz:
                    acc_data = np.array(npz["per_sample_accuracy"])  # (N,)
                    trajectory_data['accuracy_per_sample'] = acc_data.reshape(-1, 1)  # (N, 1)
                    trajectory_data['accuracy_mean'] = np.array([np.mean(acc_data)])  # (1,)
                    trajectory_data['accuracy_std'] = np.array([np.std(acc_data)])  # (1,)
                    print(f"[extract_full] RS accuracy: {acc_data.shape}")
                
                # Create single budget point
                trajectory_data['budget'] = np.array([1])
            
            print(f"[extract_full] Extracted trajectory data with keys: {list(trajectory_data.keys())}")
            return trajectory_data
            
    except Exception as e:
        print(f"[extract_full] Error extracting trajectory data from {npz_path}: {e}")
        return {}


def extract_loss_from_trajectory(info: dict, method: str) -> Optional[float]:
    """
    Extract final loss from optimization trajectory using store_latent_search functions.
    
    Args:
        info: Info dictionary from model evaluation
        method: Method name ('gradient_ascent', 'evolutionary_search', 'random_search')
    
    Returns:
        Final loss value or None if not available
    """
    try:
        if method == "gradient_ascent" and "optimization_trajectory" in info:
            trajectory = info["optimization_trajectory"]
            if isinstance(trajectory, dict):
                # Extract final loss from GA trajectory
                if "log_probs" in trajectory:
                    log_probs = np.array(trajectory["log_probs"])
                    if log_probs.ndim >= 2:
                        # Get final step and best candidate
                        final_step_log_probs = log_probs[..., -1, :]  # Last step
                        best_final_log_probs = np.max(final_step_log_probs, axis=-1)  # Best candidate
                        final_losses = -best_final_log_probs  # Convert to positive loss
                        return float(np.mean(final_losses))
                elif "losses" in trajectory:
                    losses = np.array(trajectory["losses"])
                    if losses.ndim >= 1:
                        final_losses = losses[..., -1]  # Last step
                        return float(np.mean(final_losses))
                        
        elif method == "evolutionary_search" and "evolutionary_trajectory" in info:
            trajectory = info["evolutionary_trajectory"]
            if isinstance(trajectory, dict):
                # Extract final loss from ES trajectory
                if "losses_per_generation" in trajectory:
                    losses = np.array(trajectory["losses_per_generation"])
                    if losses.ndim >= 1:
                        final_losses = losses[..., -1]  # Last generation
                        return float(np.mean(final_losses))
                elif "final_best_loss" in trajectory:
                    return float(np.array(trajectory["final_best_loss"]))
                    
        elif method == "random_search" and "search_trajectory" in info:
            trajectory = info["search_trajectory"]
            if isinstance(trajectory, dict):
                # Extract final accuracy from RS trajectory
                if "best_accuracy_progression" in trajectory:
                    accuracies = np.array(trajectory["best_accuracy_progression"])
                    if accuracies.ndim >= 1:
                        final_acc = accuracies[..., -1]  # Last sample
                        # Convert accuracy to loss (1 - accuracy)
                        final_losses = 1.0 - final_acc
                        return float(np.mean(final_losses))
                        
    except Exception as e:
        print(f"⚠️  Failed to extract loss from trajectory for {method}: {e}")
    
    return None


def log_evaluation_start(method: str, budget_info: Dict[str, Any], method_kwargs: Dict[str, Any], 
                        checkpoint_name: str, checkpoint_step: int, args: Optional[Any] = None) -> None:
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
        # Add subspace parameters if available
        if args is not None and hasattr(args, 'es_use_subspace_mutation') and args.es_use_subspace_mutation:
            print(f"   • Subspace Mutation: Enabled (dim={args.es_subspace_dim}, ga_step={args.es_ga_step_length})")
            if args.es_trust_region_radius is not None:
                print(f"   • Trust Region Radius: {args.es_trust_region_radius}")
        else:
            print(f"   • Subspace Mutation: Disabled (standard isotropic mutation)")
    
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
        # Check if we have enough data to create a meaningful plot
        if not results_data or len(results_data) == 0:
            print("⚠️  No results data available for checkpoint figure generation")
            return None
            
        # SAFETY CHECK: Prevent issues with extremely large step values
        if checkpoint_step > 1000000:  # 1 million steps
            print(f"⚠️  WARNING: Extremely large step value detected for checkpoint figure!")
            print(f"   Step: {checkpoint_step}")
            print(f"   This could cause plotting issues. Skipping checkpoint figure generation.")
            return None
            
        # ADDITIONAL SAFETY CHECK: Ensure step value is reasonable for filename
        if checkpoint_step > 999999:  # Limit to 6 digits for filename safety
            print(f"⚠️  Step value {checkpoint_step} is too large for safe filename generation")
            print(f"   Using truncated value for filename")
            safe_step = checkpoint_step % 1000000  # Use modulo to get reasonable value
        else:
            safe_step = checkpoint_step
            
        # Limit the number of methods to prevent extremely wide images
        max_methods = min(4, len(plot_methods)) if plot_methods else 2
        
        # Calculate appropriate figure size based on data
        fig_width = min(20, max(8, max_methods * 3))  # Cap width at 20 inches
        fig_height = 6
        
        # Create a simple progress visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        
        # Plot 1: Training Progress
        progress_pct = (training_progress / max(total_checkpoints - 1, 1)) * 100
        ax1.bar(['Training Progress'], [progress_pct], color='skyblue', alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Progress (%)')
        ax1.set_title(f'Checkpoint Progress: {training_progress}/{total_checkpoints-1}')
        ax1.text(0, progress_pct + 2, f'{progress_pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Method Performance (if we have data)
        if results_data:
            methods = list(set([r['method'] for r in results_data]))[:max_methods]  # Limit methods
            # Use custom color palette
            method_colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
            
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
        
        # Rotate x-axis labels if there are many methods
        if len(methods) > 3:
            ax2.tick_params(axis='x', labelrotation=45, ha='right')
        
        # Overall title - use safe step value for display
        fig.suptitle(f'Checkpoint {checkpoint_name} - Step {safe_step}', fontsize=16, y=0.95)
        
        # Save figure - use safe step value for filename
        out_dir = Path("results")
        fig_path = out_dir / f"checkpoint_{safe_step}_progress_{training_progress}.png"
        
        # ADDITIONAL SAFETY CHECK: Verify figure dimensions before saving
        fig_width_px = int(fig_width * 200)  # 200 DPI
        fig_height_px = int(fig_height * 200)  # 200 DPI
        
        if fig_width_px > 65000 or fig_height_px > 65000:
            print(f"⚠️  Figure dimensions too large: {fig_width_px}x{fig_height_px} pixels")
            print(f"   Reducing figure size to prevent issues")
            # Reduce figure size if too large
            fig.set_size_inches(min(16, fig_width), min(8, fig_height))
        
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
    

def run_evaluation_inprocess(
    train_state,
    evaluator,
    method: str,
    method_kwargs: Dict[str, Any],
    dataset_folder: str,
    dataset_length: Optional[int],
    dataset_batch_size: int,
    dataset_use_hf: bool,
    dataset_seed: int,
    preloaded_data: Optional[Dict] = None,
) -> Tuple[bool, Optional[float], Dict[str, Optional[float]], float]:
    """
    Run evaluation in-process to access trajectory data directly.
    This is much more efficient than subprocess calls and gives us access to loss data.
    """
    try:
        from evaluate_checkpoint import evaluate_custom_dataset
        
        # Use preloaded data if available
        if preloaded_data is not None:
            # We need to modify the evaluator to use our preloaded data
            # This is a bit complex, so for now we'll fall back to subprocess
            print("⚠️  In-process evaluation with preloaded data not yet implemented, falling back to subprocess")
            return False, None, {}, 0.0
        
        # For now, we'll use the existing evaluate_custom_dataset function
        # but we need to ensure it returns the info we need
        print("⚠️  In-process evaluation not yet fully implemented, falling back to subprocess")
        return False, None, {}, 0.0
        
    except Exception as e:
        print(f"⚠️  In-process evaluation failed: {e}")
        return False, None, {}, 0.0


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
    args: Optional[Any] = None,
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
        
        # Add advanced GA parameters
        if method_kwargs.get("accumulate_gradients_decoder_pairs"):
            cmd.extend(["--accumulate-gradients-decoder-pairs", "true"])
        if method_kwargs.get("scan_gradients_latents"):
            cmd.extend(["--scan-gradients-latents", "true"])
        if method_kwargs.get("include_mean_latent"):
            cmd.extend(["--include-mean-latent", "true"])
        if method_kwargs.get("include_all_latents"):
            cmd.extend(["--include-all-latents", "true"])
        if method_kwargs.get("random_perturbation"):
            cmd.extend(["--random-perturbation", method_kwargs["random_perturbation"]])
        if method_kwargs.get("track_progress"):
            cmd.extend(["--track-progress"])
            
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
        
        # Add advanced RS parameters
        if method_kwargs.get("include_mean_latent"):
            cmd.extend(["--include-mean-latent", "true"])
        if method_kwargs.get("include_all_latents"):
            cmd.extend(["--include-all-latents", "true"])
        if method_kwargs.get("random_perturbation"):
            cmd.extend(["--random-perturbation", method_kwargs["random_perturbation"]])
        if method_kwargs.get("track_progress"):
            cmd.extend(["--track-progress"])
            
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
        
        # Add advanced ES parameters
        if method_kwargs.get("mutation_decay") is not None:
            cmd.extend(["--mutation-decay", str(method_kwargs["mutation_decay"])])
        if method_kwargs.get("elite_size") is not None:
            cmd.extend(["--elite-size", str(method_kwargs["elite_size"])])
        if method_kwargs.get("track_progress"):
            cmd.extend(["--track-progress"])
        
        # Add subspace parameters if enabled
        if args is not None and args.es_use_subspace_mutation:
            cmd.extend(["--use-subspace-mutation"])
            cmd.extend(["--subspace-dim", str(args.es_subspace_dim)])
            cmd.extend(["--ga-step-length", str(args.es_ga_step_length)])
            if args.es_trust_region_radius is not None:
                cmd.extend(["--trust-region-radius", str(args.es_trust_region_radius)])
    else:
        print(f"❌ Unknown method: {method}")
        return False, None, {}, ""

    # Enable trajectory storage to get loss data
    cmd.extend(["--store-latents", f"temp_trajectories/{method}_{artifact_path.split('/')[-1]}.npz"])
    
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
            # Loss metrics
            "total_final_loss": r"total_final_loss:\s*([0-9]*\.?[0-9]+)",
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
            # Try to extract loss data from the saved trajectory file
            trajectory_path = f"temp_trajectories/{method}_{artifact_path.split('/')[-1]}.npz"
            if os.path.exists(trajectory_path):
                extracted_loss = extract_loss_from_trajectory_file(trajectory_path, method)
                if extracted_loss is not None:
                    metrics["total_final_loss"] = extracted_loss
                    print(f"📊 Extracted loss from trajectory: {extracted_loss:.6f}")
                else:
                    print(f"⚠️  Could not extract loss from trajectory file")
            else:
                print(f"⚠️  Trajectory file not found: {trajectory_path}")
            
            print(
                f"✅ {method} evaluation completed successfully"
                + (f" | accuracy={acc}" if acc is not None else "")
                + (f" | shape_acc={metrics.get('top_1_shape_accuracy', 'N/A')}" if metrics.get('top_1_shape_accuracy') is not None else "")
                + (f" | pixel_acc={metrics.get('top_1_pixel_correctness', 'N/A')}" if metrics.get('top_1_pixel_correctness') is not None else "")
                + (f" | correct_shapes={metrics.get('correct_shapes')}" if metrics.get('correct_shapes') is not None else "")
                + (f" | pixel_correctness={metrics.get('pixel_correctness')}" if metrics.get('pixel_correctness') is not None else "")
                + (f" | loss={metrics.get('total_final_loss', 'N/A')}" if metrics.get('total_final_loss') is not None else "")
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
                        # Try to extract loss data from the saved trajectory file for retry
                        trajectory_path = f"temp_trajectories/{method}_{artifact_path.split('/')[-1]}.npz"
                        if os.path.exists(trajectory_path):
                            extracted_loss = extract_loss_from_trajectory_file(trajectory_path, method)
                            if extracted_loss is not None:
                                retry_metrics["total_final_loss"] = extracted_loss
                                print(f"📊 Extracted loss from trajectory (retry): {extracted_loss:.6f}")
                            else:
                                print(f"⚠️  Could not extract loss from trajectory file (retry)")
                        else:
                            print(f"⚠️  Trajectory file not found (retry): {trajectory_path}")
                        
                        print(
                            f"✅ {method} evaluation (retry) completed successfully"
                            + (f" | accuracy={retry_acc}" if retry_acc is not None else "")
                            + (f" | shape_acc={retry_metrics.get('top_1_shape_accuracy', 'N/A')}" if retry_metrics.get('top_1_shape_accuracy') is not None else "")
                            + (f" | pixel_acc={retry_metrics.get('top_1_pixel_correctness', 'N/A')}" if retry_metrics.get('top_1_pixel_correctness') is not None else "")
                            + (f" | correct_shapes={retry_metrics.get('correct_shapes')}" if retry_metrics.get('correct_shapes') is not None else "")
                            + (f" | pixel_correctness={retry_metrics.get('pixel_correctness')}" if retry_metrics.get('pixel_correctness') is not None else "")
                            + (f" | loss={retry_metrics.get('total_final_loss', 'N/A')}" if retry_metrics.get('total_final_loss') is not None else "")
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


def setup_trajectory_storage():
    """Create temporary directory for trajectory storage and clean up old files."""
    temp_dir = Path("temp_trajectories")
    temp_dir.mkdir(exist_ok=True)
    
    # Clean up old trajectory files
    for old_file in temp_dir.glob("*.npz"):
        try:
            old_file.unlink()
            print(f"🧹 Cleaned up old trajectory file: {old_file}")
        except Exception as e:
            print(f"⚠️  Failed to clean up {old_file}: {e}")


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
    parser.add_argument("--dataset_length", type=int, default=32, help="Max examples to eval")
    parser.add_argument("--dataset_batch_size", type=int, default=8, help="Batch size for dataset eval")
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
    
    # Advanced Gradient Ascent parameters (from store_latent_search.py)
    parser.add_argument("--ga_lr", type=float, default=None,
                   help="Override learning rate (step size) for gradient_ascent")
    parser.add_argument("--ga_steps", type=int, default=None,
                   help="Override number of GA steps (overrides budget/2 calculation)")
    parser.add_argument("--ga_optimizer", type=str, default=None,
                   help="Override optimizer for gradient_ascent")
    parser.add_argument("--ga_lr_schedule", type=str, default=None,
                   help="Override LR schedule for gradient_ascent (true/false)")
    parser.add_argument("--ga_lr_schedule_exponent", type=float, default=None,
                   help="Override LR schedule exponent for gradient_ascent")
    parser.add_argument("--ga_accumulate_gradients_decoder_pairs", type=str, default=None,
                   help="Whether to accumulate gradients for decoder pairs in gradient_ascent (true/false)")
    parser.add_argument("--ga_scan_gradients_latents", type=str, default=None,
                   help="Whether to scan gradients for latents in gradient_ascent (true/false)")
    parser.add_argument("--ga_include_mean_latent", type=str, default=None,
                   help="Whether to include mean latent in gradient_ascent (true/false)")
    parser.add_argument("--ga_include_all_latents", type=str, default=None,
                   help="Whether to include all latents in gradient_ascent (true/false)")
    parser.add_argument("--ga_random_perturbation", type=str, default=None,
                   help="Random perturbation kwargs for gradient_ascent (JSON string)")
    parser.add_argument("--ga_track_progress", action="store_true",
                   help="Enable progress tracking for gradient_ascent")
    
    # Advanced Random Search parameters (from store_latent_search.py)
    parser.add_argument("--rs_scale", type=float, default=None,
                   help="Override scale for random_search")
    parser.add_argument("--rs_scan_batch_size", type=int, default=None,
                   help="Override scan batch size for random_search")
    parser.add_argument("--rs_random_search_seed", type=int, default=None,
                   help="Override random search seed for random_search")
    parser.add_argument("--rs_include_mean_latent", type=str, default=None,
                   help="Whether to include mean latent in random_search (true/false)")
    parser.add_argument("--rs_include_all_latents", type=str, default=None,
                   help="Whether to include all latents in random_search (true/false)")
    parser.add_argument("--rs_random_perturbation", type=str, default=None,
                   help="Random perturbation kwargs for random_search (JSON string)")
    parser.add_argument("--rs_track_progress", action="store_true",
                   help="Enable progress tracking for random_search")
    
    # Advanced Evolutionary Search parameters (from store_latent_search.py)
    parser.add_argument("--es_mutation_std", type=float, default=None,
                   help="Override mutation standard deviation for evolutionary_search")
    parser.add_argument("--es_population", type=int, default=None,
                   help="Override population size for evolutionary_search (overrides sqrt(budget) calculation)")
    parser.add_argument("--es_generations", type=int, default=None,
                   help="Override number of generations for evolutionary_search (overrides budget/pop calculation)")
    parser.add_argument("--es_mutation_decay", type=float, default=None,
                   help="Multiply mutation_std by this factor each generation (default: 0.95)")
    parser.add_argument("--es_elite_size", type=int, default=None,
                   help="Number of top candidates preserved each generation (default: population//2)")
    parser.add_argument("--es_track_progress", action="store_true",
                   help="Enable progress tracking for evolutionary_search")
    
    # Subspace evolutionary search parameters (from store_latent_search.py)
    parser.add_argument("--es_use_subspace_mutation", action="store_true",
                   help="Enable subspace mutation for evolutionary search")
    parser.add_argument("--es_subspace_dim", type=int, default=32,
                   help="Subspace dimension for evolutionary search (default: 32)")
    parser.add_argument("--es_ga_step_length", type=float, default=0.5,
                   help="Target GA step length for automatic sigma scaling (default: 0.5)")
    parser.add_argument("--es_trust_region_radius", type=float, default=None,
                   help="Trust region radius for evolutionary search (default: None)")
    
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
    
    # Loss vs accuracy plotting
    parser.add_argument("--loss", action="store_true",
                       help="Plot loss differences instead of accuracies (requires two methods in plot_methods)")
    
    # Budget configuration options
    parser.add_argument("--budget_start", type=int, default=1, 
                       help="Starting budget value (default: 1)")
    parser.add_argument("--budget_end", type=int, default=100, 
                       help="Ending budget value (default: 100)")
    parser.add_argument("--budget_period", type=int, default=25, 
                       help="Period between budget values (default: 25)")
    
    # Advanced optimization features (from store_latent_search.py)
    parser.add_argument("--n_samples", type=int, default=1,
                       help="Number of times to run each evaluation with different random seeds (for statistical analysis)")
    parser.add_argument("--aggregate_statistics", action="store_true",
                       help="Aggregate per-sample metrics across n_samples runs and generate statistical plots")
    parser.add_argument("--track_progress", action="store_true",
                       help="Enable progress tracking for all optimization methods")
    
    # Background visualization parameters (from store_latent_search.py)
    parser.add_argument("--background_resolution", type=int, default=400,
                       help="Base resolution for background heatmap (higher = smoother)")
    parser.add_argument("--background_smoothing", action="store_true",
                       help="Enable additional Gaussian smoothing for small-scale searches")
    parser.add_argument("--background_knn", type=int, default=5,
                       help="k-NN parameter for adaptive bandwidth (3-7 recommended)")
    parser.add_argument("--background_bandwidth_scale", type=float, default=1.25,
                       help="Bandwidth scaling factor (bigger = softer, more overlap)")
    parser.add_argument("--background_global_mix", type=float, default=0.05,
                       help="Global mixing strength (0.02-0.1 recommended, 0 to disable)")
    
    # Output and logging options
    parser.add_argument("--out_dir", type=str, default="results",
                       help="Output directory for results and plots")
    parser.add_argument("--no_files", action="store_true",
                       help="Disable file generation and plotting (faster, just return values)")
    
    # High-granularity evaluation options
    parser.add_argument("--high_granularity", action="store_true",
                       help="Enable high-granularity evaluation: only evaluate highest budget and extract full trajectory data for detailed heatmaps")
    parser.add_argument("--max_budget_only", action="store_true",
                       help="Only evaluate the maximum budget value (for high-granularity mode)")
    
    args = parser.parse_args()
    
    # Setup trajectory storage for loss extraction
    setup_trajectory_storage()
    
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
    
    # Apply high-granularity mode: only use highest budget for detailed trajectory extraction
    if args.high_granularity or args.max_budget_only:
        shared_budgets = [max(shared_budgets)] if shared_budgets else [args.budget_end]
        print(f"🔬 High-granularity mode: using only highest budget {shared_budgets[0]} for detailed trajectory extraction")
    
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
    
    # Validate loss plotting requirements
    if args.loss and len(args.plot_methods) != 2:
        print("❌ --loss flag requires exactly 2 methods in --plot_methods for loss difference plotting.")
        print(f"   Current methods: {args.plot_methods} (count: {len(args.plot_methods)})")
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
            
            # Gradient Ascent parameters
            "ga_lr": args.ga_lr,
            "ga_steps": args.ga_steps,
            "ga_optimizer": args.ga_optimizer,
            "ga_lr_schedule": args.ga_lr_schedule,
            "ga_lr_schedule_exponent": args.ga_lr_schedule_exponent,
            "ga_accumulate_gradients_decoder_pairs": args.ga_accumulate_gradients_decoder_pairs,
            "ga_scan_gradients_latents": args.ga_scan_gradients_latents,
            "ga_include_mean_latent": args.ga_include_mean_latent,
            "ga_include_all_latents": args.ga_include_all_latents,
            "ga_random_perturbation": args.ga_random_perturbation,
            "ga_track_progress": args.ga_track_progress,
            
            # Random Search parameters
            "rs_scale": args.rs_scale,
            "rs_scan_batch_size": args.rs_scan_batch_size,
            "rs_random_search_seed": args.rs_random_search_seed,
            "rs_include_mean_latent": args.rs_include_mean_latent,
            "rs_include_all_latents": args.rs_include_all_latents,
            "rs_random_perturbation": args.rs_random_perturbation,
            "rs_track_progress": args.rs_track_progress,
            
            # Evolutionary Search parameters
            "es_mutation_std": args.es_mutation_std,
            "es_population": args.es_population,
            "es_generations": args.es_generations,
            "es_mutation_decay": args.es_mutation_decay,
            "es_elite_size": args.es_elite_size,
            "es_track_progress": args.es_track_progress,
            
            # Subspace parameters
            "es_use_subspace_mutation": args.es_use_subspace_mutation,
            "es_subspace_dim": args.es_subspace_dim,
            "es_ga_step_length": args.es_ga_step_length,
            "es_trust_region_radius": args.es_trust_region_radius,
            
            # Advanced features
            "n_samples": args.n_samples,
            "aggregate_statistics": args.aggregate_statistics,
            "track_progress": args.track_progress,
            
            # Background visualization
            "background_resolution": args.background_resolution,
            "background_smoothing": args.background_smoothing,
            "background_knn": args.background_knn,
            "background_bandwidth_scale": args.background_bandwidth_scale,
            "background_global_mix": args.background_global_mix,
            
            # Plotting options
            "loss_plotting": args.loss,
            "plot_methods": args.plot_methods,
        },
    )

    # Mirror all subsequent prints to Weights & Biases terminal logs
    try:
        import builtins as _builtins

        _original_print = _builtins.print

        def _wandb_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            # Always print to the real stdout first
            _original_print(*args, **kwargs)
            try:
                msg = sep.join(str(a) for a in args) + end
                msg = msg.rstrip("\n")
                if hasattr(wandb, "termlog"):
                    wandb.termlog(msg)
                else:
                    # Fallback: log as a text line in the history
                    wandb.log({"logs/print": msg})
            except Exception:
                # Never fail the run due to logging issues
                pass

        _builtins.print = _wandb_print
    except Exception:
        # If installing the hook fails, continue without W&B print mirroring
        pass

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
            "accumulate_gradients_decoder_pairs": False,
            "scan_gradients_latents": False,
            "include_mean_latent": True,
            "include_all_latents": False,
            "random_perturbation": None,
            "track_progress": False,
        },
        "random_search": {
            "scale": 1.0,
            "scan_batch_size": 10,
            "random_search_seed": 0,
            "include_mean_latent": True,
            "include_all_latents": False,
            "random_perturbation": None,
            "track_progress": False,
        },
        "evolutionary_search": {
            "population_size": 32,
            "num_generations": 25,
            "mutation_std": 0.5,
            "mutation_decay": 0.95,
            "elite_size": None,  # Will be calculated as population//2
            "track_progress": False,
        },
    }

    # Apply CLI overrides if provided
    if args.ga_lr is not None:
        try:
            base_methods["gradient_ascent"]["lr"] = float(args.ga_lr)
            print(f"⚙️  Overriding gradient_ascent lr -> {base_methods['gradient_ascent']['lr']}")
        except Exception:
            pass
    if args.ga_steps is not None:
        try:
            base_methods["gradient_ascent"]["num_steps"] = int(args.ga_steps)
            print(f"⚙️  Overriding gradient_ascent num_steps -> {base_methods['gradient_ascent']['num_steps']}")
        except Exception:
            pass
    if args.ga_optimizer is not None:
        try:
            base_methods["gradient_ascent"]["optimizer"] = args.ga_optimizer
            print(f"⚙️  Overriding gradient_ascent optimizer -> {base_methods['gradient_ascent']['optimizer']}")
        except Exception:
            pass
    if args.ga_lr_schedule is not None:
        try:
            base_methods["gradient_ascent"]["lr_schedule"] = str(args.ga_lr_schedule).lower() == "true"
            print(f"⚙️  Overriding gradient_ascent lr_schedule -> {base_methods['gradient_ascent']['lr_schedule']}")
        except Exception:
            pass
    if args.ga_lr_schedule_exponent is not None:
        try:
            base_methods["gradient_ascent"]["lr_schedule_exponent"] = float(args.ga_lr_schedule_exponent)
            print(f"⚙️  Overriding gradient_ascent lr_schedule_exponent -> {base_methods['gradient_ascent']['lr_schedule_exponent']}")
        except Exception:
            pass
    if args.ga_accumulate_gradients_decoder_pairs is not None:
        try:
            base_methods["gradient_ascent"]["accumulate_gradients_decoder_pairs"] = str(args.ga_accumulate_gradients_decoder_pairs).lower() == "true"
            print(f"⚙️  Overriding gradient_ascent accumulate_gradients_decoder_pairs -> {base_methods['gradient_ascent']['accumulate_gradients_decoder_pairs']}")
        except Exception:
            pass
    if args.ga_scan_gradients_latents is not None:
        try:
            base_methods["gradient_ascent"]["scan_gradients_latents"] = str(args.ga_scan_gradients_latents).lower() == "true"
            print(f"⚙️  Overriding gradient_ascent scan_gradients_latents -> {base_methods['gradient_ascent']['scan_gradients_latents']}")
        except Exception:
            pass
    if args.ga_include_mean_latent is not None:
        try:
            base_methods["gradient_ascent"]["include_mean_latent"] = str(args.ga_include_mean_latent).lower() == "true"
            print(f"⚙️  Overriding gradient_ascent include_mean_latent -> {base_methods['gradient_ascent']['include_mean_latent']}")
        except Exception:
            pass
    if args.ga_include_all_latents is not None:
        try:
            base_methods["gradient_ascent"]["include_all_latents"] = str(args.ga_include_all_latents).lower() == "true"
            print(f"⚙️  Overriding gradient_ascent include_all_latents -> {base_methods['gradient_ascent']['include_all_latents']}")
        except Exception:
            pass
    if args.ga_random_perturbation is not None:
        try:
            base_methods["gradient_ascent"]["random_perturbation"] = args.ga_random_perturbation
            print(f"⚙️  Overriding gradient_ascent random_perturbation -> {base_methods['gradient_ascent']['random_perturbation']}")
        except Exception:
            pass
    if args.ga_track_progress or args.track_progress:
        base_methods["gradient_ascent"]["track_progress"] = True
        print(f"⚙️  Enabling gradient_ascent track_progress")
    
    # Random Search overrides
    if args.rs_scale is not None:
        try:
            base_methods["random_search"]["scale"] = float(args.rs_scale)
            print(f"⚙️  Overriding random_search scale -> {base_methods['random_search']['scale']}")
        except Exception:
            pass
    if args.rs_scan_batch_size is not None:
        try:
            base_methods["random_search"]["scan_batch_size"] = int(args.rs_scan_batch_size)
            print(f"⚙️  Overriding random_search scan_batch_size -> {base_methods['random_search']['scan_batch_size']}")
        except Exception:
            pass
    if args.rs_random_search_seed is not None:
        try:
            base_methods["random_search"]["random_search_seed"] = int(args.rs_random_search_seed)
            print(f"⚙️  Overriding random_search random_search_seed -> {base_methods['random_search']['random_search_seed']}")
        except Exception:
            pass
    if args.rs_include_mean_latent is not None:
        try:
            base_methods["random_search"]["include_mean_latent"] = str(args.rs_include_mean_latent).lower() == "true"
            print(f"⚙️  Overriding random_search include_mean_latent -> {base_methods['random_search']['include_mean_latent']}")
        except Exception:
            pass
    if args.rs_include_all_latents is not None:
        try:
            base_methods["random_search"]["include_all_latents"] = str(args.rs_include_all_latents).lower() == "true"
            print(f"⚙️  Overriding random_search include_all_latents -> {base_methods['random_search']['include_all_latents']}")
        except Exception:
            pass
    if args.rs_random_perturbation is not None:
        try:
            base_methods["random_search"]["random_perturbation"] = args.rs_random_perturbation
            print(f"⚙️  Overriding random_search random_perturbation -> {base_methods['random_search']['random_perturbation']}")
        except Exception:
            pass
    if args.rs_track_progress or args.track_progress:
        base_methods["random_search"]["track_progress"] = True
        print(f"⚙️  Enabling random_search track_progress")
    
    # Evolutionary Search overrides
    if args.es_mutation_std is not None:
        try:
            base_methods["evolutionary_search"]["mutation_std"] = float(args.es_mutation_std)
            print(f"⚙️  Overriding evolutionary_search mutation_std -> {base_methods['evolutionary_search']['mutation_std']}")
        except Exception:
            pass
    # FIXED: Only set population_size and num_generations if explicitly provided AND we want fixed values
    # Otherwise, let the budget-based scaling work automatically
    if args.es_population is not None and args.es_generations is not None:
        # User wants fixed values for all budgets
        base_methods["evolutionary_search"]["population_size"] = int(args.es_population)
        base_methods["evolutionary_search"]["num_generations"] = int(args.es_generations)
        print(f"⚙️  Using fixed ES configuration: population={args.es_population}, generations={args.es_generations}")
    else:
        # User wants automatic budget-based scaling - remove fixed values
        if "population_size" in base_methods["evolutionary_search"]:
            del base_methods["evolutionary_search"]["population_size"]
        if "num_generations" in base_methods["evolutionary_search"]:
            del base_methods["evolutionary_search"]["num_generations"]
        print(f"⚙️  Using automatic budget-based ES scaling with multiplier: {args.es_budget_multiplier}x")
    if args.es_mutation_decay is not None:
        try:
            base_methods["evolutionary_search"]["mutation_decay"] = float(args.es_mutation_decay)
            print(f"⚙️  Overriding evolutionary_search mutation_decay -> {base_methods['evolutionary_search']['mutation_decay']}")
        except Exception:
            pass
    if args.es_elite_size is not None:
        try:
            base_methods["evolutionary_search"]["elite_size"] = int(args.es_elite_size)
            print(f"⚙️  Overriding evolutionary_search elite_size -> {base_methods['evolutionary_search']['elite_size']}")
        except Exception:
            pass
    if args.es_track_progress or args.track_progress:
        base_methods["evolutionary_search"]["track_progress"] = True
        print(f"⚙️  Enabling evolutionary_search track_progress")
    
    # Evolutionary search budget: balance population and generations first.
    # Choose population ≈ sqrt(budget), enforce at least 3 and cap at 100, then set generations = ceil(budget / population)
    # Apply budget multiplier to scale both population and generations
    # If explicit population/generations are provided, use those instead
    es_configs = []  # list of {budget, population_size, num_generations}
    # FIXED: Only get max_pop if it exists, otherwise use reasonable default
    max_pop = base_methods["evolutionary_search"].get("population_size", 100)
    
    # Check if explicit values are provided
    explicit_pop = base_methods["evolutionary_search"].get("population_size")
    explicit_gens = base_methods["evolutionary_search"].get("num_generations")
    
    if explicit_pop is not None and explicit_gens is not None:
        print(f"⚙️  Using explicit ES configuration: population={explicit_pop}, generations={explicit_gens}")
        for b in shared_budgets:
            es_configs.append({
                "budget": int(b),
                "scaled_budget": b,  # No scaling when using explicit values
                "population_size": int(explicit_pop),
                "num_generations": int(explicit_gens)
            })
    else:
        print(f"⚙️  Using budget-based ES configuration with multiplier: {args.es_budget_multiplier}x")
        for b in shared_budgets:
            # Apply budget multiplier
            scaled_budget = b * args.es_budget_multiplier
            proposed_pop = int(round(np.sqrt(scaled_budget)))
            # FIXED: Use a reasonable max_pop when not explicitly set
            max_pop_reasonable = 100  # Allow larger populations for higher budgets
            proposed_pop = max(3, min(max_pop_reasonable, proposed_pop))
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"eval_{args.run_name}_{timestamp}.csv"
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

    # Open CSV file for writing
    f_csv = out_csv.open("a", newline="")
    writer = csv.writer(f_csv)
    
    if write_header:
        # Always include all possible columns for consistent CSV structure
        csv_headers = ["timestamp", "run_name", "checkpoint_name", "checkpoint_step", "method", "budget_type", "budget", 
                      "overall_accuracy", "top_1_shape_accuracy", "top_1_accuracy", "top_1_pixel_correctness",
                      "top_2_shape_accuracy", "top_2_accuracy", "top_2_pixel_correctness",
                      "total_final_loss"]
        
        # Add sample information if multiple samples
        if args.n_samples > 1:
            csv_headers.extend(["sample_number", "sample_seed"])
        
        # Add subspace parameters if enabled
        if args.es_use_subspace_mutation:
            csv_headers.extend(["subspace_enabled", "subspace_dim", "ga_step_length", "trust_region_radius"])
        
        writer.writerow(csv_headers)

    # Iterate checkpoints
    for i, checkpoint in enumerate(checkpoints, 1):
        step = checkpoint["step"]
        if step is None:
            print(f"⚠️  Skipping checkpoint {checkpoint['name']} (no step info)")
            continue
        
        # Handle multiple samples if requested
        sample_seeds = [args.dataset_seed]
        if args.n_samples > 1:
            sample_seeds = [args.dataset_seed + run_idx for run_idx in range(args.n_samples)]
            print(f"🧪 Running {args.n_samples} samples with seeds: {sample_seeds}")
        
        for sample_idx, sample_seed in enumerate(sample_seeds):
            if args.n_samples > 1:
                print(f"\n🔬 Sample {sample_idx + 1}/{args.n_samples} with seed {sample_seed}")
                # Update dataset seed for this sample
                current_dataset_seed = sample_seed
            else:
                current_dataset_seed = args.dataset_seed
            
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
            
            # Calculate training progress percentage based on actual checkpoint versions
            # Find the maximum training progress value across all checkpoints
            max_training_progress = 0
            for cp in checkpoints:
                cp_name = cp["name"]
                if "--checkpoint:" in cp_name:
                    try:
                        version_part = cp_name.split("--checkpoint:")[1]
                        version_num = int(version_part[1:])
                        max_training_progress = max(max_training_progress, version_num)
                    except ValueError:
                        pass
            
            denom = max(max_training_progress, 1)
            pct = int((training_progress / denom) * 100) if denom > 0 else 0

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
                        
                        # Use explicit ga_steps if provided, otherwise calculate from budget
                        if base_methods["gradient_ascent"].get("num_steps") is not None:
                            num_steps = base_methods["gradient_ascent"]["num_steps"]
                            print(f"   📊 Using explicit GA steps: {num_steps} (budget calculation overridden)")
                        else:
                            num_steps = int(np.ceil(scaled_budget / 2))
                            print(f"   📊 Calculated GA steps: {num_steps} from budget {compute_budget}")
                        
                        method_kwargs = dict(base_methods["gradient_ascent"])
                        method_kwargs["num_steps"] = num_steps

                        # Log evaluation start
                        budget_info = {"type": "budget", "value": compute_budget, "num_steps": num_steps, "scaled_budget": scaled_budget}
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step, args)

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
                            dataset_seed=current_dataset_seed,
                            args=args,
                        )

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

                        if ok:
                            results["method_results"]["gradient_ascent"]["success"] += 1
                            results["successful_evals"] += 1
                            
                            # Simplified W&B logging: training_step_{i}/ with budget vs best loss
                            try:
                                log_data = {
                                    f"training_step_{step}/ga/budget_{compute_budget}/best_loss": metrics.get("total_final_loss", 0.0) or 0.0,
                                    f"training_step_{step}/ga/budget_{compute_budget}/accuracy": acc or 0.0,
                                    f"training_step_{step}/ga/budget_{compute_budget}/execution_time": execution_time,
                                }
                                wandb.log(log_data)
                            except Exception as e:
                                print(f"⚠️  Failed to log to W&B: {e}")
                        else:
                            results["method_results"]["gradient_ascent"]["failed"] += 1
                            results["failed_evals"] += 1

                        # Prepare CSV row with subspace parameters if enabled
                        csv_row = [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], training_progress, "gradient_ascent", "budget", compute_budget, 
                                  acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                                  metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                                  metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", ""),
                                  metrics.get("total_final_loss", "")]
                        
                        # Add sample information if multiple samples
                        if args.n_samples > 1:
                            csv_row.extend([sample_idx + 1, sample_seed])
                        
                        if args.es_use_subspace_mutation:
                            csv_row.extend([False, "", "", ""])  # Not applicable for gradient ascent
                        
                        writer.writerow(csv_row)
                        f_csv.flush()  # Ensure data is written to disk immediately
                        print(f"📝 CSV: Written row for {method} budget {compute_budget} -> loss: {metrics.get('total_final_loss', 'N/A')}")
                elif method == "random_search":
                    print("\n🔧 Testing random_search across budgets...")
                    for num_samples in rs_samples:
                        method_kwargs = dict(base_methods["random_search"])
                        method_kwargs["num_samples"] = num_samples

                        # Log evaluation start
                        budget_info = {"type": "num_samples", "value": num_samples}
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step, args)

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
                            dataset_seed=current_dataset_seed,
                            args=args,
                        )

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

                        if ok:
                            results["method_results"]["random_search"]["success"] += 1
                            results["successful_evals"] += 1
                            
                            # Simplified W&B logging: training_step_{i}/ with budget vs best loss
                            try:
                                log_data = {
                                    f"training_step_{step}/rs/budget_{num_samples}/best_loss": metrics.get("total_final_loss", 0.0) or 0.0,
                                    f"training_step_{step}/rs/budget_{num_samples}/accuracy": acc or 0.0,
                                    f"training_step_{step}/rs/budget_{num_samples}/execution_time": execution_time,
                                }
                                wandb.log(log_data)
                            except Exception as e:
                                print(f"⚠️  Failed to log to W&B: {e}")
                        else:
                            results["method_results"]["random_search"]["failed"] += 1
                            results["failed_evals"] += 1

                        # Prepare CSV row with subspace parameters if enabled
                        csv_row = [time.strftime("%Y-%m-%d %H:%M:%S"), args.run_name, checkpoint["name"], training_progress, "random_search", "num_samples", num_samples, 
                                  acc or "", metrics.get("top_1_shape_accuracy", ""), metrics.get("top_1_accuracy", ""),
                                  metrics.get("top_1_pixel_correctness", ""), metrics.get("top_2_shape_accuracy", ""),
                                  metrics.get("top_2_accuracy", ""), metrics.get("top_2_pixel_correctness", ""),
                                  metrics.get("total_final_loss", "")]
                        
                        # Add sample information if multiple samples
                        if args.n_samples > 1:
                            csv_row.extend([sample_idx + 1, sample_seed])
                        
                        if args.es_use_subspace_mutation:
                            csv_row.extend([False, "", "", ""])  # Not applicable for random search
                        
                        writer.writerow(csv_row)
                        f_csv.flush()  # Ensure data is written to disk immediately
                        print(f"📝 CSV: Written row for {method} budget {es_cfg['budget']} -> loss: {metrics.get('total_final_loss', 'N/A')}")
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
                            "num_generations": es_cfg["num_generations"],
                        }
                        log_evaluation_start(method, budget_info, method_kwargs, checkpoint["name"], step, args)

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
                            dataset_seed=current_dataset_seed,
                            args=args,
                        )

                        # Log evaluation results and summary
                        log_evaluation_results(method, metrics, execution_time, ok)
                        summary = log_evaluation_summary(checkpoint["name"], step, method, budget_info, ok, execution_time)

                        if ok:
                            results["method_results"]["evolutionary_search"]["success"] += 1
                            results["successful_evals"] += 1

                            # Simplified W&B logging: training_step_{i}/ with budget vs best loss
                            try:
                                log_data = {
                                    f"training_step_{step}/es/budget_{es_cfg['budget']}/best_loss": metrics.get("total_final_loss", 0.0) or 0.0,
                                    f"training_step_{step}/es/budget_{es_cfg['budget']}/accuracy": acc or 0.0,
                                    f"training_step_{step}/es/budget_{es_cfg['budget']}/execution_time": execution_time,
                                    f"training_step_{step}/es/budget_{es_cfg['budget']}/population_size": es_cfg["population_size"],
                                    f"training_step_{step}/es/budget_{es_cfg['budget']}/num_generations": es_cfg["num_generations"],
                                }
                                
                                # Add subspace parameters if enabled
                                if args.es_use_subspace_mutation:
                                    log_data[f"training_step_{step}/es/budget_{es_cfg['budget']}/subspace_enabled"] = True
                                    log_data[f"training_step_{step}/es/budget_{es_cfg['budget']}/subspace_dim"] = args.es_subspace_dim
                                    log_data[f"training_step_{step}/es/budget_{es_cfg['budget']}/ga_step_length"] = args.es_ga_step_length
                                    if args.es_trust_region_radius is not None:
                                        log_data[f"training_step_{step}/es/budget_{es_cfg['budget']}/trust_region_radius"] = args.es_trust_region_radius
                                else:
                                    log_data[f"training_step_{step}/es/budget_{es_cfg['budget']}/subspace_enabled"] = False
                                
                                wandb.log(log_data)
                            except Exception as e:
                                print(f"⚠️  Failed to log to W&B: {e}")
                        else:
                            results["method_results"]["evolutionary_search"]["failed"] += 1
                            results["failed_evals"] += 1

                        # Prepare CSV row with subspace parameters if enabled
                        csv_row = [
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            args.run_name,
                            checkpoint["name"],
                            training_progress,
                            "evolutionary_search",
                            "budget",
                            es_cfg["budget"],
                            acc or "",
                            metrics.get("top_1_shape_accuracy", ""),
                            metrics.get("top_1_accuracy", ""),
                            metrics.get("top_1_pixel_correctness", ""),
                            metrics.get("top_2_shape_accuracy", ""),
                            metrics.get("top_2_accuracy", ""),
                            metrics.get("top_2_pixel_correctness", ""),
                            metrics.get("total_final_loss", ""),
                        ]

                        # Add sample information if multiple samples
                        if args.n_samples > 1:
                            csv_row.extend([sample_idx + 1, sample_seed])

                        if args.es_use_subspace_mutation:
                            csv_row.extend([True, args.es_subspace_dim, args.es_ga_step_length, args.es_trust_region_radius or ""])

                        writer.writerow(csv_row)
                        f_csv.flush()  # Ensure data is written to disk immediately
                        print(f"📝 CSV: Written row for evolutionary_search budget {es_cfg['budget']} -> loss: {metrics.get('total_final_loss', 'N/A')}")

                # Perform statistical analysis if both GA and ES were evaluated
                if ("gradient_ascent" in args.plot_methods and "evolutionary_search" in args.plot_methods and 
                    args.dataset_length and args.dataset_length > 1):
                    
                    # Find the trajectory files for this checkpoint
                    checkpoint_name = checkpoint["name"]
                    ga_trajectory_path = f"temp_trajectories/gradient_ascent_{checkpoint_name}.npz"
                    es_trajectory_path = f"temp_trajectories/evolutionary_search_{checkpoint_name}.npz"
                    
                    # Compute statistical analysis (final values)
                    try:
                        stats_results = compute_statistical_analysis(
                            ga_trajectory_path, 
                            es_trajectory_path, 
                            args.dataset_length
                        )
                        
                        if stats_results:
                            # Log statistical analysis results to W&B
                            try:
                                log_data = {}
                                for metric in ['accuracy', 'shape_correctness', 'pixel_correctness', 'best_loss']:
                                    if f"{metric}_pvalue" in stats_results:
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/pvalue"] = stats_results[f"{metric}_pvalue"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/statistic"] = stats_results[f"{metric}_statistic"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/test_type"] = stats_results[f"{metric}_test"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/mean_diff"] = stats_results[f"{metric}_mean_diff"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/cohens_dz"] = stats_results[f"{metric}_cohens_dz"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/ci_low"] = stats_results[f"{metric}_ci_low"]
                                        log_data[f"training_step_{step}/statistical_analysis/{metric}/ci_high"] = stats_results[f"{metric}_ci_high"]
                                        
                                        # Log with budget on x-axis as requested
                                        for budget in ga_budgets + [es_cfg["budget"] for es_cfg in es_configs]:
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/pvalue"] = stats_results[f"{metric}_pvalue"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/statistic"] = stats_results[f"{metric}_statistic"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/test_type"] = stats_results[f"{metric}_test"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/mean_diff"] = stats_results[f"{metric}_mean_diff"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/cohens_dz"] = stats_results[f"{metric}_cohens_dz"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/ci_low"] = stats_results[f"{metric}_ci_low"]
                                            log_data[f"statistical_analysis/{metric}/checkpoint_{step}/budget_{budget}/ci_high"] = stats_results[f"{metric}_ci_high"]
                                
                                wandb.log(log_data)
                                print(f"📊 Logged final statistical analysis results for checkpoint {step} to W&B")
                            except Exception as e:
                                print(f"⚠️  Failed to log final statistical analysis to W&B: {e}")
                        else:
                            print(f"⚠️  No final statistical analysis results computed for checkpoint {step}")
                    except Exception as e:
                        print(f"⚠️  Failed to compute final statistical analysis for checkpoint {step}: {e}")
                    
                    # Compute per-budget statistical analysis
                    try:
                        per_budget_stats = compute_statistical_analysis_per_budget(
                            ga_trajectory_path, 
                            es_trajectory_path, 
                            args.dataset_length
                        )
                        
                        if per_budget_stats:
                            # Log per-budget statistical analysis results to W&B with budget as x-axis tracker
                            try:
                                # Group statistics by metric and stat_type for time series logging
                                metric_groups = {}
                                
                                for key, value in per_budget_stats.items():
                                    # Parse the key to extract metric and budget info
                                    if "_budget_" in key:
                                        parts = key.split("_budget_")
                                        if len(parts) == 2:
                                            metric_part = parts[0]
                                            budget_part = int(parts[1].split("_")[0])
                                            stat_type = "_".join(parts[1].split("_")[1:])
                                            
                                            # Create metric group key
                                            group_key = f"{metric_part}_{stat_type}"
                                            if group_key not in metric_groups:
                                                metric_groups[group_key] = []
                                            
                                            metric_groups[group_key].append((budget_part, value))
                                
                                # Log each metric group as a time series with budget as x-axis
                                for group_key, budget_value_pairs in metric_groups.items():
                                    # Sort by budget for proper time series
                                    budget_value_pairs.sort(key=lambda x: x[0])
                                    
                                    # Create time series data
                                    budgets = [pair[0] for pair in budget_value_pairs]
                                    values = [pair[1] for pair in budget_value_pairs]
                                    
                                    # Log as time series with budget as x-axis
                                    for budget, value in budget_value_pairs:
                                        wandb.log({
                                            f"per_budget_statistical_analysis/{group_key}/checkpoint_{step}": value,
                                            "budget": budget
                                        })
                                
                                # Also log the traditional format for backward compatibility
                                log_data = {}
                                for key, value in per_budget_stats.items():
                                    if "_budget_" in key:
                                        parts = key.split("_budget_")
                                        if len(parts) == 2:
                                            metric_part = parts[0]
                                            budget_part = parts[1].split("_")[0]
                                            stat_type = "_".join(parts[1].split("_")[1:])
                                            
                                            # Log with budget on x-axis as requested
                                            log_data[f"per_budget_statistical_analysis/{metric_part}/checkpoint_{step}/budget_{budget_part}/{stat_type}"] = value
                                
                                wandb.log(log_data)
                                print(f"📊 Logged per-budget statistical analysis results for checkpoint {step} to W&B ({len(per_budget_stats)} metrics) with budget as x-axis tracker")
                            except Exception as e:
                                print(f"⚠️  Failed to log per-budget statistical analysis to W&B: {e}")
                        else:
                            print(f"⚠️  No per-budget statistical analysis results computed for checkpoint {step}")
                    except Exception as e:
                        print(f"⚠️  Failed to compute per-budget statistical analysis for checkpoint {step}: {e}")

                # Progress update after each checkpoint
                total_evals = results["successful_evals"] + results["failed_evals"]
                selected_counts = []
                if "gradient_ascent" in args.plot_methods:
                    selected_counts.append(len(ga_budgets))
                if "random_search" in args.plot_methods:
                    selected_counts.append(len(rs_samples))
                if "evolutionary_search" in args.plot_methods:
                    selected_counts.append(len(es_configs))
                total_expected = sum(selected_counts) * args.n_samples  # Account for multiple samples
                print(f"\n📊 Checkpoint {i}/{len(checkpoints)} complete. Total evaluations: {total_evals}/{total_expected * i}")
                print(f"   ⏱️  Timing info available in W&B logs for each method and budget")
                if args.n_samples > 1:
                    print(f"   🧪 Multiple samples: {args.n_samples} runs per evaluation")
                
                # Debug: Check CSV file status
                try:
                    csv_size = out_csv.stat().st_size if out_csv.exists() else 0
                    print(f"   📝 CSV file size: {csv_size} bytes")
                    if csv_size > 0:
                        print(f"   📁 CSV file: {out_csv}")
                except Exception as e:
                    print(f"   ⚠️  Could not check CSV file status: {e}")

            
            # Generate and upload analysis figure for this checkpoint (unless --no_files is specified)
            if not args.no_files:
                try:
                    # Create simple budget vs best loss plot for ES and GA
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Use the consistent color palette
                    colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
                    
                    # Plot data for each method
                    for i, method in enumerate(args.plot_methods):
                        if method == "gradient_ascent":
                            # Get GA data from W&B logs (we'll need to collect it during evaluation)
                            method_label = "Gradient Ascent"
                            color = colors[0]  # Orange
                        elif method == "evolutionary_search":
                            method_label = "Evolutionary Search"
                            color = colors[2]  # Blue
                        elif method == "random_search":
                            method_label = "Random Search"
                            color = colors[1]  # Pink
                        else:
                            continue
                        
                        # For now, we'll create a placeholder plot
                        # In a real implementation, you'd collect the actual loss data
                        ax.plot([], [], marker='o', linewidth=2, markersize=8,
                               color=color, label=method_label, alpha=0.8)
                    
                    ax.set_xlabel("Budget", fontsize=14)
                    ax.set_ylabel("Best Loss", fontsize=14)
                    ax.set_title(f"Budget vs Best Loss - Checkpoint {step}\n"
                               f"Training Progress: {training_progress}/{len(checkpoints)-1}", fontsize=16)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=12)
                    ax.set_ylim(bottom=0)  # Loss is typically non-negative
                    
                    # Save figure
                    out_dir = Path("results")
                    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                    fig_path = out_dir / f"analysis_budget_vs_loss_checkpoint_{step}.png"
                    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    
                    if fig_path.exists():
                        # Upload to wandb under analysis_figure/ panel
                        try:
                            # Convert Path to string for wandb.Image to avoid any Path object issues
                            fig_path_str = str(fig_path)
                            wandb.log(
                                {
                                    f"analysis_figure/checkpoint_{step}_budget_vs_loss": wandb.Image(fig_path_str),
                                    f"analysis_figure/checkpoint_{step}_training_progress": training_progress,
                                    f"analysis_figure/checkpoint_{step}_total_checkpoints": len(checkpoints),
                                }
                            )
                            print(f"📊 Generated and uploaded analysis figure: {fig_path_str}")
                        except Exception as e:
                            print(f"⚠️  Failed to upload analysis figure to W&B: {e}")
                            print(f"   Error type: {type(e).__name__}")
                            print(f"   Error details: {str(e)}")
                            import traceback
                            print(f"   Traceback: {traceback.format_exc()}")
                    else:
                        print("⚠️  Failed to generate analysis figure")

                except Exception as e:
                    print(f"⚠️  Failed to generate or upload analysis figure: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error details: {str(e)}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
            else:
                print("📁 File generation disabled (--no_files flag)")

            # Generate and upload comparison plot for this step (unless --no_files is specified)
            if not args.no_files:
                try:
                    # Accumulate data from CSV for selected methods only
                    method_to_step_to_budget: Dict[str, Dict[int, Dict[int, float]]] = {}
                    for method in args.plot_methods:
                        method_to_step_to_budget[method] = {}

                    if out_csv.exists():
                        print(f"📖 Reading CSV file: {out_csv}")
                        csv_size = out_csv.stat().st_size
                        print(f"   📏 CSV file size: {csv_size} bytes")
                        
                        with out_csv.open("r") as f:
                            reader = csv.DictReader(f)
                            row_count = 0
                            for row in reader:
                                row_count += 1
                                if row_count <= 3:  # Show first 3 rows for debugging
                                    print(f"   📋 Row {row_count}: {dict(row)}")
                            print(f"   📊 Total rows read: {row_count}")
                            
                            # Reset file pointer to beginning
                            f.seek(0)
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
                                        if args.loss and len(args.plot_methods) == 2:
                                            # Use loss for loss difference plotting
                                            # This extracts the best (lowest) loss achieved by each method
                                            acc_val = (
                                                float(row["total_final_loss"])
                                                if row["total_final_loss"] not in ("", None)
                                                else np.nan
                                            )
                                        else:
                                            # Use accuracy for regular plotting
                                            acc_val = (
                                                float(row["overall_accuracy"])
                                                if row["overall_accuracy"] not in ("", None)
                                                else np.nan
                                            )
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

                            # SAFETY CHECK: Prevent extremely large arrays that would cause plotting issues
                            if len(all_steps) > 1000 or len(all_budgets) > 1000:
                                print(f"⚠️  WARNING: Extremely large data dimensions detected!")
                                print(f"   Steps: {len(all_steps)} (range: {min(all_steps)} to {max(all_steps)})")
                                print(f"   Budgets: {len(all_budgets)} (range: {min(all_budgets)} to {max(all_budgets)})")
                                print(f"   This would create a {len(all_budgets)}x{len(all_steps)} array = {len(all_budgets) * len(all_steps)} elements")
                                print(f"   Skipping plot generation to prevent memory/plotting issues")
                                continue

                            # Additional safety check for reasonable step values
                            if max(all_steps) > 10000:
                                print(f"⚠️  WARNING: Extremely large step numbers detected!")
                                print(f"   Max step: {max(all_steps)}")
                                print(f"   All steps: {all_steps[:10]}...")  # Show first 10
                                print(f"   Skipping plot generation due to unreasonable step values")
                                continue

                            # NEW: Check for extremely large step values that would cause plotting issues
                            # Even with few steps, if step values are huge, the plot will be too large
                            if max(all_steps) > 1000000:  # 1 million steps
                                print(f"⚠️  WARNING: Extremely large step values detected!")
                                print(f"   Max step: {max(all_steps)}")
                                print(f"   This would create a plot with width of {max(all_steps)} pixels")
                                print(f"   Skipping plot generation due to unreasonable step values")
                                continue

                            # NEW: Check total array size to prevent memory issues
                            total_elements = len(all_budgets) * len(all_steps)
                            if total_elements > 1000000:  # 1 million elements
                                print(f"⚠️  WARNING: Total array size too large for plotting!")
                                print(f"   Budgets: {len(all_budgets)} x Steps: {len(all_steps)} = {total_elements} elements")
                                print(f"   Skipping plot generation to prevent memory/plotting issues")
                                continue

                            # NEW: Filter steps to reasonable range if they're too spread out
                            if len(all_steps) > 1 and (max(all_steps) - min(all_steps)) > 10000:
                                print(f"⚠️  WARNING: Step range too large, filtering to reasonable subset")
                                print(f"   Original step range: {min(all_steps)} to {max(all_steps)}")
                                
                                # Take every nth step to reduce the range
                                step_range = max(all_steps) - min(all_steps)
                                if step_range > 100000:
                                    step_skip = max(1, step_range // 1000)  # Aim for ~1000 steps max
                                else:
                                    step_skip = max(1, step_range // 100)   # Aim for ~100 steps max
                                
                                filtered_steps = all_steps[::step_skip]
                                if len(filtered_steps) < 2:
                                    filtered_steps = [all_steps[0], all_steps[-1]]  # At least 2 points
                                
                                print(f"   Filtered to {len(filtered_steps)} steps with skip={step_skip}")
                                print(f"   New step range: {min(filtered_steps)} to {max(filtered_steps)}")
                                all_steps = filtered_steps

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
                                                method_arrays[method][k, j] = method_to_step_to_budget[method].get(s, {}).get(
                                                    b, np.nan
                                                )

                                # Create plot with selected methods
                                if len(args.plot_methods) == 2:
                                    if args.loss:
                                        # Loss difference plotting: show difference between methods (raw delta)
                                        # For loss, lower is better, so we show method_B - method_A (positive = method_A better)
                                        loss_diff = method_arrays[args.plot_methods[1]] - method_arrays[args.plot_methods[0]]
                                        
                                        print(f"📊 Creating checkpoint-level loss difference heatmap...")
                                        print(f"   Steps: {len(all_steps)} (range: {min(all_steps)} to {max(all_steps)})")
                                        print(f"   Budgets: {len(all_budgets)} (range: {min(all_budgets)} to {max(all_budgets)})")
                                        print(f"   Method A: {args.plot_methods[0]}")
                                        print(f"   Method B: {args.plot_methods[1]}")
                                        
                                        try:
                                            # Use the specialized loss difference visualization (raw delta)
                                            from visualization import visualize_loss_difference_heatmap
                                            fig = visualize_loss_difference_heatmap(
                                                steps=np.array(all_steps),
                                                budgets=np.array(all_budgets),
                                                loss_diff=loss_diff,
                                                method_A_name=args.plot_methods[0].replace("_", " ").title(),  # First method (typically GA)
                                                method_B_name=args.plot_methods[1].replace("_", " ").title(),  # Second method (typically ES)
                                            )
                                            print(f"✅ Successfully created checkpoint-level loss difference heatmap")
                                        except Exception as e:
                                            print(f"❌ Failed to create checkpoint-level loss difference heatmap: {e}")
                                            print(f"   Error type: {type(e).__name__}")
                                            import traceback
                                            print(f"   Traceback: {traceback.format_exc()}")
                                            # Fall back to regular comparison plot
                                            print(f"📊 Falling back to regular comparison plot...")
                                            fig = visualize_optimization_comparison(
                                                steps=np.array(all_steps),
                                                budgets=np.array(all_budgets),
                                                acc_A=loss_diff,  # This will be the loss difference
                                                acc_B=np.full_like(loss_diff, np.nan),  # Not used for difference plot
                                                method_A_name=f"Loss Diff ({args.plot_methods[1].replace('_', ' ').title()} - {args.plot_methods[0].replace('_', ' ').title()})",
                                                method_B_name="",
                                            )
                                    else:
                                        # Regular accuracy comparison
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

                                if args.loss and len(args.plot_methods) == 2:
                                    plot_type = "Loss Difference"
                                    plot_description = (
                                        f"Positive values = {args.plot_methods[0].replace('_', ' ').title()} better (lower loss)"
                                    )
                                else:
                                    plot_type = "Accuracy Comparison"
                                    plot_description = "Higher values = better performance"

                                if args.loss and len(args.plot_methods) == 2:
                                    # For loss difference plots, no title (clean heatmap)
                                    pass
                                else:
                                    # Regular title for accuracy comparisons
                                    fig.suptitle(
                                        f"{plot_type} - Accumulated Data\n"
                                        f"{plot_description}\n"
                                        f"Current Training Progress: {training_progress}/{denom} ({pct}%)\n"
                                        f"Checkpoint {i}/{len(checkpoints)} | Total Steps: {len(all_steps)}, Budgets: {len(all_budgets)}",
                                        fontsize=14,
                                        y=0.98,
                                    )

                                step_plot_path = out_dir / f"optim_comparison_accumulated_progress_{training_progress}.png"
                                fig.savefig(step_plot_path, dpi=200, bbox_inches="tight")
                                plt.close(fig)

                                # Count available data points without shadowing names
                                data_point_count = 0
                                for method_name, step_map in method_to_step_to_budget.items():
                                    for _step, budget_map in step_map.items():
                                        for v in budget_map.values():
                                            if not np.isnan(v):
                                                data_point_count += 1

                                wandb.log(
                                    {
                                        f"checkpoint_{training_progress}/optimization_comparison": wandb.Image(str(step_plot_path)),
                                        f"checkpoint_{training_progress}/plot_step": training_progress,
                                        f"checkpoint_{training_progress}/plot_checkpoint_number": i,
                                        f"checkpoint_{training_progress}/plot_total_checkpoints": len(checkpoints),
                                        f"checkpoint_{training_progress}/plot_available_steps": len(all_steps),
                                        f"checkpoint_{training_progress}/plot_available_budgets": len(all_budgets),
                                        f"checkpoint_{training_progress}/plot_accumulated_data": True,
                                    }
                                )

                                wandb.log(
                                    {
                                        "plot_progression/current_step": training_progress,
                                        "plot_progression/checkpoint_number": i,
                                        "plot_progression/total_checkpoints": len(checkpoints),
                                        "plot_progression/comparison_plot": wandb.Image(str(step_plot_path)),
                                        "plot_progression/available_data_points": data_point_count,
                                        "plot_progression/accumulated_steps": len(all_steps),
                                        "plot_progression/accumulated_budgets": len(all_budgets),
                                    }
                                )

                                print(
                                    f"📊 Generated and uploaded accumulated comparison plot for training progress {training_progress}/{denom} ({pct}%)"
                                )
                                print(f"   📈 Available steps: {all_steps}")
                                print(f"   💰 Available budgets: {all_budgets}")
                                print(f"   🔍 Data coverage: {data_point_count} data points")

                                # Generate per-checkpoint GA/ES single-method and difference heatmaps (overall & pixel)
                                # Use full trajectory data for high-granularity heatmaps instead of just CSV data
                                try:
                                    # Generate high-granularity heatmaps using full trajectory data
                                    
                                    # Extract full trajectory data for each method
                                    trajectory_data_by_method = {}
                                    for method in args.plot_methods:
                                        trajectory_path = f"temp_trajectories/{method}_{checkpoint['name']}.npz"
                                        if os.path.exists(trajectory_path):
                                            trajectory_data = extract_full_trajectory_data(
                                                trajectory_path, method, args.dataset_length or 1
                                            )
                                            if trajectory_data:
                                                trajectory_data_by_method[method] = trajectory_data
                                                # Extracted trajectory data for method
                                            else:
                                                print(f"⚠️  No trajectory data extracted for {method}")
                                        else:
                                            print(f"⚠️  Trajectory file not found for {method}: {trajectory_path}")
                                    
                                    # Build high-granularity arrays from trajectory data
                                    method_arrays_high_granularity = {}
                                    method_arrays_pixel_high_granularity = {}
                                    
                                    for method in args.plot_methods:
                                        if method in trajectory_data_by_method:
                                            traj_data = trajectory_data_by_method[method]
                                            
                                            # Get the actual budget points from trajectory (much higher granularity)
                                            if 'budget' in traj_data:
                                                traj_budgets = traj_data['budget']
                                                # Method trajectory budgets extracted
                                                
                                                # Create high-granularity arrays
                                                method_arrays_high_granularity[method] = np.full((len(traj_budgets), len(all_steps)), np.nan)
                                                method_arrays_pixel_high_granularity[method] = np.full((len(traj_budgets), len(all_steps)), np.nan)
                                                
                                                # Fill with trajectory data (only for current checkpoint step)
                                                current_step_idx = all_steps.index(training_progress) if training_progress in all_steps else 0
                                                
                                                # Ensure array dimensions match
                                                array_height = method_arrays_high_granularity[method].shape[0]
                                                
                                                # Overall accuracy/loss data
                                                if 'accuracy_mean' in traj_data:
                                                    data = traj_data['accuracy_mean']
                                                    # Truncate or pad to match array height
                                                    if len(data) > array_height:
                                                        data = data[:array_height]
                                                    elif len(data) < array_height:
                                                        data = np.pad(data, (0, array_height - len(data)), mode='constant', constant_values=np.nan)
                                                    method_arrays_high_granularity[method][:, current_step_idx] = data
                                                elif 'losses_mean' in traj_data:
                                                    # Convert losses to accuracy-like values (inverted)
                                                    losses = traj_data['losses_mean']
                                                    # Truncate or pad to match array height
                                                    if len(losses) > array_height:
                                                        losses = losses[:array_height]
                                                    elif len(losses) < array_height:
                                                        losses = np.pad(losses, (0, array_height - len(losses)), mode='constant', constant_values=np.nan)
                                                    # Normalize losses to 0-1 range for visualization
                                                    if losses.max() > losses.min():
                                                        normalized = 1.0 - (losses - losses.min()) / (losses.max() - losses.min())
                                                    else:
                                                        normalized = np.ones_like(losses) * 0.5
                                                    method_arrays_high_granularity[method][:, current_step_idx] = normalized
                                                
                                                # Pixel correctness data
                                                if 'pixel_correctness_mean' in traj_data:
                                                    data = traj_data['pixel_correctness_mean']
                                                    # Truncate or pad to match array height
                                                    if len(data) > array_height:
                                                        data = data[:array_height]
                                                    elif len(data) < array_height:
                                                        data = np.pad(data, (0, array_height - len(data)), mode='constant', constant_values=np.nan)
                                                    method_arrays_pixel_high_granularity[method][:, current_step_idx] = data
                                                
                                                # High-granularity array created
                                            else:
                                                print(f"⚠️  No budget data in trajectory for {method}")
                                        else:
                                            print(f"⚠️  No trajectory data available for {method}")
                                    
                                    # Fallback to CSV data if trajectory data is insufficient
                                    if not method_arrays_high_granularity:
                                        print("📊 Falling back to CSV data for heatmaps...")
                                        
                                        # Build pixel-correctness maps for selected methods from CSV
                                        method_to_step_to_budget_pixel: Dict[str, Dict[int, Dict[int, float]]] = {}
                                        for m in args.plot_methods:
                                            method_to_step_to_budget_pixel[m] = {}

                                        with out_csv.open("r") as f_pix:
                                            reader_pix = csv.DictReader(f_pix)
                                            for rowp in reader_pix:
                                                try:
                                                    row_stepp = int(rowp["checkpoint_step"]) if rowp["checkpoint_step"] else None
                                                except Exception:
                                                    row_stepp = None
                                                if row_stepp is None:
                                                    continue
                                                mth = rowp["method"]
                                                try:
                                                    bud = int(rowp["budget"]) if rowp["budget"] else None
                                                except Exception:
                                                    bud = None
                                                if bud is None:
                                                    continue
                                                try:
                                                    pix = float(rowp["top_1_pixel_correctness"]) if rowp["top_1_pixel_correctness"] not in ("", None) else np.nan
                                                except Exception:
                                                    pix = np.nan
                                                if mth in method_to_step_to_budget_pixel:
                                                    method_to_step_to_budget_pixel[mth].setdefault(row_stepp, {})[bud] = pix

                                        # Build arrays for pixel correctness aligned to all_steps/all_budgets
                                        method_arrays_pixel: Dict[str, np.ndarray] = {}
                                        for m in args.plot_methods:
                                            method_arrays_pixel[m] = np.full((len(all_budgets), len(all_steps)), np.nan)
                                        for j, s_ in enumerate(all_steps):
                                            for k, b_ in enumerate(all_budgets):
                                                for m in args.plot_methods:
                                                    if m in method_to_step_to_budget_pixel:
                                                        method_arrays_pixel[m][k, j] = method_to_step_to_budget_pixel[m].get(s_, {}).get(b_, np.nan)
                                        
                                        # Use CSV data as fallback
                                        method_arrays_high_granularity = method_arrays
                                        method_arrays_pixel_high_granularity = method_arrays_pixel

                                    # Helper to render heatmap with consistent color palette and statistical annotations
                                    def _save_heatmap(data: np.ndarray, steps_list: List[int], budgets_list: List[int], title: str, center: float | None, stats_data: Dict[str, Any] = None) -> Path | None:
                                        try:
                                            if data is None or np.all(np.isnan(data)):
                                                return None
                                            
                                            # Use the same color palette as the reference (cool colormap)
                                            fig_h, ax_h = plt.subplots(figsize=(12, 8))
                                            
                                            if center is not None:
                                                dmin = np.nanmin(data)
                                                dmax = np.nanmax(data)
                                                span = max(abs(dmin - center), abs(dmax - center))
                                                vmin, vmax = center - span, center + span
                                            else:
                                                vmax_abs = float(np.nanmax(np.abs(data))) if not np.all(np.isnan(data)) else 1.0
                                                vmin, vmax = -vmax_abs, vmax_abs
                                            
                                            im = ax_h.imshow(
                                                data,
                                                aspect="auto",
                                                origin="lower",
                                                extent=[min(steps_list), max(steps_list), min(budgets_list), max(budgets_list)],
                                                vmin=vmin,
                                                vmax=vmax,
                                                cmap="cool",  # Use same colormap as reference
                                            )
                                            
                                            ax_h.set_xlabel("Training Step", fontsize=12)
                                            ax_h.set_ylabel("Budget", fontsize=12)
                                            ax_h.set_title(title, fontsize=14)
                                            
                                            # Add colorbar
                                            cbar = fig_h.colorbar(im, ax=ax_h)
                                            cbar.ax.tick_params(length=3, pad=3)
                                            
                                            # Add statistical annotations if available
                                            if stats_data:
                                                # Create text box for statistical values
                                                stats_text = []
                                                
                                                # Extract relevant statistics based on the title
                                                if "ga_minus_es" in title:
                                                    # For difference plots, show comparison statistics
                                                    for metric in ['accuracy', 'pixel_correctness']:
                                                        if metric in title:
                                                            if f"{metric}_pvalue" in stats_data:
                                                                p_val = stats_data[f"{metric}_pvalue"]
                                                                stat_val = stats_data.get(f"{metric}_statistic", "N/A")
                                                                test_type = stats_data.get(f"{metric}_test", "N/A")
                                                                
                                                                stats_text.append(f"{metric.upper()}:")
                                                                stats_text.append(f"  p = {p_val:.4f}")
                                                                if "mcnemar" in test_type.lower():
                                                                    stats_text.append(f"  χ² = {stat_val:.4f}")
                                                                elif "ttest" in test_type.lower():
                                                                    stats_text.append(f"  t = {stat_val:.4f}")
                                                                elif "wilcoxon" in test_type.lower():
                                                                    stats_text.append(f"  W = {stat_val:.4f}")
                                                                stats_text.append("")
                                                else:
                                                    # For individual method plots, show method-specific stats
                                                    method = "GA" if "ga_" in title else "ES"
                                                    for metric in ['accuracy', 'pixel_correctness']:
                                                        if metric in title:
                                                            if f"{metric}_pvalue" in stats_data:
                                                                p_val = stats_data[f"{metric}_pvalue"]
                                                                stat_val = stats_data.get(f"{metric}_statistic", "N/A")
                                                                test_type = stats_data.get(f"{metric}_test", "N/A")
                                                                
                                                                stats_text.append(f"{method} {metric.upper()}:")
                                                                stats_text.append(f"  p = {p_val:.4f}")
                                                                if "mcnemar" in test_type.lower():
                                                                    stats_text.append(f"  χ² = {stat_val:.4f}")
                                                                elif "ttest" in test_type.lower():
                                                                    stats_text.append(f"  t = {stat_val:.4f}")
                                                                elif "wilcoxon" in test_type.lower():
                                                                    stats_text.append(f"  W = {stat_val:.4f}")
                                                                stats_text.append("")
                                                
                                                if stats_text:
                                                    # Add text box with statistical values
                                                    stats_str = "\n".join(stats_text)
                                                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                                                    ax_h.text(0.02, 0.98, stats_str, transform=ax_h.transAxes, fontsize=9,
                                                            verticalalignment='top', bbox=props)
                                            
                                            out_dir_local = Path("results")
                                            out_dir_local.mkdir(parents=True, exist_ok=True)
                                            safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title)
                                            out_path = out_dir_local / f"{safe_title}.png"
                                            fig_h.savefig(out_path, dpi=200, bbox_inches="tight")
                                            plt.close(fig_h)
                                            return out_path
                                        except Exception:
                                            return None

                                    step_tag = training_progress
                                    
                                    # Compute statistical analysis for heatmap annotations if dataset_length > 1
                                    heatmap_stats = {}
                                    if args.dataset_length and args.dataset_length > 1:
                                        try:
                                            # Try to find the most recent trajectory files for statistical analysis
                                            latest_checkpoint = None
                                            for checkpoint in checkpoints:
                                                if checkpoint["step"] <= training_progress:
                                                    latest_checkpoint = checkpoint
                                            
                                            if latest_checkpoint:
                                                ga_trajectory_path = f"temp_trajectories/gradient_ascent_{latest_checkpoint['name']}.npz"
                                                es_trajectory_path = f"temp_trajectories/evolutionary_search_{latest_checkpoint['name']}.npz"
                                                
                                                if os.path.exists(ga_trajectory_path) and os.path.exists(es_trajectory_path):
                                                    heatmap_stats = compute_statistical_analysis(
                                                        ga_trajectory_path, 
                                                        es_trajectory_path, 
                                                        args.dataset_length
                                                    )
                                                    print(f"📊 Computed statistical analysis for heatmap annotations: {len(heatmap_stats)} metrics")
                                        except Exception as e:
                                            print(f"⚠️  Failed to compute statistical analysis for heatmap annotations: {e}")
                                    
                                    # Use high-granularity data for heatmaps if available, otherwise fallback to CSV data
                                    arrays_to_use = method_arrays_high_granularity if method_arrays_high_granularity else method_arrays
                                    pixel_arrays_to_use = method_arrays_pixel_high_granularity if method_arrays_pixel_high_granularity else method_arrays_pixel
                                    
                                    # Debug: Print data availability
                                    print(f"🔍 DEBUG: Heatmap data availability:")
                                    print(f"   method_arrays_high_granularity: {bool(method_arrays_high_granularity)}")
                                    print(f"   method_arrays_pixel_high_granularity: {bool(method_arrays_pixel_high_granularity)}")
                                    print(f"   method_arrays: {bool(method_arrays)}")
                                    print(f"   method_arrays_pixel: {bool(method_arrays_pixel)}")
                                    print(f"   arrays_to_use keys: {list(arrays_to_use.keys()) if arrays_to_use else 'None'}")
                                    print(f"   pixel_arrays_to_use keys: {list(pixel_arrays_to_use.keys()) if pixel_arrays_to_use else 'None'}")
                                    
                                    # Debug: Check data content
                                    for method in args.plot_methods:
                                        if method in arrays_to_use:
                                            data = arrays_to_use[method]
                                            valid_count = np.sum(~np.isnan(data))
                                            total_count = data.size
                                            print(f"   {method} overall data: {data.shape}, {valid_count}/{total_count} valid values")
                                        if method in pixel_arrays_to_use:
                                            data = pixel_arrays_to_use[method]
                                            valid_count = np.sum(~np.isnan(data))
                                            total_count = data.size
                                            print(f"   {method} pixel data: {data.shape}, {valid_count}/{total_count} valid values")
                                    
                                    # Get the appropriate budget list for each method's data
                                    def get_budget_list_for_method(method: str) -> List[int]:
                                        if method_arrays_high_granularity and method in trajectory_data_by_method:
                                            # Use trajectory budgets for high granularity
                                            traj_budgets = trajectory_data_by_method[method].get('budget', all_budgets)
                                            budget_list = traj_budgets.tolist() if hasattr(traj_budgets, 'tolist') else traj_budgets
                                            return budget_list
                                        else:
                                            return all_budgets
                                    
                                    # Debug: Print budget information for each method
                                    for method in args.plot_methods:
                                        budget_list = get_budget_list_for_method(method)
                                        print(f"📊 {method} using {len(budget_list)} budget points")
                                    
                                    # GA overall accuracy heatmap (symmetric around 0)
                                    if "gradient_ascent" in arrays_to_use:
                                        print(f"🔍 DEBUG: Generating GA overall accuracy heatmap...")
                                        ga_budget_list = get_budget_list_for_method("gradient_ascent")
                                        p = _save_heatmap(arrays_to_use["gradient_ascent"], all_steps, ga_budget_list, f"checkpoint_{step_tag}_ga_overall_accuracy", center=0, stats_data=heatmap_stats)
                                        if p and p.exists():
                                            print(f"✅ DEBUG: GA overall accuracy heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/ga_overall_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: GA overall accuracy heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: GA not found in arrays_to_use")
                                    # ES overall accuracy heatmap (symmetric around 0)
                                    if "evolutionary_search" in arrays_to_use:
                                        print(f"🔍 DEBUG: Generating ES overall accuracy heatmap...")
                                        es_budget_list = get_budget_list_for_method("evolutionary_search")
                                        p = _save_heatmap(arrays_to_use["evolutionary_search"], all_steps, es_budget_list, f"checkpoint_{step_tag}_es_overall_accuracy", center=0, stats_data=heatmap_stats)
                                        if p and p.exists():
                                            print(f"✅ DEBUG: ES overall accuracy heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/es_overall_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: ES overall accuracy heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: ES not found in arrays_to_use")
                                    # GA pixel accuracy heatmap (symmetric around 0)
                                    if "gradient_ascent" in pixel_arrays_to_use:
                                        print(f"🔍 DEBUG: Generating GA pixel accuracy heatmap...")
                                        ga_budget_list = get_budget_list_for_method("gradient_ascent")
                                        p = _save_heatmap(pixel_arrays_to_use["gradient_ascent"], all_steps, ga_budget_list, f"checkpoint_{step_tag}_ga_pixel_accuracy", center=0, stats_data=heatmap_stats)
                                        if p and p.exists():
                                            print(f"✅ DEBUG: GA pixel accuracy heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/ga_pixel_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: GA pixel accuracy heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: GA not found in pixel_arrays_to_use")
                                    # ES pixel accuracy heatmap (symmetric around 0)
                                    if "evolutionary_search" in pixel_arrays_to_use:
                                        print(f"🔍 DEBUG: Generating ES pixel accuracy heatmap...")
                                        es_budget_list = get_budget_list_for_method("evolutionary_search")
                                        p = _save_heatmap(pixel_arrays_to_use["evolutionary_search"], all_steps, es_budget_list, f"checkpoint_{step_tag}_es_pixel_accuracy", center=0, stats_data=heatmap_stats)
                                        if p and p.exists():
                                            print(f"✅ DEBUG: ES pixel accuracy heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/es_pixel_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: ES pixel accuracy heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: ES not found in pixel_arrays_to_use")
                                    # GA − ES overall accuracy diff (center at 0)
                                    if "gradient_ascent" in arrays_to_use and "evolutionary_search" in arrays_to_use:
                                        print(f"🔍 DEBUG: Generating GA-ES overall accuracy difference heatmap...")
                                        ga_data = arrays_to_use["gradient_ascent"]
                                        es_data = arrays_to_use["evolutionary_search"]
                                        
                                        # Check if arrays have compatible shapes for subtraction
                                        if ga_data.shape == es_data.shape:
                                            diff_overall = ga_data - es_data
                                            # Use the budget list from the first method (GA) for consistency
                                            ga_budget_list = get_budget_list_for_method("gradient_ascent")
                                            p = _save_heatmap(diff_overall, all_steps, ga_budget_list, f"checkpoint_{step_tag}_ga_minus_es_overall_accuracy", center=0, stats_data=heatmap_stats)
                                        else:
                                            print(f"❌ DEBUG: Cannot subtract GA-ES arrays - GA shape: {ga_data.shape}, ES shape: {es_data.shape}")
                                            p = None
                                        if p and p.exists():
                                            print(f"✅ DEBUG: GA-ES overall accuracy difference heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/ga_minus_es_overall_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: GA-ES overall accuracy difference heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: Cannot generate GA-ES difference - GA in arrays: {'gradient_ascent' in arrays_to_use}, ES in arrays: {'evolutionary_search' in arrays_to_use}")
                                    # GA − ES pixel accuracy diff (center at 0)
                                    if "gradient_ascent" in pixel_arrays_to_use and "evolutionary_search" in pixel_arrays_to_use:
                                        print(f"🔍 DEBUG: Generating GA-ES pixel accuracy difference heatmap...")
                                        ga_pixel_data = pixel_arrays_to_use["gradient_ascent"]
                                        es_pixel_data = pixel_arrays_to_use["evolutionary_search"]
                                        
                                        # Check if arrays have compatible shapes for subtraction
                                        if ga_pixel_data.shape == es_pixel_data.shape:
                                            diff_pixel = ga_pixel_data - es_pixel_data
                                            # Use the budget list from the first method (GA) for consistency
                                            ga_budget_list = get_budget_list_for_method("gradient_ascent")
                                            p = _save_heatmap(diff_pixel, all_steps, ga_budget_list, f"checkpoint_{step_tag}_ga_minus_es_pixel_accuracy", center=0, stats_data=heatmap_stats)
                                        else:
                                            print(f"❌ DEBUG: Cannot subtract GA-ES pixel arrays - GA shape: {ga_pixel_data.shape}, ES shape: {es_pixel_data.shape}")
                                            p = None
                                        if p and p.exists():
                                            print(f"✅ DEBUG: GA-ES pixel accuracy difference heatmap saved to {p}")
                                            wandb.log({f"checkpoint_{training_progress}/ga_minus_es_pixel_accuracy": wandb.Image(str(p))})
                                        else:
                                            print(f"❌ DEBUG: GA-ES pixel accuracy difference heatmap generation failed (p={p})")
                                    else:
                                        print(f"❌ DEBUG: Cannot generate GA-ES pixel difference - GA in pixel_arrays: {'gradient_ascent' in pixel_arrays_to_use}, ES in pixel_arrays: {'evolutionary_search' in pixel_arrays_to_use}")
                                    print("📊 Generated and uploaded per-checkpoint GA/ES heatmaps (overall/pixel and diffs)")
                                except Exception as e:
                                    print(f"⚠️  Failed to generate per-checkpoint GA/ES heatmaps: {e}")

                                # Generate additional loss plots if --loss flag is enabled
                                if args.loss and len(args.plot_methods) == 2:
                                    try:
                                        # Generate Loss vs Budget plot
                                        loss_budget_plot_path = generate_loss_vs_budget_plot(
                                            method_arrays=method_arrays,
                                            budgets=all_budgets,
                                            method_names=args.plot_methods,
                                            checkpoint_name=checkpoint["name"],
                                            checkpoint_step=step,
                                        )

                                        # Generate Loss vs Training Progress plot
                                        loss_training_plot_path = generate_loss_vs_training_plot(
                                            method_arrays=method_arrays,
                                            steps=all_steps,
                                            method_names=args.plot_methods,
                                            checkpoint_name=checkpoint["name"],
                                            checkpoint_step=step,
                                            total_checkpoints=len(checkpoints),
                                        )
                                        
                                        # Generate new budget-based plots (similar to store_latent_search.py)
                                        if args.dataset_folder and args.dataset_length > 0:
                                            try:
                                                # Find GA and ES NPZ files for this checkpoint
                                                ga_npz_path = None
                                                es_npz_path = None
                                                
                                                # Find GA and ES NPZ files for this checkpoint
                                                ga_trajectory_path = f"temp_trajectories/gradient_ascent_{checkpoint['name']}.npz"
                                                es_trajectory_path = f"temp_trajectories/evolutionary_search_{checkpoint['name']}.npz"
                                                
                                                if os.path.exists(ga_trajectory_path):
                                                    ga_npz_path = ga_trajectory_path
                                                if os.path.exists(es_trajectory_path):
                                                    es_npz_path = es_trajectory_path
                                                
                                                if ga_npz_path and es_npz_path:
                                                    budget_plots = generate_budget_based_plots(
                                                        ga_npz_path=ga_npz_path,
                                                        es_npz_path=es_npz_path,
                                                        out_dir=out_dir,
                                                        dataset_length=args.dataset_length,
                                                        checkpoint_name=checkpoint["name"],
                                                        checkpoint_step=step
                                                    )
                                                    print(f"📊 Generated budget-based plots: {list(budget_plots.keys())}")
                                                else:
                                                    print("⚠️  Missing GA or ES NPZ files for budget-based plotting")
                                            except Exception as e:
                                                print(f"⚠️  Failed to generate budget-based plots: {e}")

                                        # Upload both plots to W&B
                                        if loss_budget_plot_path and loss_training_plot_path:
                                            try:
                                                wandb.log(
                                                    {
                                                        f"checkpoint_{training_progress}/loss_vs_budget": wandb.Image(
                                                            str(loss_budget_plot_path)
                                                        ),
                                                        f"checkpoint_{training_progress}/loss_vs_training": wandb.Image(
                                                            str(loss_training_plot_path)
                                                        ),
                                                    }
                                                )
                                                print("📊 Generated and uploaded loss plots:")
                                                print(f"   • Loss vs Budget: {loss_budget_plot_path}")
                                                print(f"   • Loss vs Training: {loss_training_plot_path}")
                                            except Exception as e:
                                                print(f"⚠️  Failed to upload loss plots to W&B: {e}")
                                                print(f"   Error type: {type(e).__name__}")
                                                print(f"   Error details: {str(e)}")
                                        else:
                                            print("⚠️  Failed to generate one or both loss plots")

                                    except Exception as e:
                                        print(f"⚠️  Failed to generate loss plots: {e}")
                                        print(f"   Error type: {type(e).__name__}")
                                        print(f"   Error details: {str(e)}")
                                        import traceback
                                        print(f"   Traceback: {traceback.format_exc()}")

                except Exception as e:
                    print(f"⚠️  Failed to generate comparison plot for training progress {training_progress}: {e}")
            else:
                print("📁 Comparison plot generation disabled (--no_files flag)")

            
            # Log checkpoint completion to W&B
            try:
                wandb.log(
                    {
                        f"checkpoint_{training_progress}/completion": 1.0,
                        f"checkpoint_{training_progress}/total_evaluations": total_evals,
                        f"checkpoint_{training_progress}/successful_evaluations": results["successful_evals"],
                        f"checkpoint_{training_progress}/failed_evaluations": results["failed_evals"],
                    }
                )

                overall_progress = i / len(checkpoints)
                wandb.log(
                    {
                        "overall/progress": overall_progress,
                        "overall/checkpoints_completed": i,
                        "overall/total_checkpoints": len(checkpoints),
                        "overall/total_evaluations": total_evals,
                        "overall/successful_evaluations": results["successful_evals"],
                        "overall/failed_evaluations": results["failed_evals"],
                    }
                )
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint completion to W&B: {e}")

            # Close sample loop
            if args.n_samples > 1:
                print(f"🔬 Completed {args.n_samples} samples for checkpoint {i}/{len(checkpoints)}")

            # Upload CSV artifact
            try:
                artifact = wandb.Artifact(f"{args.run_name}--budgets-eval", type="evaluation")
                artifact.add_file(str(out_csv))
                run.log_artifact(artifact)
            except Exception as e:
                print(f"⚠️  Failed to upload CSV artifact: {e}")


            # Build final optimization comparison plot from CSV (overall summary) (unless --no_files is specified)
            if not args.no_files:
                try:
                    steps_list: List[int] = []
                    method_maps: Dict[str, Dict[int, Dict[int, float]]] = {}
                    for method in args.plot_methods:
                        method_maps[method] = {}

                    print(f"📖 Reading final CSV file: {out_csv}")
                    csv_size = out_csv.stat().st_size
                    print(f"   📏 CSV file size: {csv_size} bytes")
                    
                    with out_csv.open("r") as f:
                        reader = csv.DictReader(f)
                        row_count = 0
                        for row in reader:
                            row_count += 1
                            if row_count <= 3:  # Show first 3 rows for debugging
                                print(f"   📋 Row {row_count}: {dict(row)}")
                        print(f"   📊 Total rows read: {row_count}")
                        
                        # Reset file pointer to beginning
                        f.seek(0)
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
                                if args.loss and len(args.plot_methods) == 2:
                                    # Use loss for loss difference plotting
                                    loss_val = float(row["total_final_loss"]) if row["total_final_loss"] not in ("", None) else np.nan
                                    acc_val = loss_val
                                else:
                                    # Use accuracy for regular plotting
                                    acc_val = float(row["overall_accuracy"]) if row["overall_accuracy"] not in ("", None) else np.nan
                            except Exception:
                                acc_val = np.nan
                            if budget is None:
                                continue
                            if method in args.plot_methods:
                                method_maps[method].setdefault(step, {})[budget] = acc_val

                    steps_sorted = sorted(set(steps_list))
                    actual_budgets = shared_budgets

                    # SAFETY CHECK: Ensure steps_sorted is not empty before proceeding
                    if not steps_sorted:
                        print(f"⚠️  WARNING: No valid checkpoint steps found in CSV for final plot!")
                        print(f"   CSV file: {out_csv}")
                        print(f"   Steps list: {steps_list}")
                        print(f"   Skipping final plot generation due to empty data")
                        continue

                    # ENHANCED SAFETY CHECKS: Prevent extremely large arrays that would cause plotting issues
                    if len(steps_sorted) > 1000 or len(actual_budgets) > 1000:
                        print(f"⚠️  WARNING: Extremely large data dimensions detected for final plot!")
                        print(f"   Steps: {len(steps_sorted)} (range: {min(steps_sorted)} to {max(steps_sorted)})")
                        print(f"   Budgets: {len(actual_budgets)} (range: {min(actual_budgets)} to {max(actual_budgets)})")
                        print(f"   This would create a {len(actual_budgets)}x{len(steps_sorted)} array = {len(actual_budgets) * len(steps_sorted)} elements")
                        print(f"   Skipping final plot generation to prevent memory/plotting issues")
                        continue

                    # Additional safety check for reasonable step values
                    if max(steps_sorted) > 10000:
                        print(f"⚠️  WARNING: Extremely large step numbers detected for final plot!")
                        print(f"   Max step: {max(steps_sorted)}")
                        print(f"   All steps: {steps_sorted[:10]}...")  # Show first 10
                        print(f"   Skipping final plot generation due to unreasonable step values")
                        continue

                    # NEW: Check for extremely large step values that would cause plotting issues
                    if max(steps_sorted) > 1000000:  # 1 million steps
                        print(f"⚠️  WARNING: Extremely large step values detected for final plot!")
                        print(f"   Max step: {max(steps_sorted)}")
                        print(f"   This would create a plot with width of {max(steps_sorted)} pixels")
                        print(f"   Skipping final plot generation due to unreasonable step values")
                        continue

                    # NEW: Check total array size to prevent memory issues
                    total_elements = len(actual_budgets) * len(steps_sorted)
                    if total_elements > 1000000:  # 1 million elements
                        print(f"⚠️  WARNING: Total array size too large for final plot!")
                        print(f"   Budgets: {len(actual_budgets)} x Steps: {len(steps_sorted)} = {total_elements} elements")
                        print(f"   Skipping final plot generation to prevent memory/plotting issues")
                        continue

                    # NEW: Filter steps to reasonable range if they're too spread out
                    if len(steps_sorted) > 1 and (max(steps_sorted) - min(steps_sorted)) > 10000:
                        print(f"⚠️  WARNING: Step range too large for final plot, filtering to reasonable subset")
                        print(f"   Original step range: {min(steps_sorted)} to {max(steps_sorted)}")
                        
                        # Take every nth step to reduce the range
                        step_range = max(steps_sorted) - min(steps_sorted)
                        if step_range > 100000:
                            step_skip = max(1, step_range // 1000)  # Aim for ~1000 steps max
                        else:
                            step_skip = max(1, step_range // 100)   # Aim for ~100 steps max
                        
                        filtered_steps = steps_sorted[::step_skip]
                        if len(filtered_steps) < 2:
                            filtered_steps = [steps_sorted[0], steps_sorted[-1]]  # At least 2 points
                        
                        print(f"   Filtered to {len(filtered_steps)} steps with skip={step_skip}")
                        print(f"   New step range: {min(filtered_steps)} to {max(filtered_steps)}")
                        steps_sorted = filtered_steps

                    # Create data arrays for selected methods
                    method_arrays = {}
                    for method in args.plot_methods:
                        method_arrays[method] = np.full((len(actual_budgets), len(steps_sorted)), np.nan)

                    for j, s in enumerate(steps_sorted):
                        for k, b in enumerate(actual_budgets):
                            for method in args.plot_methods:
                                method_arrays[method][k, j] = method_maps[method].get(s, {}).get(b, np.nan)

                    # NEW: Check if we have any valid data (non-NaN values) before plotting
                    has_valid_data = False
                    for method in args.plot_methods:
                        if method in method_arrays:
                            method_data = method_arrays[method]
                            if not np.all(np.isnan(method_data)):
                                has_valid_data = True
                                break
                    
                    if not has_valid_data:
                        print(f"⚠️  WARNING: No valid data found for final comparison plot!")
                        print(f"   All method arrays contain only NaN values")
                        print(f"   Skipping final plot generation")
                        continue

                    # ADDITIONAL SAFETY CHECK: Ensure step values are reasonable to prevent extremely large images
                    if max(steps_sorted) > 10000:
                        print(f"⚠️  WARNING: Step values too large for final plot!")
                        print(f"   Max step: {max(steps_sorted)}")
                        print(f"   This would create an extremely wide image. Skipping final plot generation.")
                        continue

                    print(f"📊 Creating final comparison plot...")
                    print(f"   Steps: {len(steps_sorted)} (range: {min(steps_sorted)} to {max(steps_sorted)})")
                    print(f"   Budgets: {len(actual_budgets)} (range: {min(actual_budgets)} to {max(actual_budgets)})")
                    print(f"   Methods: {args.plot_methods}")
                    print(f"   Data shapes:")
                    for method in args.plot_methods:
                        if method in method_arrays:
                            method_data = method_arrays[method]
                            valid_count = np.sum(~np.isnan(method_data))
                            total_count = method_data.size
                            print(f"     • {method}: {method_data.shape}, {valid_count}/{total_count} valid values")

                    # Create plot with selected methods
                    if len(args.plot_methods) == 2:
                        if args.loss:
                            # Loss difference plotting: show difference between methods (raw delta)
                            # For loss, lower is better, so we show method_B - method_A (positive = method_A better)
                            # This creates a n_checkpoints × n_budgets matrix showing raw loss differences (delta)
                            # where positive values = method_A (GA) better, negative = method_B (ES) better
                            loss_diff = method_arrays[args.plot_methods[1]] - method_arrays[args.plot_methods[0]]
                            
                            print(f"📊 Creating final loss difference heatmap...")
                            print(f"   Steps array shape: {np.array(steps_sorted).shape}")
                            print(f"   Budgets array shape: {np.array(actual_budgets).shape}")
                            print(f"   Loss diff array shape: {loss_diff.shape}")
                            print(f"   Method A: {args.plot_methods[0]}")
                            print(f"   Method B: {args.plot_methods[1]}")
                            
                            try:
                                # Use the new specialized loss difference visualization (raw delta)
                                from visualization import visualize_loss_difference_heatmap
                                fig = visualize_loss_difference_heatmap(
                                    steps=np.array(steps_sorted),
                                    budgets=np.array(actual_budgets),
                                    loss_diff=loss_diff,
                                    method_A_name=args.plot_methods[0].replace("_", " ").title(),  # First method (typically GA)
                                    method_B_name=args.plot_methods[1].replace("_", " ").title(),  # Second method (typically ES)
                                )
                                print(f"✅ Successfully created final loss difference heatmap")
                            except Exception as e:
                                print(f"❌ Failed to create final loss difference heatmap: {e}")
                                print(f"   Error type: {type(e).__name__}")
                                import traceback
                                print(f"   Traceback: {traceback.format_exc()}")
                                # Fall back to regular comparison plot
                                print(f"📊 Falling back to regular comparison plot...")
                                fig = visualize_optimization_comparison(
                                    steps=np.array(steps_sorted),
                                    budgets=np.array(actual_budgets),
                                    acc_A=method_arrays[args.plot_methods[0]],
                                    acc_B=method_arrays[args.plot_methods[1]],
                                    method_A_name=args.plot_methods[0].replace("_", " ").title(),
                                    method_B_name=args.plot_methods[1].replace("_", " ").title(),
                                )
                        else:
                            # Regular accuracy comparison
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

                    # SAFETY CHECK: Ensure steps_sorted is not empty before calling max()
                    if not steps_sorted:
                        print(f"⚠️  WARNING: steps_sorted is empty, cannot calculate progress percentage")
                        max_progress = 0
                        progress_percentage = 0
                    else:
                        max_progress = max(steps_sorted)
                        # Use the length of steps_sorted instead of checkpoints since checkpoints is not in scope here
                        denom_final = max(len(steps_sorted) - 1, 1)
                        progress_percentage = int((max_progress / denom_final) * 100)

                    if args.loss and len(args.plot_methods) == 2:
                        plot_type = "Loss Difference (Delta)"
                        plot_description = f"Raw loss difference (delta): {args.plot_methods[1].replace('_', ' ').title()} - {args.plot_methods[0].replace('_', ' ').title()}"
                    else:
                        plot_type = "Accuracy Comparison"
                        plot_description = "Higher values = better performance"

                    if args.loss and len(args.plot_methods) == 2:
                        # For loss difference plots, no title (clean heatmap)
                        pass
                    else:
                        # Regular title for accuracy comparisons
                        fig.suptitle(
                            f"Final {plot_type} - {args.run_name}\n"
                            f"{plot_description}\n"
                            f"Training Progress: {len(steps_sorted)} steps (0% → {progress_percentage}%), Budgets: {len(actual_budgets)}",
                            fontsize=14,
                            y=0.98,
                        )

                    plot_path = out_dir / f"optim_comparison_final_{args.run_name}.png"
                    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)

                    wandb.log(
                        {
                            "final/optimization_comparison": wandb.Image(str(plot_path)),
                            "final/total_checkpoints": len(steps_sorted),
                            "final/total_budgets": len(actual_budgets),
                            "final/checkpoint_steps": steps_sorted,
                            "final/budget_values": actual_budgets,
                            "final/training_progress_percentage": progress_percentage,
                        }
                    )

                    plot_art = wandb.Artifact(f"{args.run_name}--final-optim-comparison", type="evaluation")
                    plot_art.add_file(str(plot_path))
                    run.log_artifact(plot_art)

                    print(
                        f"📊 Generated and uploaded final comparison plot with {len(steps_sorted)} training progress steps (0% → {progress_percentage}%) and {len(actual_budgets)} budgets"
                    )
                    print(f"   📈 Methods: {', '.join(args.plot_methods).replace('_', ' ').title()}")

                    # Generate additional loss plots if --loss flag is enabled
                    if args.loss and len(args.plot_methods) == 2:
                        try:
                            # Generate Loss vs Budget plot
                            final_loss_budget_plot_path = generate_loss_vs_budget_plot(
                                method_arrays=method_arrays,
                                budgets=actual_budgets,
                                method_names=args.plot_methods,
                                checkpoint_name="final_summary",
                                checkpoint_step=max_progress,
                            )

                            # Generate Loss vs Training Progress plot
                            final_loss_training_plot_path = generate_loss_vs_training_plot(
                                method_arrays=method_arrays,
                                steps=steps_sorted,
                                method_names=args.plot_methods,
                                checkpoint_name="final_summary",
                                checkpoint_step=max_progress,
                                total_checkpoints=len(steps_sorted),  # Use steps_sorted length instead of checkpoints
                            )
                            
                            # Generate final budget-based plots (similar to store_latent_search.py)
                            if args.dataset_folder and args.dataset_length > 0:
                                try:
                                    # Find GA and ES NPZ files from the last checkpoint
                                    ga_npz_path = None
                                    es_npz_path = None
                                    
                                    # Find GA and ES NPZ files from the last checkpoint
                                    last_checkpoint = checkpoints[-1] if checkpoints else None
                                    if last_checkpoint:
                                        ga_trajectory_path = f"temp_trajectories/gradient_ascent_{last_checkpoint['name']}.npz"
                                        es_trajectory_path = f"temp_trajectories/evolutionary_search_{last_checkpoint['name']}.npz"
                                        
                                        if os.path.exists(ga_trajectory_path):
                                            ga_npz_path = ga_trajectory_path
                                        if os.path.exists(es_trajectory_path):
                                            es_npz_path = es_trajectory_path
                                    
                                    if ga_npz_path and es_npz_path:
                                        final_budget_plots = generate_budget_based_plots(
                                            ga_npz_path=ga_npz_path,
                                            es_npz_path=es_npz_path,
                                            out_dir=out_dir,
                                            dataset_length=args.dataset_length,
                                            checkpoint_name="final_summary",
                                            checkpoint_step=max_progress
                                        )
                                        print(f"📊 Generated final budget-based plots: {list(final_budget_plots.keys())}")
                                    else:
                                        print("⚠️  Missing GA or ES NPZ files for final budget-based plotting")
                                except Exception as e:
                                    print(f"⚠️  Failed to generate final budget-based plots: {e}")

                            # Upload both plots to W&B
                            if final_loss_budget_plot_path and final_loss_training_plot_path:
                                try:
                                    wandb.log(
                                        {
                                            "final/loss_vs_budget": wandb.Image(str(final_loss_budget_plot_path)),
                                            "final/loss_vs_training": wandb.Image(str(final_loss_training_plot_path)),
                                        }
                                    )

                                    # Also upload as artifacts
                                    loss_budget_art = wandb.Artifact(f"{args.run_name}--final-loss-vs-budget", type="evaluation")
                                    loss_budget_art.add_file(str(final_loss_budget_plot_path))
                                    run.log_artifact(loss_budget_art)

                                    loss_training_art = wandb.Artifact(f"{args.run_name}--final-loss-vs-training", type="evaluation")
                                    loss_training_art.add_file(str(final_loss_training_plot_path))
                                    run.log_artifact(loss_training_art)

                                    print("📊 Generated and uploaded final loss plots:")
                                    print(f"   • Loss vs Budget: {final_loss_budget_plot_path}")
                                    print(f"   • Loss vs Training: {final_loss_training_plot_path}")
                                except Exception as e:
                                    print(f"⚠️  Failed to upload final loss plots to W&B: {e}")
                                    print(f"   Error type: {type(e).__name__}")
                                    print(f"   Error details: {str(e)}")
                                    import traceback
                                    print(f"   Traceback: {traceback.format_exc()}")
                            else:
                                print("⚠️  Failed to generate one or both final loss plots")

                        except Exception as e:
                            print(f"⚠️  Failed to generate final loss plots: {e}")
                            print(f"   Error type: {type(e).__name__}")
                            print(f"   Error details: {str(e)}")
                            import traceback
                            print(f"   Traceback: {traceback.format_exc()}")

                except Exception as e:
                    print(f"⚠️  Failed to generate or upload final comparison plot: {e}")
                    print(f"   Error details: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
            else:
                print("📁 Final plot generation disabled (--no_files flag)")

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

            print(f"\n📊 Output configuration:")
            print(f"   • Output directory: {args.out_dir}")
            print(f"   • CSV saved to: {out_csv}")
            print(f"   • File generation: {'enabled' if not args.no_files else 'disabled (--no_files flag)'}")
            print(f"📅 Timestamp: {timestamp}")
            print("📈 Available metrics in CSV:")
            print("   - overall_accuracy")
            print("   - top_1_shape_accuracy")
            print("   - top_1_accuracy")
            print("   - top_1_pixel_correctness")
            print("   - top_2_shape_accuracy")
            print("   - top_2_accuracy")
            print("   - top_2_pixel_correctness")
            print("   - total_final_loss")
            if args.n_samples > 1:
                print("   - sample_number")
                print("   - sample_seed")
            if args.es_use_subspace_mutation:
                print("   - subspace_enabled")
                print("   - subspace_dim")
                print("   - ga_step_length")
                print("   - trust_region_radius")

            print(f"\n🔬 Methods evaluated:")
            for method in ["gradient_ascent", "random_search", "evolutionary_search"]:
                if method in args.plot_methods:
                    print(
                        f"   - {method} ({'num_steps' if method == 'gradient_ascent' else 'num_samples' if method == 'random_search' else 'num_generations'})"
                    )
                else:
                    print(f"   - {method} (skipped - not in plot_methods)")

            # Sample information
            if args.n_samples > 1:
                print(f"\n🧪 Sample configuration:")
                print(f"   • Multiple samples: {args.n_samples} runs per evaluation")
                print(f"   • Base seed: {args.dataset_seed}")
                print(f"   • Sample seeds: {[args.dataset_seed + i for i in range(args.n_samples)]}")
                if args.aggregate_statistics:
                    print("   • Statistical aggregation: enabled")

            # Comprehensive logging summary
            print(f"\n{'=' * 80}")
            print("📋 COMPREHENSIVE EVALUATION LOG")
            print(f"{'=' * 80}")
            print(f"🕐 Run completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Run name: {args.run_name}")
            print(f"🎯 Evaluation source: {'JSON' if using_json else 'Dataset'}")
            print(f"📊 Plotting mode: {'Loss Difference' if args.loss else 'Accuracy Comparison'}")
            if args.loss:
                print(
                    f"   • Loss difference: {args.plot_methods[1].replace('_', ' ').title()} - {args.plot_methods[0].replace('_', ' ').title()}"
                )
                print(f"   • Positive values = {args.plot_methods[0].replace('_', ' ').title()} better (lower loss)")
                print("   • Additional plots: Loss vs Budget, Loss vs Training Progress")
            if using_json:
                print(f"   • Challenges: {args.json_challenges}")
                print(f"   • Solutions: {args.json_solutions}")
                print(f"   • Tasks limited to: {args.only_n_tasks}")
            if args.dataset_folder:
                print(f"   • Dataset: {args.dataset_folder}")
                print(f"   • Length: {args.dataset_length}")
                print(f"   • Batch size: {args.dataset_batch_size}")

            # Advanced features
            if args.n_samples > 1:
                print(f"   • Multiple samples: {args.n_samples} runs with different seeds")
                if args.aggregate_statistics:
                    print("   • Statistical aggregation: enabled")

            # Output configuration
            print(f"   • Output directory: {args.out_dir}")
            print(f"   • File generation: {'enabled' if not args.no_files else 'disabled'}")
    
    print(f"\n⚙️  Method configurations:")
    for method in args.plot_methods:
        if method == "gradient_ascent":
            ga_config = base_methods[method]
            print(f"   • {method}: lr={ga_config.get('lr')}, optimizer={ga_config.get('optimizer')}")
            if ga_config.get('lr_schedule'):
                print(f"     - LR Schedule: enabled (exponent={ga_config.get('lr_schedule_exponent')})")
            if ga_config.get('accumulate_gradients_decoder_pairs'):
                print(f"     - Accumulate gradients decoder pairs: enabled")
            if ga_config.get('scan_gradients_latents'):
                print(f"     - Scan gradients latents: enabled")
            if ga_config.get('include_mean_latent'):
                print(f"     - Include mean latent: enabled")
            if ga_config.get('include_all_latents'):
                print(f"     - Include all latents: enabled")
            if ga_config.get('random_perturbation'):
                print(f"     - Random perturbation: {ga_config.get('random_perturbation')}")
            if ga_config.get('track_progress'):
                print(f"     - Progress tracking: enabled")
        elif method == "random_search":
            rs_config = base_methods[method]
            print(f"   • {method}: scale={rs_config.get('scale')}, scan_batch_size={rs_config.get('scan_batch_size')}")
            if rs_config.get('include_mean_latent'):
                print(f"     - Include mean latent: enabled")
            if rs_config.get('include_all_latents'):
                print(f"     - Include all latents: enabled")
            if rs_config.get('random_perturbation'):
                print(f"     - Random perturbation: {rs_config.get('random_perturbation')}")
            if rs_config.get('track_progress'):
                print(f"     - Progress tracking: enabled")
        elif method == "evolutionary_search":
            es_config = base_methods[method]
            print(f"   • {method}: mutation_std={es_config.get('mutation_std')}")
            if es_config.get('mutation_decay') is not None:
                print(f"     - Mutation decay: {es_config.get('mutation_decay')}")
            if es_config.get('elite_size') is not None:
                print(f"     - Elite size: {es_config.get('elite_size')}")
            if es_config.get('track_progress'):
                print(f"     - Progress tracking: enabled")
            if args.es_use_subspace_mutation:
                print(f"     - Subspace mutation: enabled (dim={args.es_subspace_dim}, ga_step={args.es_ga_step_length})")
                if args.es_trust_region_radius is not None:
                    print(f"     - Trust region radius: {args.es_trust_region_radius}")
            else:
                print(f"     - Subspace mutation: disabled (standard isotropic mutation)")
    
    # Advanced features
    if args.n_samples > 1:
        print(f"\n🧪 Advanced features:")
        print(f"   • Multiple samples: {args.n_samples} runs with different seeds")
        if args.aggregate_statistics:
            print(f"   • Aggregate statistics: enabled")
    
    # Background visualization
    if args.background_resolution != 400 or args.background_smoothing or args.background_knn != 5 or args.background_bandwidth_scale != 1.25 or args.background_global_mix != 0.05:
        print(f"\n🎨 Background visualization:")
        print(f"   • Resolution: {args.background_resolution}")
        if args.background_smoothing:
            print(f"   • Smoothing: enabled")
        print(f"   • k-NN: {args.background_knn}")
        print(f"   • Bandwidth scale: {args.background_bandwidth_scale}")
        print(f"   • Global mix: {args.background_global_mix}")
    
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

    # Close CSV file
    try:
        f_csv.close()
        print(f"📁 CSV file closed: {out_csv}")
    except Exception as e:
        print(f"⚠️  Failed to close CSV file: {e}")

    try:
        run.finish()
    except Exception:
        pass
    
    # Clean up trajectory files
    try:
        temp_dir = Path("temp_trajectories")
        if temp_dir.exists():
            for trajectory_file in temp_dir.glob("*.npz"):
                trajectory_file.unlink()
            temp_dir.rmdir()
            print(f"🧹 Cleaned up trajectory storage directory")
    except Exception as e:
        print(f"⚠️  Failed to clean up trajectory storage: {e}")


if __name__ == "__main__":
    main()
