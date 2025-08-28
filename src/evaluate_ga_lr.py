#!/usr/bin/env python3
"""
Evaluate gradient ascent with different learning rates and budgets using store_latent_search.py.
Creates a heatmap showing lr vs budget colored by total loss after optimization.

USAGE EXAMPLES:
==============

1. BASIC USAGE (sweep lr from 1e-3 to 1.0, budget from 50 to 200):
   python3 src/evaluate_ga_lr.py \
     --run_name "winter-fire-132" \
     --dataset_folder "pattern2d_eval" \
     --lr_start 1e-3 \
     --lr_end 1.0 \
     --lr_steps 8 \
     --budget_start 50 \
     --budget_end 200 \
     --budget_steps 6

2. QUICK TEST (fewer values for fast testing):
   python3 src/evaluate_ga_lr.py \
     --run_name "winter-fire-132" \
     --dataset_folder "pattern2d_eval" \
     --lr_start 1e-2 \
     --lr_end 0.5 \
     --lr_steps 4 \
     --budget_start 50 \
     --budget_end 100 \
     --budget_steps 3
"""

import os
import sys
import argparse
import subprocess
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import wandb  # Optional: used for logging results
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def run_store_latent_search(
    artifact_path: str,
    lr: float,
    budget: int,
    dataset_folder: str,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: str = "true",
    dataset_seed: int = 0,
    out_dir: str = "results/ga_lr_budget_sweep"
) -> Tuple[bool, Optional[float], float]:
    """Run store_latent_search.py for a specific lr and budget."""
    
    # Create unique output directory for this run
    run_out_dir = os.path.join(out_dir, f"lr_{lr:.6f}_budget_{budget}")
    os.makedirs(run_out_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "src/store_latent_search.py",
        "--wandb_artifact_path", artifact_path,
        "--budget", str(budget),
        "--ga_lr", str(lr),
        # Mirror the working invocation: explicit GA/ES decomposition and disable files/plots
        "--ga_steps", str(max(1, budget // 2)),
        "--es_population", "1",
        "--es_generations", str(budget),
        "--no_files",
        "--background_resolution", "400",
        "--dataset_folder", dataset_folder,
        "--out_dir", run_out_dir,
        "--wandb_project", "none",  # Use "none" to disable W&B
    ]
    
    if dataset_length:
        cmd.extend(["--dataset_length", str(dataset_length)])
    if dataset_batch_size:
        cmd.extend(["--dataset_batch_size", str(dataset_batch_size)])
    
    cmd.extend([
        "--dataset_use_hf", dataset_use_hf,
        "--dataset_seed", str(dataset_seed),
    ])
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ store_latent_search completed successfully in {execution_time:.2f}s")
            
            # Extract total loss from GA NPZ file
            ga_npz_path = os.path.join(run_out_dir, "ga_latents.npz")
            total_loss = extract_total_loss(ga_npz_path)
            
            return True, total_loss, execution_time
        else:
            print(f"❌ store_latent_search failed with return code {result.returncode}")
            
            # Try to extract loss from stderr even when evaluation fails
            total_loss = None
            if result.stderr.strip():
                print(f"Error output:\n{result.stderr}")
                total_loss = extract_loss_from_stderr(result.stderr)
            
            # Also try to extract from NPZ file if it exists (partial success)
            if total_loss is None:
                ga_npz_path = os.path.join(run_out_dir, "ga_latents.npz")
                total_loss = extract_total_loss(ga_npz_path)
            
            # Return False for success but include any loss we found
            return False, total_loss, execution_time
            
    except Exception as e:
        print(f"❌ Error running store_latent_search: {e}")
        return False, None, 0.0


def run_store_latent_search_single(
    artifact_path: str,
    lr: float,
    max_budget: int,
    dataset_folder: str,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: str = "true",
    dataset_seed: int = 0,
    out_dir: str = "results/ga_lr_budget_sweep",
    no_files: bool = False
) -> Tuple[bool, Dict[int, float], float]:
    """Run store_latent_search.py once with max_budget and extract intermediate results."""
    
    # Create unique output directory for this run
    run_out_dir = os.path.join(out_dir, f"lr_{lr:.6f}_budget_{max_budget}")
    os.makedirs(run_out_dir, exist_ok=True)
    
    # Build command with maximum budget
    cmd = [
        sys.executable, "src/store_latent_search.py",
        "--wandb_artifact_path", artifact_path,
        "--budget", str(max_budget),
        "--ga_lr", str(lr),
        # Ensure same GA/ES decomposition as the working command
        "--ga_steps", str(max(1, max_budget // 2)),
        "--es_population", "1",
        "--es_generations", str(max_budget),
        "--dataset_folder", dataset_folder,
        "--out_dir", run_out_dir,
        "--wandb_project", "none",  # Use "none" to disable W&B
    ]
    
    # Add no_files flag if requested
    if no_files:
        cmd.extend(["--no_files"])
    
    if dataset_length:
        cmd.extend(["--dataset_length", str(dataset_length)])
    if dataset_batch_size:
        cmd.extend(["--dataset_batch_size", str(dataset_batch_size)])
    
    cmd.extend([
        "--dataset_use_hf", dataset_use_hf,
        "--dataset_seed", str(dataset_seed),
    ])
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ store_latent_search completed successfully in {execution_time:.2f}s")
            
            # Extract intermediate losses at different budget checkpoints
            ga_npz_path = os.path.join(run_out_dir, "ga_latents.npz")
            intermediate_losses = extract_intermediate_losses(ga_npz_path, max_budget)
            
            return True, intermediate_losses, execution_time
        else:
            print(f"❌ store_latent_search failed with return code {result.returncode}")
            
            # Try to extract any available loss data
            intermediate_losses = {}
            
            # If no_files is enabled, try to extract from output directly
            if no_files:
                print(f"📊 no_files enabled - extracting loss from evaluation output...")
                intermediate_losses = extract_loss_from_evaluation_output(result.stdout, result.stderr)
            else:
                # First try stdout (might have loss information)
                if result.stdout.strip():
                    print(f"📊 Checking stdout for loss data...")
                    stdout_loss = extract_loss_from_stdout(result.stdout)
                    if stdout_loss is not None:
                        intermediate_losses[max_budget] = stdout_loss
                
                # Then try stderr
                if result.stderr.strip():
                    print(f"Error output:\n{result.stderr}")
                    # Try to extract final loss from stderr
                    final_loss = extract_loss_from_stderr(result.stderr)
                    if final_loss is not None:
                        intermediate_losses[max_budget] = final_loss
                
                # Also try to extract from NPZ file if it exists (partial success)
                if not intermediate_losses:
                    ga_npz_path = os.path.join(run_out_dir, "ga_latents.npz")
                    intermediate_losses = extract_intermediate_losses(ga_npz_path, max_budget)
                    
                    # If still no data, search for any available loss files
                    if not intermediate_losses:
                        print(f"🔍 No NPZ data found, searching for any available loss files...")
                        intermediate_losses = search_for_loss_files(run_out_dir)
            
            # Return False for success but include any loss data we found
            return False, intermediate_losses, execution_time
            
    except Exception as e:
        print(f"❌ Error running store_latent_search: {e}")
        return False, {}, 0.0


def run_ga_only_single(
    artifact_path: str,
    lr: float,
    max_budget: int,
    dataset_folder: str,
    dataset_length: Optional[int] = None,
    dataset_batch_size: Optional[int] = None,
    dataset_use_hf: str = "true",
    dataset_seed: int = 0,
    out_dir: str = "results/ga_lr_budget_sweep",
) -> Tuple[bool, Dict[int, float], float]:
    """Run evaluate_checkpoint.py in GA mode only and extract intermediate losses."""
    run_out_dir = os.path.join(out_dir, f"ga_only_lr_{lr:.6f}_budget_{max_budget}")
    os.makedirs(run_out_dir, exist_ok=True)

    ga_steps = max(1, max_budget // 2)
    ga_npz_path = os.path.join(run_out_dir, "ga_latents.npz")

    cmd = [
        sys.executable, "src/evaluate_checkpoint.py",
        "-w", artifact_path,
        "-i", "gradient_ascent",
        "--num-steps", str(ga_steps),
        "--lr", str(lr),
        "--no-wandb-run", "true",
        "--store-latents", ga_npz_path,
        "-d", dataset_folder,
    ]
    if dataset_length is not None:
        cmd += ["--dataset-length", str(dataset_length)]
    if dataset_batch_size is not None:
        cmd += ["--dataset-batch-size", str(dataset_batch_size)]
    cmd += ["--dataset-use-hf", dataset_use_hf, "--dataset-seed", str(dataset_seed)]

    print(f"\n[GA-ONLY] Running: {' '.join(cmd)}")
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        execution_time = time.time() - start_time
        if result.returncode != 0:
            print(f"[GA-ONLY] ❌ evaluate_checkpoint failed (rc={result.returncode})")
            if result.stderr.strip():
                print(f"[GA-ONLY] stderr:\n{result.stderr}")
            if result.stdout.strip():
                print(f"[GA-ONLY] stdout:\n{result.stdout}")

        # Extract intermediate losses directly from the GA NPZ file
        intermediate_losses = extract_intermediate_losses(ga_npz_path, max_budget)
        return result.returncode == 0 and bool(intermediate_losses), intermediate_losses, execution_time
    except Exception as e:
        print(f"[GA-ONLY] ❌ Error: {e}")
        return False, {}, 0.0

def extract_total_loss(npz_path: str) -> Optional[float]:
    """Extract total final loss from GA NPZ file."""
    try:
        if not os.path.exists(npz_path):
            print(f"⚠️  NPZ file not found: {npz_path}")
            return None
            
        data = np.load(npz_path, allow_pickle=True)
        
        # Try different possible loss keys
        loss_keys = ['ga_losses', 'total_final_loss', 'final_loss']
        for key in loss_keys:
            if key in data:
                losses = np.array(data[key])
                if losses.size > 0:
                    # Take the final loss value
                    final_loss = float(losses.reshape(-1)[-1])
                    print(f"📊 Extracted {key}: {final_loss:.6f}")
                    return final_loss
        
        print(f"⚠️  No loss data found in NPZ. Available keys: {list(data.keys())}")
        return None
        
    except Exception as e:
        print(f"⚠️  Failed to extract loss from {npz_path}: {e}")
        return None


def extract_intermediate_losses(npz_path: str, max_budget: int) -> Dict[int, float]:
    """Extract intermediate losses at different budget checkpoints from GA NPZ file."""
    try:
        if not os.path.exists(npz_path):
            print(f"⚠️  NPZ file not found: {npz_path}")
            return {}
            
        data = np.load(npz_path, allow_pickle=True)
        
        # Prefer per-sample trajectories when available for correct per-budget averaging
        if 'ga_losses_per_sample' in data and 'ga_budget' in data:
            per_sample = np.array(data['ga_losses_per_sample'])  # (N, S)
            ga_budget = np.array(data['ga_budget']).reshape(-1)  # (S,)
            if per_sample.size > 0 and ga_budget.size > 0:
                print(f"📊 Using per-sample losses: {per_sample.shape}, ga_budget: {ga_budget.shape}")
                # Map each available budget in ga_budget to the mean loss across samples
                intermediate_losses = {}
                S = min(per_sample.shape[1], ga_budget.shape[0])
                for s in range(S):
                    b = int(ga_budget[s])
                    if b <= max_budget:
                        mean_loss = float(np.mean(per_sample[:, s]))
                        intermediate_losses[b] = mean_loss
                        print(f"📊 Budget {b} → Mean loss over {per_sample.shape[0]} samples: {mean_loss:.6f}")
                if intermediate_losses:
                    return intermediate_losses
        
        # Fallback to representative single trajectory if per-sample not present
        losses = None
        for key in ['ga_losses', 'total_final_loss', 'final_loss']:
            if key in data:
                losses = np.array(data[key])
                if losses.size > 0:
                    print(f"📊 Found {key} with {losses.size} steps")
                    break
        if losses is None:
            print(f"⚠️  No loss data found in NPZ. Available keys: {list(data.keys())}")
            return {}
        losses = losses.reshape(-1)
        checkpoints = list(2 * np.arange(1, len(losses) + 1))
        checkpoints = [int(b) for b in checkpoints if b <= max_budget]
        intermediate_losses = {}
        for budget in checkpoints:
            step_idx = min(max(0, budget // 2 - 1), len(losses) - 1)
            intermediate_losses[budget] = float(losses[step_idx])
            print(f"📊 Budget {budget} → Step {step_idx} → Loss {intermediate_losses[budget]:.6f}")
        
        return intermediate_losses
        
    except Exception as e:
        print(f"⚠️  Failed to extract intermediate losses from {npz_path}: {e}")
        return {}


def search_for_loss_files(run_out_dir: str) -> Dict[int, float]:
    """Search for any available loss data in the run directory."""
    intermediate_losses = {}
    
    # Look for any files that might contain loss information
    possible_files = [
        "ga_latents.npz",
        "es_latents.npz", 
        "*.log",
        "*.txt",
        "*.json"
    ]
    
    print(f"🔍 Searching for loss data in: {run_out_dir}")
    
    # Check if directory exists
    if not os.path.exists(run_out_dir):
        print(f"⚠️  Run directory not found: {run_out_dir}")
        return {}
    
    # List all files in the directory
    try:
        files = os.listdir(run_out_dir)
        print(f"📁 Files found: {files}")
        
        # Try to extract from any available NPZ files
        for file in files:
            if file.endswith('.npz'):
                npz_path = os.path.join(run_out_dir, file)
                print(f"🔍 Checking NPZ file: {file}")
                
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    print(f"📊 NPZ keys: {list(data.keys())}")
                    
                    # Look for any loss-related data
                    for key in data.keys():
                        if 'loss' in key.lower() or 'score' in key.lower():
                            arr = np.array(data[key])
                            if arr.size > 0:
                                # Take the first available loss value
                                loss_val = float(arr.reshape(-1)[0])
                                print(f"📊 Found loss data in {key}: {loss_val:.6f}")
                                
                                # Assign to a reasonable budget (use file size as proxy for budget)
                                if 'ga' in file.lower():
                                    # For GA, assume this is the final loss
                                    intermediate_losses[100] = loss_val
                                else:
                                    # For other files, assign to max budget
                                    intermediate_losses[100] = loss_val
                                break
                except Exception as e:
                    print(f"⚠️  Failed to read NPZ file {file}: {e}")
                    
    except Exception as e:
        print(f"⚠️  Failed to list directory {run_out_dir}: {e}")
    
    return intermediate_losses


def extract_loss_from_evaluation_output(stdout: str, stderr: str) -> Dict[int, float]:
    """Extract loss values from evaluation output when no_files is used."""
    intermediate_losses = {}
    
    # Try to extract from stdout first
    if stdout.strip():
        stdout_loss = extract_loss_from_stdout(stdout)
        if stdout_loss is not None:
            # Assume this is the final loss at max budget
            intermediate_losses[100] = stdout_loss
            print(f"📊 Extracted final loss from stdout: {stdout_loss:.6f}")
    
    # Try to extract from stderr as fallback
    if stderr.strip():
        stderr_loss = extract_loss_from_stderr(stderr)
        if stderr_loss is not None:
            # Use stderr loss if no stdout loss found
            if not intermediate_losses:
                intermediate_losses[100] = stderr_loss
                print(f"📊 Extracted final loss from stderr: {stderr_loss:.6f}")
    
    return intermediate_losses


def extract_loss_from_stdout(stdout: str) -> Optional[float]:
    """Try to extract loss information from stdout output."""
    try:
        # Look for any loss-like patterns in the stdout
        import re
        
        # Common loss patterns in stdout
        loss_patterns = [
            r'loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'Loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'final[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'step[:\s]*\d+[:\s]*loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',  # Step-based loss
            r'generation[:\s]*\d+[:\s]*loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',  # Generation-based loss
        ]
        
        for pattern in loss_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                # Convert to float and return the last match (most recent)
                try:
                    loss_value = float(matches[-1])
                    print(f"📊 Extracted loss from stdout: {loss_value:.6f}")
                    return loss_value
                except ValueError:
                    continue
        
        # Look for any numeric values that might be losses
        numeric_patterns = [
            r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)',  # Scientific notation
            r'([+-]?\d+\.\d+)',  # Decimal numbers
            r'([+-]?\d+)',  # Integer numbers
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, stdout)
            if matches:
                # Filter out very large or very small numbers that are unlikely to be losses
                for match in matches:
                    try:
                        val = float(match)
                        # Reasonable loss range: between -1000 and 1000
                        if -1000 < val < 1000:
                            print(f"📊 Extracted potential loss from stdout: {val:.6f}")
                            return val
                    except ValueError:
                        continue
        
        return None
        
    except Exception as e:
        print(f"⚠️  Failed to extract loss from stdout: {e}")
        return None


def extract_loss_from_stderr(stderr: str) -> Optional[float]:
    """Try to extract loss information from stderr output when evaluation fails."""
    try:
        # Look for any loss-like patterns in the error output
        import re
        
        # Common loss patterns in error messages
        loss_patterns = [
            r'loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'Loss[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'final[:\s]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        ]
        
        for pattern in loss_patterns:
            matches = re.findall(pattern, stderr, re.IGNORECASE)
            if matches:
                # Convert to float and return the first match
                try:
                    loss_value = float(matches[0])
                    print(f"📊 Extracted loss from stderr: {loss_value:.6f}")
                    return loss_value
                except ValueError:
                    continue
        
        # Look for any numeric values that might be losses
        # This is more aggressive and might catch losses in different formats
        numeric_patterns = [
            r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)',  # Scientific notation
            r'([+-]?\d+\.\d+)',  # Decimal numbers
            r'([+-]?\d+)',  # Integer numbers
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, stderr)
            if matches:
                # Filter out very large or very small numbers that are unlikely to be losses
                for match in matches:
                    try:
                        val = float(match)
                        # Reasonable loss range: between -1000 and 1000
                        if -1000 < val < 1000:
                            print(f"📊 Extracted potential loss from stderr: {val:.6f}")
                            return val
                    except ValueError:
                        continue
        
        return None
        
    except Exception as e:
        print(f"⚠️  Failed to extract loss from stderr: {e}")
        return None


def create_heatmap(
    lrs: List[float], 
    budgets: List[int], 
    results_matrix: np.ndarray,
    out_dir: str
) -> str:
    """Create a heatmap showing lr vs budget colored by total loss."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create custom colormap to match store_latent_search
    from matplotlib.colors import LinearSegmentedColormap
    custom_colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', custom_colors, N=256)

    # Create the heatmap
    im = ax.imshow(results_matrix, cmap=custom_cmap, aspect='auto', 
                   extent=[budgets[0], budgets[-1], lrs[0], lrs[-1]],
                   origin='lower')
    
    # No colorbar and no cell labels per request
    
    # Set labels and title (no cell labels)
    ax.set_xlabel('Search Budget', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('GA Learning Rate vs Budget: Total Loss Heatmap', fontsize=16, fontweight='bold')
    
    # Keep linear axes
    # Ensure ticks reflect the actual lr and budget values we used
    ax.set_xticks(list(budgets))
    ax.set_yticks(list(lrs))
    # Format y ticks (learning rate) for readability
    ax.set_yticklabels([f"{v:.3g}" for v in lrs])

    # Add grid
    ax.grid(True, alpha=0.3)
    
    # No per-cell text annotations
    
    plt.tight_layout()
    
    # Save plot
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "ga_lr_budget_heatmap.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return str(fig_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GA with different learning rates and budgets")
    parser.add_argument("--artifact_path", type=str, required=True, help="W&B artifact path (e.g., 'entity/project/artifact_name')")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Dataset folder under 'src/datasets'")
    
    # LR sweep configuration
    parser.add_argument("--lr_start", type=float, default=1e-3, help="Starting learning rate (default: 1e-3)")
    parser.add_argument("--lr_end", type=float, default=1.0, help="Ending learning rate (default: 1.0)")
    parser.add_argument("--lr_steps", type=int, default=8, help="Number of learning rate values (default: 8)")
    
    # Budget sweep configuration
    parser.add_argument("--budget_start", type=int, default=50, help="Starting budget (default: 50)")
    parser.add_argument("--budget_end", type=int, default=200, help="Ending budget (default: 200)")
    parser.add_argument("--budget_steps", type=int, default=6, help="Number of budget values (default: 6)")
    
    # Dataset parameters
    parser.add_argument("--dataset_length", type=int, default=None, help="Max examples to eval")
    parser.add_argument("--dataset_batch_size", type=int, default=None, help="Batch size for dataset eval")
    parser.add_argument("--dataset_use_hf", type=str, default="true", help="Use HF hub (true/false)")
    parser.add_argument("--dataset_seed", type=int, default=0, help="Seed for dataset subsampling")
    
    # W&B parameters
    parser.add_argument("--project", type=str, default="LPN-ARC", help="W&B project name")
    parser.add_argument("--entity", type=str, default="ga624-imperial-college-london", help="W&B entity")
    parser.add_argument("--checkpoint_strategy", type=str, default="last", choices=["even", "first", "last"])
    
    # Performance options
    parser.add_argument("--no_files", action="store_true", help="Disable file generation and plotting (faster, just return values)")

    # W&B logging
    parser.add_argument("--wandb_project", type=str, default="GA_LR_SWEEP", help="Weights & Biases project to log results")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (team/user)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging even if available")
    
    args = parser.parse_args()
    
    # Generate LR and budget values
    lrs = np.logspace(np.log10(args.lr_start), np.log10(args.lr_end), args.lr_steps)
    budgets = np.logspace(np.log10(args.budget_start), np.log10(args.budget_end), args.budget_steps).astype(int)
    
    print(f"🔬 Learning Rate Sweep Configuration:")
    print(f"   - Start: {args.lr_start}")
    print(f"   - End: {args.lr_end}")
    print(f"   - Steps: {args.lr_steps}")
    print(f"   - Values: {lrs}")
    print(f"🔬 Budget Sweep Configuration:")
    print(f"   - Start: {args.budget_start}")
    print(f"   - End: {args.budget_end}")
    print(f"   - Steps: {args.budget_steps}")
    print(f"   - Values: {budgets}")
    
    # Use provided artifact path
    artifact_path = args.artifact_path
    print(f"🔍 Using artifact path: {artifact_path}")
    
    # Create output directory
    out_dir = Path("results/ga_lr_budget_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Results matrix: [lr_idx, budget_idx] -> total_loss (budget-aligned)
    results_matrix = np.full((len(lrs), len(budgets)), np.nan)
    execution_times = np.full((len(lrs), len(budgets)), np.nan)
    
    # CSV logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Extract a name from the artifact path for the CSV filename
    artifact_name = args.artifact_path.split('/')[-1]
    csv_path = out_dir / f"ga_lr_budget_sweep_{artifact_name}_{timestamp}.csv"
    
    # Initialize W&B run
    run = None
    if not args.no_wandb and _WANDB_AVAILABLE:
        run_name = f"ga_lr_sweep_{artifact_name}_{timestamp}"
        try:
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config={
                "artifact_path": args.artifact_path,
                "dataset_folder": args.dataset_folder,
                "dataset_length": args.dataset_length,
                "dataset_batch_size": args.dataset_batch_size,
                "dataset_use_hf": args.dataset_use_hf,
                "dataset_seed": args.dataset_seed,
                "lr_start": args.lr_start,
                "lr_end": args.lr_end,
                "lr_steps": args.lr_steps,
                "budget_start": args.budget_start,
                "budget_end": args.budget_end,
                "budget_steps": args.budget_steps,
            })
        except Exception as _we:
            print(f"⚠️  Failed to initialize W&B: {_we}")
            run = None

    # Run evaluations
    successful_evals = 0
    failed_evals = 0
    
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "timestamp", "artifact_name", "lr", "budget", "total_loss", "execution_time", "success", "status"
        ])
        
        for i, lr in enumerate(lrs):
            print(f"\n🔬 Testing lr = {lr:.6f} ({i+1}/{len(lrs)})")
            
            # Run GA-only with maximum budget to get intermediate results (avoid ES path)
            success, intermediate_losses, exec_time = run_ga_only_single(
                artifact_path=artifact_path,
                lr=lr,
                max_budget=args.budget_end,
                dataset_folder=args.dataset_folder,
                dataset_length=args.dataset_length,
                dataset_batch_size=args.dataset_batch_size,
                dataset_use_hf=args.dataset_use_hf,
                dataset_seed=args.dataset_seed,
                out_dir=str(out_dir),
            )
            
            # Store results for each budget checkpoint
            if intermediate_losses:
                for budget, loss in intermediate_losses.items():
                    # Find the closest budget index in our budget array
                    budget_idx = np.argmin(np.abs(np.array(budgets) - budget))
                    results_matrix[i, budget_idx] = loss
                    execution_times[i, budget_idx] = exec_time
                
                if success:
                    successful_evals += 1
                    print(f"✅ Success: lr={lr:.6f}, extracted {len(intermediate_losses)} budget points, time={exec_time:.2f}s")
                else:
                    print(f"⚠️  Partial success: lr={lr:.6f}, extracted {len(intermediate_losses)} budget points, time={exec_time:.2f}s (evaluation failed but losses extracted)")
            else:
                failed_evals += 1
                print(f"❌ Failed: lr={lr:.6f} (no loss data available)")
            
            # Write to CSV for each budget point
            for budget, loss in intermediate_losses.items():
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    artifact_name,
                    lr,
                    budget,
                    loss,
                    exec_time,
                    success,
                    "partial" if not success else "complete"
                ])

                # Log per-point result to W&B
                if run is not None:
                    try:
                        wandb.log({
                            "lr": float(lr),
                            "budget": int(budget),
                            "loss": float(loss),
                            "success": bool(success),
                        })
                    except Exception as _wl:
                        print(f"⚠️  Failed to log to W&B: {_wl}")
    
    # Forward-fill missing cells per LR to avoid empty spots (e.g., budgets below first available)
    try:
        for i in range(results_matrix.shape[0]):
            row = results_matrix[i]
            mask = ~np.isnan(row)
            if not np.any(mask):
                continue
            first_idx = int(np.where(mask)[0][0])
            # Backfill leading NaNs with first available value
            if first_idx > 0:
                row[:first_idx] = row[first_idx]
            # Forward-fill subsequent NaNs
            for j in range(first_idx + 1, row.shape[0]):
                if np.isnan(row[j]):
                    row[j] = row[j - 1]
            results_matrix[i] = row
    except Exception as _ff:
        print(f"⚠️  Forward-fill failed: {_ff}")

    # Create heatmap
    if successful_evals > 0:
        print(f"\n📊 Creating heatmap with {successful_evals} successful evaluations...")
        heatmap_path = create_heatmap(lrs, budgets, results_matrix, str(out_dir))
        print(f"📊 Heatmap saved to: {heatmap_path}")
        # Log heatmap to W&B
        if run is not None:
            try:
                wandb.log({"heatmap": wandb.Image(heatmap_path)})
            except Exception as _wl2:
                print(f"⚠️  Failed to log heatmap to W&B: {_wl2}")
    else:
        print(f"\n❌ No successful evaluations to create heatmap")
        heatmap_path = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 GA LR vs BUDGET SWEEP SUMMARY")
    print("=" * 60)
    print(f"Artifact: {args.artifact_path}")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Failed evaluations: {failed_evals}")
    print(f"Total evaluations: {len(lrs)} (one per learning rate with intermediate budget extraction)")
    print(f"LR range: {args.lr_start} to {args.lr_end}")
    print(f"Budget range: {args.budget_start} to {args.budget_end}")
    print(f"\n📊 CSV saved to: {csv_path}")
    if heatmap_path:
        print(f"📊 Heatmap saved to: {heatmap_path}")
    print(f"📅 Timestamp: {timestamp}")
    
    # Show best configuration
    if successful_evals > 0:
        best_idx = np.nanargmin(results_matrix)
        best_lr_idx, best_budget_idx = np.unravel_index(best_idx, results_matrix.shape)
        best_lr = lrs[best_lr_idx]
        best_budget = budgets[best_budget_idx]
        best_loss = results_matrix[best_lr_idx, best_budget_idx]
        print(f"\n🏆 Best configuration:")
        print(f"   Learning Rate: {best_lr:.6f}")
        print(f"   Budget: {best_budget}")
        print(f"   Total Loss: {best_loss:.6f}")
        if run is not None:
            try:
                wandb.log({
                    "best/lr": float(best_lr),
                    "best/budget": int(best_budget),
                    "best/loss": float(best_loss),
                })
            except Exception as _wl3:
                print(f"⚠️  Failed to log best config to W&B: {_wl3}")

    # Upload CSV as an artifact
    if run is not None:
        try:
            art = wandb.Artifact(name=f"ga_lr_budget_sweep_{artifact_name}_{timestamp}", type="ga_lr_sweep")
            art.add_file(str(csv_path))
            run.log_artifact(art)
        except Exception as _wa:
            print(f"⚠️  Failed to upload CSV artifact to W&B: {_wa}")

    # Finish the W&B run
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()


