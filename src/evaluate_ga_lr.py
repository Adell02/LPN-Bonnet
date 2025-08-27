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
            if result.stderr.strip():
                print(f"Error output:\n{result.stderr}")
            return False, None, execution_time
            
    except Exception as e:
        print(f"❌ Error running store_latent_search: {e}")
        return False, None, 0.0


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


def create_heatmap(
    lrs: List[float], 
    budgets: List[int], 
    results_matrix: np.ndarray,
    out_dir: str
) -> str:
    """Create a heatmap showing lr vs budget colored by total loss."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the heatmap
    im = ax.imshow(results_matrix, cmap='viridis_r', aspect='auto', 
                   extent=[budgets[0], budgets[-1], lrs[0], lrs[-1]],
                   origin='lower')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Total Final Loss (lower is better)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Search Budget', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('GA Learning Rate vs Budget: Total Loss Heatmap', fontsize=16, fontweight='bold')
    
    # Set axis scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for each cell
    for i, lr in enumerate(lrs):
        for j, budget in enumerate(budgets):
            loss_val = results_matrix[i, j]
            if not np.isnan(loss_val):
                ax.text(budget, lr, f'{loss_val:.3f}', 
                       ha='center', va='center', fontsize=8, color='white',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
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
    parser.add_argument("--run_name", type=str, required=True, help="Name of the W&B run")
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
    
    # Get checkpoint artifact path
    artifact_path = f"{args.entity}/{args.project}/{args.run_name}"
    print(f"🔍 Using artifact path: {artifact_path}")
    
    # Create output directory
    out_dir = Path("results/ga_lr_budget_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Results matrix: [lr_idx, budget_idx] -> total_loss
    results_matrix = np.full((len(lrs), len(budgets)), np.nan)
    execution_times = np.full((len(lrs), len(budgets)), np.nan)
    
    # CSV logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ga_lr_budget_sweep_{args.run_name}_{timestamp}.csv"
    
    # Run evaluations
    successful_evals = 0
    failed_evals = 0
    
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "timestamp", "run_name", "lr", "budget", "total_loss", "execution_time", "success"
        ])
        
        for i, lr in enumerate(lrs):
            for j, budget in enumerate(budgets):
                print(f"\n🔬 Testing lr = {lr:.6f}, budget = {budget} ({i*len(budgets) + j + 1}/{len(lrs) * len(budgets)})")
                
                success, total_loss, exec_time = run_store_latent_search(
                    artifact_path=artifact_path,
                    lr=lr,
                    budget=budget,
                    dataset_folder=args.dataset_folder,
                    dataset_length=args.dataset_length,
                    dataset_batch_size=args.dataset_batch_size,
                    dataset_use_hf=args.dataset_use_hf,
                    dataset_seed=args.dataset_seed,
                    out_dir=str(out_dir)
                )
                
                # Store results
                if success and total_loss is not None:
                    results_matrix[i, j] = total_loss
                    execution_times[i, j] = exec_time
                    successful_evals += 1
                    print(f"✅ Success: lr={lr:.6f}, budget={budget}, loss={total_loss:.6f}, time={exec_time:.2f}s")
                else:
                    failed_evals += 1
                    print(f"❌ Failed: lr={lr:.6f}, budget={budget}")
                
                # Write to CSV
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    args.run_name,
                    lr,
                    budget,
                    total_loss if total_loss is not None else "",
                    exec_time,
                    success
                ])
    
    # Create heatmap
    if successful_evals > 0:
        print(f"\n📊 Creating heatmap with {successful_evals} successful evaluations...")
        heatmap_path = create_heatmap(lrs, budgets, results_matrix, str(out_dir))
        print(f"📊 Heatmap saved to: {heatmap_path}")
    else:
        print(f"\n❌ No successful evaluations to create heatmap")
        heatmap_path = None
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 GA LR vs BUDGET SWEEP SUMMARY")
    print("=" * 60)
    print(f"Run: {args.run_name}")
    print(f"Successful evaluations: {successful_evals}")
    print(f"Failed evaluations: {failed_evals}")
    print(f"Total evaluations: {len(lrs) * len(budgets)}")
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


if __name__ == "__main__":
    main()


