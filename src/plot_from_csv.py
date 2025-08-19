#!/usr/bin/env python3
"""
Plot optimization comparison from CSV data using a simplified visualization function.

Usage:
    python3 src/plot_from_csv.py --csv results/eval_908l681z.csv
    python3 src/plot_from_csv.py --csv results/eval_908l681z.csv --output_dir plots
    python3 src/plot_from_csv.py --csv results/eval_908l681z.csv --methods gradient_ascent,random_search
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Simplified visualization function that doesn't depend on external modules
def visualize_optimization_comparison_simple(
    steps: np.ndarray,
    budgets: np.ndarray, 
    acc_A: np.ndarray,
    acc_B: np.ndarray,
    method_A_name: str = "Method A",
    method_B_name: str = "Method B"
) -> plt.Figure:
    """
    Simplified version of visualize_optimization_comparison that works without chex.
    
    Args:
        steps: 1D array of training steps [S]
        budgets: 1D array of search budgets [B] 
        acc_A: 2D array of accuracies for method A [B, S]
        acc_B: 2D array of accuracies for method B [B, S]
        method_A_name: Name of first method
        method_B_name: Name of second method
        
    Returns:
        Figure showing heatmap of accuracy differences with crossing contour
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Ensure numpy arrays
    steps = np.asarray(steps)
    budgets = np.asarray(budgets)
    acc_A = np.asarray(acc_A, dtype=float)
    acc_B = np.asarray(acc_B, dtype=float)

    # diff heatmap data [B,S]
    diff = acc_A - acc_B
    diff_masked = np.ma.masked_invalid(diff)
    if diff_masked.count() > 0:
        vmax = float(np.nanmax(np.abs(diff_masked))) or 1.0
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(12, 8))

    # heatmap
    # Handle single-point axes for sane extents
    x0, x1 = (steps[0] - 0.5, steps[0] + 0.5) if steps.size == 1 else (steps[0], steps[-1])
    y0, y1 = (budgets[0] - 0.5, budgets[0] + 0.5) if budgets.size == 1 else (budgets[0], budgets[-1])
    im = ax.imshow(
        diff_masked,
        extent=[x0, x1, y0, y1],
        origin='lower', aspect='auto',
        cmap='viridis', vmin=-vmax, vmax=+vmax
    )

    # zero contour A==B, and make it show in legend
    X, Y = np.meshgrid(steps, budgets)
    try:
        cs = ax.contour(X, Y, diff, levels=[0.0], colors='black', linewidths=2.0, alpha=0.9)
        # label the first collection so legend picks it up
        if cs.collections:
            cs.collections[0].set_label('Equal accuracy (A = B)')
    except (ValueError, RuntimeError, TypeError):
        cs = None  # ignore if not possible

    # axes labels/title
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Search budget", fontsize=12)
    ax.set_title(f"Optimization strategies comparison\n({method_A_name} vs {method_B_name})", fontsize=14)

    # all ticks
    ax.set_xticks(steps)
    ax.set_yticks(budgets)
    if steps.size > 12:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')

    # layout helper
    divider = make_axes_locatable(ax)

    # colorbar axis
    cax = divider.append_axes("right", size="4%", pad=0.6)
    cbar = fig.colorbar(im, cax=cax)
    # colorbar title ABOVE, horizontal
    cbar.ax.set_title(f"Accuracy diff\n({method_A_name} − {method_B_name})",
                      fontsize=11, pad=10, rotation=0, loc='center')
    cbar.ax.tick_params(length=3, pad=3)

    # separate slim axis to the RIGHT of the colorbar for the explanatory texts (increase padding to avoid overlap)
    label_ax = divider.append_axes("right", size="12%", pad=0.8)
    label_ax.axis("off")
    # two-line labels, centered vertically near top and bottom of this axis
    label_ax.text(0.05, 0.95, f"{method_A_name}\nmore accurate",
                  ha="left", va="top", fontsize=9)
    label_ax.text(0.05, 0.05, f"{method_B_name}\nmore accurate",
                  ha="left", va="bottom", fontsize=9)

    # legend: include contour and optionally a proxy for heatmap
    handles = []
    labels  = []
    if cs and cs.collections:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='black', lw=2))
        labels.append('Equal accuracy')
    # If you want a legend entry for "A−B heatmap", add a proxy
    # handles.append(Line2D([0],[0], color='none')) ; labels.append('A−B heatmap')
    if handles:
        ax.legend(handles, labels, loc='upper left', frameon=True)

    fig.tight_layout()
    return fig


def load_csv_data(csv_path: str) -> Tuple[Dict, List[int], List[int]]:
    """
    Load data from CSV and organize it for plotting.
    
    Returns:
        data: Dictionary with structure {method: {step: {budget: accuracy}}}
        steps: List of unique training steps (including checkpoint versions)
        budgets: List of unique budget values
    """
    data = {
        "gradient_ascent": {},
        "random_search": {}
    }
    
    steps_set = set()
    budgets_set = set()
    
    # Debug counters
    total_rows = 0
    skipped_rows = 0
    valid_rows = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                # Extract data
                step = int(row["checkpoint_step"]) if row["checkpoint_step"] else None
                method = row["method"]
                budget = int(row["budget"]) if row["budget"] else None
                
                # Handle missing accuracy values - convert empty strings to NaN
                accuracy_str = row["overall_accuracy"]
                if accuracy_str in ("", None, "nan", "NaN"):
                    accuracy = np.nan
                else:
                    try:
                        accuracy = float(accuracy_str)
                    except ValueError:
                        accuracy = np.nan
                
                checkpoint_name = row["checkpoint_name"]
                
                if step is None or budget is None:
                    skipped_rows += 1
                    continue
                
                # Create training progress step from checkpoint version
                # Extract version number from checkpoint name (e.g., "v0", "v1", "v2")
                version_match = 0  # Default to v0
                if "--checkpoint:" in checkpoint_name:
                    version_part = checkpoint_name.split("--checkpoint:")[1]
                    try:
                        version_match = int(version_part[1:])  # Remove 'v' and convert to int
                    except ValueError:
                        version_match = 0
                
                # Map version to training progress: v0=0, v1=1, v2=2, ..., v9=9
                # This represents training progress from 0% to 100%
                training_progress = version_match
                
                # Store data (even if accuracy is NaN)
                if method not in data:
                    data[method] = {}
                if training_progress not in data[method]:
                    data[method][training_progress] = {}
                data[method][training_progress][budget] = accuracy
                
                # Collect unique values
                steps_set.add(training_progress)
                budgets_set.add(budget)
                
                valid_rows += 1
                
            except (ValueError, KeyError) as e:
                skipped_rows += 1
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    steps = sorted(list(steps_set))
    budgets = sorted(list(budgets_set))
    
    # Debug output
    print(f"📊 Data loading summary:")
    print(f"   Total rows processed: {total_rows}")
    print(f"   Valid rows: {valid_rows}")
    print(f"   Skipped rows: {skipped_rows}")
    print(f"   Unique training steps: {len(steps)}")
    print(f"   Unique budgets: {len(budgets)}")
    
    # Show data structure for debugging
    for method in data:
        print(f"\n🔍 {method.upper()} data structure:")
        for step in sorted(data[method].keys()):
            step_data = data[method][step]
            non_nan_count = sum(1 for v in step_data.values() if not np.isnan(v))
            total_count = len(step_data)
            print(f"   Training step {step} (v{step}): {non_nan_count}/{total_count} non-NaN values")
    
    return data, steps, budgets


def create_plot_matrices(data: Dict, steps: List[int], budgets: List[int], 
                         method_a: str, method_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 2D matrices for plotting from the data.
    
    Returns:
        acc_A: 2D array [budgets, steps] for method A
        acc_B: 2D array [budgets, steps] for method B
    """
    acc_A = np.full((len(budgets), len(steps)), np.nan)
    acc_B = np.full((len(budgets), len(steps)), np.nan)
    
    for j, step in enumerate(steps):
        for k, budget in enumerate(budgets):
            # Method A
            if method_a in data and step in data[method_a] and budget in data[method_a][step]:
                acc_A[k, j] = data[method_a][step][budget]
            
            # Method B
            if method_b in data and step in data[method_b] and budget in data[method_b][step]:
                acc_B[k, j] = data[method_b][step][budget]
    
    return acc_A, acc_B


def create_readable_step_labels(steps: List[int]) -> List[str]:
    """
    Convert numeric step identifiers to readable training progress labels.
    
    Args:
        steps: List of numeric step identifiers (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
    Returns:
        List of readable labels (e.g., ["0% (v0)", "11% (v1)", "22% (v2)", ...])
    """
    labels = []
    for step in steps:
        # Calculate percentage: step / max_step * 100
        if len(steps) > 1:
            percentage = int((step / (len(steps) - 1)) * 100)
        else:
            percentage = 0
        labels.append(f"{percentage}% (v{step})")
    return labels


def plot_optimization_comparison(csv_path: str, output_dir: str = "plots", 
                                method_a: str = "gradient_ascent", 
                                method_b: str = "random_search",
                                save_plots: bool = True,
                                show_plots: bool = False) -> None:
    """
    Main function to create and save optimization comparison plots.
    """
    print(f"📊 Loading data from: {csv_path}")
    
    # Load data
    data, steps, budgets = load_csv_data(csv_path)
    
    if not steps or not budgets:
        print("❌ No valid data found in CSV")
        return
    
    # Create readable step labels
    step_labels = create_readable_step_labels(steps)
    
    print(f"📈 Found {len(steps)} training steps:")
    for i, (step, label) in enumerate(zip(steps, step_labels)):
        print(f"   {i+1:2d}. {label} (ID: {step})")
    print(f"💰 Found {len(budgets)} budget values: {budgets}")
    
    # Check if methods exist
    for method in [method_a, method_b]:
        if method not in data:
            print(f"⚠️  Warning: Method '{method}' not found in CSV")
            print(f"   Available methods: {list(data.keys())}")
            return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create matrices for plotting
    acc_A, acc_B = create_plot_matrices(data, steps, budgets, method_a, method_b)
    
    print(f"🔍 Data coverage:")
    print(f"   {method_a}: {np.sum(~np.isnan(acc_A))} data points")
    print(f"   {method_b}: {np.sum(~np.isnan(acc_B))} data points")
    
    # Debug: Show the actual matrices
    print(f"\n🔍 Matrix shapes:")
    print(f"   acc_A shape: {acc_A.shape}")
    print(f"   acc_B shape: {acc_B.shape}")
    print(f"   steps: {steps}")
    print(f"   budgets: {budgets}")
    
    # Debug: Show non-NaN values in matrices
    print(f"\n🔍 Non-NaN values in matrices:")
    print(f"   {method_a} matrix:")
    for i, budget in enumerate(budgets):
        for j, step in enumerate(steps):
            if not np.isnan(acc_A[i, j]):
                print(f"     Budget {budget}, Step {step}: {acc_A[i, j]:.4f}")
    
    print(f"   {method_b} matrix:")
    for i, budget in enumerate(budgets):
        for j, step in enumerate(steps):
            if not np.isnan(acc_B[i, j]):
                print(f"     Budget {budget}, Step {step}: {acc_B[i, j]:.4f}")
    
    # Generate the comparison plot
    print(f"\n🎨 Generating optimization comparison plot...")
    fig = visualize_optimization_comparison_simple(
        steps=np.array(steps),
        budgets=np.array(budgets),
        acc_A=acc_A,
        acc_B=acc_B,
        method_A_name=method_a.replace('_', ' ').title(),
        method_B_name=method_b.replace('_', ' ').title(),
    )
    
    # Add additional information to the plot
    csv_name = Path(csv_path).stem
    fig.suptitle(f"Optimization Comparison - {csv_name}\n"
                 f"Training Progress: {len(steps)} steps, Budgets: {len(budgets)}", 
                 fontsize=14, y=0.98)
    
    # Update x-axis labels to be more readable
    ax = fig.axes[0]  # Get the main axis
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(step_labels, rotation=45, ha='right')
    
    # Update axis labels to reflect training progress
    ax.set_xlabel("Training Progress", fontsize=12)
    ax.set_ylabel("Search Budget", fontsize=12)
    
    # Save plot
    if save_plots:
        plot_filename = f"optim_comparison_{csv_name}_{method_a}_vs_{method_b}.png"
        plot_path = output_path / plot_filename
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"💾 Plot saved to: {plot_path}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    
    plt.close(fig)
    
    # Create additional analysis plots
    create_analysis_plots(data, steps, budgets, method_a, method_b, output_path, csv_name, step_labels)
    
    print("✅ Plotting complete!")


def create_analysis_plots(data: Dict, steps: List[int], budgets: List[int], 
                         method_a: str, method_b: str, output_path: Path, csv_name: str, step_labels: List[str]) -> None:
    """
    Create additional analysis plots for deeper insights.
    """
    print("📊 Creating additional analysis plots...")
    
    # 1. Performance vs Budget plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Method A performance vs budget
    for step in steps:
        if method_a in data and step in data[method_a]:
            step_data = data[method_a][step]
            step_budgets = sorted(step_data.keys())
            step_accuracies = [step_data[b] for b in step_budgets]
            ax1.plot(step_budgets, step_accuracies, 'o-', label=f'Step {step_labels[steps.index(step)]}', alpha=0.7)
    
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{method_a.replace("_", " ").title()} Performance vs Budget')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Method B performance vs budget
    for step in steps:
        if method_b in data and step in data[method_b]:
            step_data = data[method_b][step]
            step_budgets = sorted(step_data.keys())
            step_accuracies = [step_data[b] for b in step_budgets]
            ax2.plot(step_budgets, step_accuracies, 's-', label=f'Step {step_labels[steps.index(step)]}', alpha=0.7)
    
    ax2.set_xlabel('Budget')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{method_b.replace("_", " ").title()} Performance vs Budget')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save analysis plot
    analysis_filename = f"performance_analysis_{csv_name}_{method_a}_vs_{method_b}.png"
    analysis_path = output_path / analysis_filename
    fig.savefig(analysis_path, dpi=200, bbox_inches='tight')
    print(f"💾 Analysis plot saved to: {analysis_path}")
    plt.close(fig)
    
    # 2. Summary statistics
    print("\n📈 Summary Statistics:")
    print("=" * 50)
    
    for method in [method_a, method_b]:
        if method in data:
            print(f"\n{method.replace('_', ' ').title()}:")
            all_accuracies = []
            for step in steps:
                if step in data[method]:
                    for budget in budgets:
                        if budget in data[method][step]:
                            all_accuracies.append(data[method][step][budget])
            
            if all_accuracies:
                all_accuracies = np.array(all_accuracies)
                print(f"  Mean accuracy: {np.mean(all_accuracies):.4f}")
                print(f"  Std accuracy: {np.std(all_accuracies):.4f}")
                print(f"  Min accuracy: {np.min(all_accuracies):.4f}")
                print(f"  Max accuracy: {np.max(all_accuracies):.4f}")
                print(f"  Total data points: {len(all_accuracies)}")


def main():
    parser = argparse.ArgumentParser(description="Plot optimization comparison from CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--method_a", type=str, default="gradient_ascent", help="First method name")
    parser.add_argument("--method_b", type=str, default="random_search", help="Second method name")
    parser.add_argument("--no_save", action="store_true", help="Don't save plots to files")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not Path(args.csv).exists():
        print(f"❌ CSV file not found: {args.csv}")
        return
    
    # Run plotting
    plot_optimization_comparison(
        csv_path=args.csv,
        output_dir=args.output_dir,
        method_a=args.method_a,
        method_b=args.method_b,
        save_plots=not args.no_save,
        show_plots=args.show
    )


if __name__ == "__main__":
    main()
