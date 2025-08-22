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
    acc_B: Optional[np.ndarray] = None,
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

    # Choose data to display: if acc_B is provided, show difference; else show acc_A
    if acc_B is not None:
        data = acc_A - acc_B
        colorbar_title = f"Accuracy diff\n({method_A_name} − {method_B_name})"
    else:
        data = acc_A
        colorbar_title = f"Accuracy\n({method_A_name})"

    data_masked = np.ma.masked_invalid(data)
    vmax = float(np.nanmax(np.abs(data_masked))) if data_masked.count() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot in index space so cells align with ticks and labels
    im = ax.imshow(
        data_masked,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        vmin=(-vmax if acc_B is not None else 0.0),
        vmax=+vmax,
    )

    # Zero contour (also in index space)
    B, S = data.shape
    X, Y = np.meshgrid(np.arange(S), np.arange(B))
    cs = None
    if acc_B is not None:
        try:
            cs = ax.contour(X, Y, data, levels=[0.0], colors='black', linewidths=2.0, alpha=0.9)
            if cs.collections:
                cs.collections[0].set_label('Equal accuracy (A = B)')
        except (ValueError, RuntimeError, TypeError):
            cs = None

    # Axis labels/title
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Search budget", fontsize=12)
    # Removed axis title to avoid overlap with suptitle

    # Tick locations = indices; tick labels = actual values
    ax.set_xticks(np.arange(S))
    ax.set_yticks(np.arange(B))
    ax.set_xticklabels(steps)
    ax.set_yticklabels(budgets)
    if S > 12:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')

    # Layout helper
    divider = make_axes_locatable(ax)

    # Colorbar axis
    cax = divider.append_axes("right", size="4%", pad=0.6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title(colorbar_title, fontsize=11, pad=10, rotation=0, loc='center')
    cbar.ax.tick_params(length=3, pad=3)

    # Right-side explanatory labels
    label_ax = divider.append_axes("right", size="12%", pad=0.8)
    label_ax.axis("off")
    label_ax.text(0.05, 0.95, f"{method_A_name}\nmore accurate", ha="left", va="top", fontsize=9)
    label_ax.text(0.05, 0.05, f"{method_B_name}\nmore accurate", ha="left", va="bottom", fontsize=9)

    # Legend for the contour
    if cs and cs.collections:
        from matplotlib.lines import Line2D
        ax.legend([Line2D([0], [0], color='black', lw=2)], ['Equal accuracy'], loc='upper left', frameon=True)

    fig.tight_layout()
    return fig


def visualize_single_method_heatmap(
    steps: np.ndarray,
    accuracies: np.ndarray,
    method_name: str,
    metric_name: str
) -> plt.Figure:
    """
    Create a heatmap for a single method showing training progression vs accuracy.
    
    Args:
        steps: 1D array of training steps [S]
        accuracies: 1D array of accuracies [S] 
        method_name: Name of the method
        metric_name: Name of the metric being plotted
        
    Returns:
        Figure showing heatmap of accuracy progression
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Ensure numpy arrays
    steps = np.asarray(steps)
    accuracies = np.asarray(accuracies, dtype=float)
    
    # Create 2D data for heatmap (repeat accuracy values across a single row)
    data_2d = accuracies.reshape(1, -1)  # Shape: (1, S)
    
    # Mask invalid values
    data_masked = np.ma.masked_invalid(data_2d)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot heatmap
    im = ax.imshow(
        data_masked,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
    )
    
    # Set axis labels and title
    ax.set_xlabel("Training Progress", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"{method_name.replace('_', ' ').title()} - {metric_name.replace('_', ' ').title()}", fontsize=14)
    
    # Set ticks
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks([0])
    ax.set_yticklabels([""])  # Hide y-axis labels since we only have one row
    
    # Rotate x-axis labels if there are many steps
    if len(steps) > 8:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_title("Accuracy", fontsize=11, pad=10, rotation=0, loc='center')
    cbar.ax.tick_params(length=3, pad=3)
    
    # Add accuracy values as text on the heatmap
    for i, (step, acc) in enumerate(zip(steps, accuracies)):
        if not np.isnan(acc):
            ax.text(i, 0, f'{acc:.3f}', ha='center', va='center', 
                   color='white' if acc > 0.5 else 'black', fontweight='bold', fontsize=10)
    
    fig.tight_layout()
    return fig


def load_csv_data(csv_path: str) -> Tuple[Dict, List[int], List[int], List[str]]:
    """
    Load data from CSV and organize it for plotting.
    
    Returns:
        data_by_metric: Dict with structure {metric: {method: {step: {budget: value}}}}
        steps: List of unique training steps (including checkpoint versions)
        budgets: List of unique budget values
    """
    # Initialize structure for all metrics we care about
    metric_names = [
        "overall_accuracy",
        "top_1_shape_accuracy",
        "top_1_accuracy",
        "top_1_pixel_correctness",
        "top_2_shape_accuracy",
        "top_2_accuracy",
        "top_2_pixel_correctness",
    ]
    data_by_metric: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = {m: {} for m in metric_names}
    
    steps_set = set()
    budgets_set = set()
    
    total_rows = 0
    skipped_rows = 0
    valid_rows = 0
    
    methods_found: set = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                step = int(row["checkpoint_step"]) if row["checkpoint_step"] else None
                method = row["method"]
                budget = int(row["budget"]) if row["budget"] else None
                checkpoint_name = row["checkpoint_name"]
                if step is None or budget is None:
                    skipped_rows += 1
                    continue
                methods_found.add(method)
                
                # Map version to training progress (vN -> N)
                version_match = 0
                if "--checkpoint:" in checkpoint_name:
                    version_part = checkpoint_name.split("--checkpoint:")[1]
                    try:
                        version_match = int(version_part[1:])
                    except ValueError:
                        version_match = 0
                training_progress = version_match
                
                # Store each metric (use NaN for missing)
                for metric in metric_names:
                    val_str = row.get(metric, None)
                    if val_str in ("", None, "nan", "NaN"):
                        val = float("nan")
                    else:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = float("nan")
                    data_by_metric[metric].setdefault(method, {}).setdefault(training_progress, {})[budget] = val
                
                steps_set.add(training_progress)
                budgets_set.add(budget)
                valid_rows += 1
            except (ValueError, KeyError):
                skipped_rows += 1
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
    
    return data_by_metric, steps, budgets, sorted(list(methods_found))


def create_plot_matrices_for_metric(data_by_metric: Dict, steps: List[int], budgets: List[int], 
                                   metric: str, method_a: str, method_b: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
    """
    Build matrices for plotting. If two methods are provided, align budgets by rank:
    i-th smallest budget of method_a pairs with i-th smallest budget of method_b.
    Returns (acc_A, acc_B, budgets_labels) where budgets_labels are display labels per row.
    """
    data_A = data_by_metric.get(metric, {}).get(method_a, {})
    data_B = data_by_metric.get(metric, {}).get(method_b, {}) if method_b is not None else {}

    # Collect sorted unique budgets per method
    budgets_a_sorted = sorted({b for step_map in data_A.values() for b in step_map.keys()})
    budgets_b_sorted = sorted({b for step_map in data_B.values() for b in step_map.keys()}) if method_b is not None else []

    if method_b is not None and budgets_b_sorted:
        pair_count = min(len(budgets_a_sorted), len(budgets_b_sorted))
        if pair_count == 0:
            # Fallback to single-method plot
            acc_A = np.full((len(budgets_a_sorted), len(steps)), np.nan)
            for j, step in enumerate(steps):
                for i, b in enumerate(budgets_a_sorted):
                    acc_A[i, j] = data_A.get(step, {}).get(b, np.nan)
            return acc_A, None, budgets_a_sorted

        used_a = budgets_a_sorted[:pair_count]
        used_b = budgets_b_sorted[:pair_count]
        budgets_labels = [f"{a}\u2192{b}" for a, b in zip(used_a, used_b)]  # e.g., "1→26"

        acc_A = np.full((pair_count, len(steps)), np.nan)
        acc_B = np.full((pair_count, len(steps)), np.nan)
        for j, step in enumerate(steps):
            for i in range(pair_count):
                acc_A[i, j] = data_A.get(step, {}).get(used_a[i], np.nan)
                acc_B[i, j] = data_B.get(step, {}).get(used_b[i], np.nan)
        return acc_A, acc_B, budgets_labels
    else:
        # Single-method case
        acc_A = np.full((len(budgets_a_sorted), len(steps)), np.nan)
        for j, step in enumerate(steps):
            for i, b in enumerate(budgets_a_sorted):
                acc_A[i, j] = data_A.get(step, {}).get(b, np.nan)
        return acc_A, None, budgets_a_sorted


def create_readable_step_labels(steps: List[int], max_overall_step: Optional[int] = None) -> List[str]:
    """
    Convert numeric step identifiers to readable training progress labels.
    
    Args:
        steps: List of numeric step identifiers (e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
    Returns:
        List of readable labels (e.g., ["0% (v0)", "11% (v1)", "22% (v2)", ...])
    """
    labels = []
    # Use provided max_overall_step (e.g., max version across all checkpoints) to map vN -> % correctly
    # This keeps vN at N/max_version * 100%, even if v0 is omitted from plotting
    if max_overall_step is None:
        max_overall_step = max(steps) if steps else 1
    for step in steps:
        percentage = int((step / max_overall_step) * 100) if max_overall_step > 0 else 0
        labels.append(f"{percentage}% (v{step})")
    return labels


def plot_optimization_comparison(csv_path: str, output_dir: str = "plots", 
                                method_a: Optional[str] = None, 
                                method_b: Optional[str] = None,
                                save_plots: bool = True,
                                show_plots: bool = False,
                                zdim: Optional[int] = None) -> None:
    """
    Main function to create and save optimization comparison plots.
    """
    print(f"📊 Loading data from: {csv_path}")
    
    # Load data for all metrics
    data_by_metric, steps, budgets, methods_in_csv = load_csv_data(csv_path)
    
    if not steps or not budgets:
        print("❌ No valid data found in CSV")
        return
    
    # Remove the first training step (v0 → 0%), which is empty by design
    original_steps = list(steps)
    steps = [s for s in steps if s != 0]
    if not steps:
        print("❌ After removing v0 (0%), no training steps remain to plot")
        return

    # Create readable step labels
    max_overall_step = max(original_steps) if original_steps else max(steps)
    step_labels = create_readable_step_labels(steps, max_overall_step=max_overall_step)
    
    print(f"📈 Found {len(steps)} training steps (excluding v0):")
    for i, (step, label) in enumerate(zip(steps, step_labels)):
        print(f"   {i+1:2d}. {label} (ID: {step})")
    print(f"💰 Found {len(budgets)} budget values: {budgets}")
    
    # Auto-select methods if not provided: prefer two, else one
    detected_methods = [m for m in methods_in_csv if m]
    if method_a is None and method_b is None:
        if len(detected_methods) >= 2:
            method_a, method_b = detected_methods[:2]
        elif len(detected_methods) == 1:
            method_a, method_b = detected_methods[0], None
        else:
            print("❌ No methods found in CSV")
            return
    elif method_a is None and method_b is not None:
        method_a, method_b = method_b, None
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_names = [
        "overall_accuracy",
        "top_1_shape_accuracy",
        "top_1_accuracy",
        "top_1_pixel_correctness",
        "top_2_shape_accuracy",
        "top_2_accuracy",
        "top_2_pixel_correctness",
    ]

    # Iterate metrics and generate a plot per metric
    for metric in metric_names:
        print(f"\n🎨 Generating plot for metric: {metric}...")
        
        if method_b is not None:
            # Two method comparison - use existing comparison logic
            acc_A, acc_B, budgets_aligned = create_plot_matrices_for_metric(
                data_by_metric, steps, budgets, metric, method_a, method_b
            )
            budgets = budgets_aligned

            print(f"🔍 Data coverage for {metric}:")
            print(f"   {method_a}: {np.sum(~np.isnan(acc_A))} data points")
            print(f"   {method_b}: {np.sum(~np.isnan(acc_B))} data points")

            # Two method comparison
            fig = visualize_optimization_comparison_simple(
                steps=np.array(steps),
                budgets=np.array(budgets),
                acc_A=acc_A,
                acc_B=acc_B,
                method_A_name=method_a.replace('_', ' ').title(),
                method_B_name=method_b.replace('_', ' ').title(),
            )
            
            # Title using --zdim if provided
            title_metric = metric.replace('_', ' ')
            if zdim is not None:
                title = f"Latent search on Z_dim {zdim} - {title_metric}"
            else:
                title = f"Latent search - {title_metric}"
            fig.suptitle(title, fontsize=14, y=0.98)

            # Update x-axis labels to be more readable
            ax = fig.axes[0]
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(step_labels, rotation=45, ha='right')
            ax.set_xlabel("Training Progress", fontsize=12)
            ax.set_ylabel("Search Budget", fontsize=12)
            
        else:
            # Single method - create individual heatmap
            print(f"📊 Single method detected: {method_a}")
            
            # Extract accuracy data for this method and metric
            method_data = data_by_metric.get(metric, {}).get(method_a, {})
            
            # Create accuracy array for each step
            accuracies = []
            for step in steps:
                # For single method, we'll average across all budgets for this step
                step_accuracies = []
                for budget in budgets:
                    if step in method_data and budget in method_data[step]:
                        val = method_data[step][budget]
                        if not np.isnan(val):
                            step_accuracies.append(val)
                
                if step_accuracies:
                    # Average across budgets for this step
                    avg_acc = np.mean(step_accuracies)
                    accuracies.append(avg_acc)
                else:
                    accuracies.append(np.nan)
            
            accuracies = np.array(accuracies)
            
            print(f"🔍 Data coverage for {metric}: {np.sum(~np.isnan(accuracies))} valid steps")
            
            # Create single method heatmap
            fig = visualize_single_method_heatmap(
                steps=np.array(steps),
                accuracies=accuracies,
                method_name=method_a,
                metric_name=metric
            )
            
            # Update x-axis labels to be more readable
            ax = fig.axes[0]
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(step_labels, rotation=45, ha='right')
            ax.set_xlabel("Training Progress", fontsize=12)

        # Save plot
        if save_plots:
            csv_name = Path(csv_path).stem
            if method_b is not None:
                plot_filename = f"optim_comparison_{csv_name}_{metric}_{method_a}_vs_{method_b}.png"
            else:
                plot_filename = f"single_method_{csv_name}_{metric}_{method_a}.png"
            plot_path = output_path / plot_filename
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            print(f"💾 Plot saved to: {plot_path}")
        
        if show_plots:
            plt.show()
        plt.close(fig)

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
    parser.add_argument("--method_a", type=str, default=None, help="First method name (auto-detect if omitted)")
    parser.add_argument("--method_b", type=str, default=None, help="Second method name (auto-detect if omitted)")
    parser.add_argument("--no_save", action="store_true", help="Don't save plots to files")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--zdim", type=int, default=None, help="Latent dimension to show in title")
    
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
        show_plots=args.show,
        zdim=args.zdim,
    )


if __name__ == "__main__":
    main()
