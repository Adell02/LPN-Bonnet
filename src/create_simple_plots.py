#!/usr/bin/env python3
"""
Simple script to create evaluation plots from CSV data.
Handles single-checkpoint cases gracefully.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_simple_evaluation_plots(csv_path: str, output_dir: str = "plots") -> None:
    """
    Create simple evaluation plots from CSV data, handling single-checkpoint cases gracefully.
    
    Args:
        csv_path: Path to the CSV file with evaluation results
        output_dir: Directory to save the plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Check if we have data
    if len(df) == 0:
        print("⚠️  No data found in CSV")
        return
    
    # Get unique values
    methods = df['method'].unique()
    budgets = sorted(df['budget'].unique())
    checkpoints = sorted(df['checkpoint_step'].unique())
    
    print(f"📈 Found {len(methods)} methods: {list(methods)}")
    print(f"💰 Found {len(budgets)} budgets: {budgets}")
    print(f"🔍 Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    # Create simple comparison plots
    if len(methods) >= 2 and len(budgets) >= 2:
        # Method comparison by budget
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy by budget for each method
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                # Group by budget and calculate mean accuracy
                budget_means = method_data.groupby('budget')['overall_accuracy'].mean()
                ax1.plot(budget_means.index, budget_means.values, 'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)
        
        ax1.set_xlabel('Search Budget')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Method Performance by Budget')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Loss by budget for each method (if available)
        if 'total_final_loss' in df.columns:
            for method in methods:
                method_data = df[df['method'] == method]
                if len(method_data) > 0:
                    budget_means = method_data.groupby('budget')['total_final_loss'].mean()
                    ax2.plot(budget_means.index, budget_means.values, 'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)
            
            ax2.set_xlabel('Search Budget')
            ax2.set_ylabel('Total Final Loss')
            ax2.set_title('Method Loss by Budget')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            # Use log scale for loss if values vary greatly
            if df['total_final_loss'].max() / df['total_final_loss'].min() > 100:
                ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Loss Data Not Available')
        
        plt.tight_layout()
        comparison_path = output_path / f"method_comparison_{Path(csv_path).stem}.png"
        fig.savefig(comparison_path, dpi=200, bbox_inches='tight')
        print(f"💾 Method comparison plot saved to: {comparison_path}")
        plt.close(fig)
    
    # Create budget comparison plots
    if len(budgets) >= 2:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Accuracy comparison across budgets
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                budget_means = method_data.groupby('budget')['overall_accuracy'].mean()
                budget_stds = method_data.groupby('budget')['overall_accuracy'].std()
                
                axes[0].errorbar(budget_means.index, budget_means.values, 
                                yerr=budget_stds.values, fmt='o-', 
                                label=method.replace('_', ' ').title(), 
                                capsize=5, capthick=2)
        
        axes[0].set_xlabel('Search Budget')
        axes[0].set_ylabel('Overall Accuracy')
        axes[0].set_title('Accuracy vs Budget Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Loss comparison across budgets (if available)
        if 'total_final_loss' in df.columns:
            for method in methods:
                method_data = df[df['method'] == method]
                if len(method_data) > 0:
                    budget_means = method_data.groupby('budget')['total_final_loss'].mean()
                    budget_stds = method_data.groupby('budget')['total_final_loss'].std()
                    
                    axes[1].errorbar(budget_means.index, budget_means.values, 
                                    yerr=budget_stds.values, fmt='o-', 
                                    label=method.replace('_', ' ').title(), 
                                    capsize=5, capthick=2)
            
            axes[1].set_xlabel('Search Budget')
            axes[1].set_ylabel('Total Final Loss')
            axes[1].set_title('Loss vs Budget Comparison')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            if df['total_final_loss'].max() / df['total_final_loss'].min() > 100:
                axes[1].set_yscale('log')
        else:
            axes[1].text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Loss Data Not Available')
        
        plt.tight_layout()
        budget_path = output_path / f"budget_comparison_{Path(csv_path).stem}.png"
        fig.savefig(budget_path, dpi=200, bbox_inches='tight')
        print(f"💾 Budget comparison plot saved to: {budget_path}")
        plt.close(fig)
    
    # Create execution time comparison (if available)
    if 'execution_time' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                budget_means = method_data.groupby('budget')['execution_time'].mean()
                ax.plot(budget_means.index, budget_means.values, 'o-', 
                       label=method.replace('_', ' ').title(), linewidth=2, markersize=8)
        
        ax.set_xlabel('Search Budget')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Method Execution Time by Budget')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        time_path = output_path / f"execution_time_{Path(csv_path).stem}.png"
        fig.savefig(time_path, dpi=200, bbox_inches='tight')
        print(f"💾 Execution time plot saved to: {time_path}")
        plt.close(fig)
    
    # Create summary table
    if len(methods) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                avg_accuracy = method_data['overall_accuracy'].mean()
                avg_loss = method_data['total_final_loss'].mean() if 'total_final_loss' in df.columns else 'N/A'
                avg_time = method_data['execution_time'].mean() if 'execution_time' in df.columns else 'N/A'
                
                summary_data.append([
                    method.replace('_', ' ').title(),
                    f"{avg_accuracy:.4f}",
                    f"{avg_loss:.6f}" if isinstance(avg_loss, (int, float)) else avg_loss,
                    f"{avg_time:.2f}s" if isinstance(avg_time, (int, float)) else avg_time
                ])
        
        # Create table
        columns = ['Method', 'Avg Accuracy', 'Avg Loss', 'Avg Time']
        table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        ax.set_title('Method Performance Summary', fontsize=16, pad=20)
        
        summary_path = output_path / f"summary_table_{Path(csv_path).stem}.png"
        fig.savefig(summary_path, dpi=200, bbox_inches='tight')
        print(f"💾 Summary table saved to: {summary_path}")
        plt.close(fig)
    
    print("✅ Simple evaluation plots created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create simple evaluation plots from CSV data")
    parser.add_argument("--csv", required=True, help="Path to CSV file with evaluation results")
    parser.add_argument("--output_dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    create_simple_evaluation_plots(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
