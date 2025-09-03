#!/usr/bin/env python3
"""
Generate and plot ARC and RE-ARC examples with 30x30 grid and 10-color palette.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
import random

# ARC 10-color palette (original)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: light blue
    '#870C25',  # 9: dark red
]

def load_arc_task(task_path):
    """Load an ARC task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)

def grid_to_array(grid, size=30):
    """Convert ARC grid to numpy array, padding to size if needed."""
    if not grid:
        return np.zeros((size, size), dtype=int)
    
    # Convert to numpy array
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    
    # Create padded array
    padded = np.zeros((size, size), dtype=int)
    padded[:rows, :cols] = arr
    return padded

def plot_grid(ax, grid, title, size=30):
    """Plot a grid with ARC colors and soft gray edges."""
    # Create custom colormap
    cmap = ListedColormap(ARC_COLORS)
    
    # Plot the grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    
    # Add soft gray grid lines (lower alpha)
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='lightgray', linewidth=0.3, alpha=0.3)
        ax.axvline(i - 0.5, color='lightgray', linewidth=0.3, alpha=0.3)
    
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return im

def generate_arc_examples():
    """Generate ARC examples from the dataset."""
    arc_dir = "/rds/general/user/ga624/home/Latent-Space-Network/re_arc/arc_original/training"
    re_arc_dir = "/rds/general/user/ga624/home/Latent-Space-Network/re_arc/re_arc/tasks"
    
    # Get available task files
    arc_files = [f for f in os.listdir(arc_dir) if f.endswith('.json')]
    re_arc_files = [f for f in os.listdir(re_arc_dir) if f.endswith('.json')]
    
    # Select 1 random task from each (to get 4 pairs from each)
    selected_arc = random.sample(arc_files, min(1, len(arc_files)))
    selected_re_arc = random.sample(re_arc_files, min(1, len(re_arc_files)))
    
    print(f"Selected ARC examples: {selected_arc}")
    print(f"Selected RE-ARC examples: {selected_re_arc}")
    
    # Create figure - 2 rows, 8 columns (8 input-output pairs: 4 ARC + 4 RE-ARC)
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle('ARC and RE-ARC Examples (30x30 grid, 10-color palette)', fontsize=16, fontweight='bold')
    
    # Plot ARC examples (1 task, 4 examples = 4 pairs)
    task_file = selected_arc[0]
    task_path = os.path.join(arc_dir, task_file)
    task = load_arc_task(task_path)
    
    for j, example in enumerate(task['train'][:4]):  # First 4 training examples
        # Input (top row)
        input_grid = grid_to_array(example['input'])
        plot_grid(axes[0, j], input_grid, f'ARC {task_file[:8]} Train {j+1}')
        
        # Output (bottom row)
        output_grid = grid_to_array(example['output'])
        plot_grid(axes[1, j], output_grid, '')
    
    # Plot RE-ARC examples (1 task, 4 examples = 4 pairs)
    task_file = selected_re_arc[0]
    task_path = os.path.join(re_arc_dir, task_file)
    task = load_arc_task(task_path)
    
    for j, example in enumerate(task[:4]):  # First 4 examples
        # Input (top row)
        input_grid = grid_to_array(example['input'])
        plot_grid(axes[0, j+4], input_grid, f'RE-ARC {task_file[:8]} Example {j+1}')
        
        # Output (bottom row)
        output_grid = grid_to_array(example['output'])
        plot_grid(axes[1, j+4], output_grid, '')
    
    # Add color legend
    legend_elements = [patches.Patch(color=ARC_COLORS[i], label=f'{i}') for i in range(10)]
    fig.legend(handles=legend_elements, title='Color Palette', 
               loc='center', bbox_to_anchor=(0.5, 0.02), ncol=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the plot
    output_path = '/rds/general/user/ga624/home/lpn/arc_rearc_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

def analyze_task_complexity():
    """Analyze the complexity of selected tasks."""
    arc_dir = "/rds/general/user/ga624/home/Latent-Space-Network/re_arc/arc_original/training"
    re_arc_dir = "/rds/general/user/ga624/home/Latent-Space-Network/re_arc/re_arc/tasks"
    
    arc_files = [f for f in os.listdir(arc_dir) if f.endswith('.json')]
    re_arc_files = [f for f in os.listdir(re_arc_dir) if f.endswith('.json')]
    
    print(f"Total ARC tasks: {len(arc_files)}")
    print(f"Total RE-ARC tasks: {len(re_arc_files)}")
    
    # Analyze a few examples
    for task_file in arc_files[:3]:
        task_path = os.path.join(arc_dir, task_file)
        task = load_arc_task(task_path)
        
        print(f"\nARC Task {task_file[:8]}:")
        print(f"  Training examples: {len(task['train'])}")
        print(f"  Test examples: {len(task['test'])}")
        
        # Analyze first training example
        if task['train']:
            train_ex = task['train'][0]
            input_size = f"{len(train_ex['input'])}x{len(train_ex['input'][0]) if train_ex['input'] else 0}"
            output_size = f"{len(train_ex['output'])}x{len(train_ex['output'][0]) if train_ex['output'] else 0}"
            print(f"  First train input size: {input_size}")
            print(f"  First train output size: {output_size}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Generating ARC and RE-ARC examples...")
    generate_arc_examples()
    
    print("\nAnalyzing task complexity...")
    analyze_task_complexity()
