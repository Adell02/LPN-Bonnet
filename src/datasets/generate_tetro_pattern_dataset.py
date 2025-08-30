#!/usr/bin/env python3
"""
Generate a tetro_pattern dataset with balanced distribution of tetromino patterns.

This script creates a dataset with 99 tasks (33 per pattern), each containing 4 input-output pairs.
The patterns are distributed uniformly:
- Pattern 1 (O tetromino): 33 tasks
- Pattern 2 (T tetromino): 33 tasks  
- Pattern 3 (L tetromino): 33 tasks

Each task uses the same pattern type for all 4 pairs to maintain consistency.
The dataset is compatible with store_latent_search.py and follows the same format as pattern_4.
"""

import numpy as np
import os
import random

def generate_tetro_pattern_dataset(length=99, num_pairs=4, seed=42):
    """
    Generate a balanced tetro_pattern dataset with all three tetromino types.
    
    Args:
        length: Number of pattern tasks to generate (should be divisible by 3)
        num_pairs: Number of input-output pairs per task
        seed: Random seed for reproducibility
    
    Returns:
        grids: Array of shape (length, num_pairs, 5, 5, 2) containing the grids
        shapes: Array of shape (length, num_pairs, 2) containing the grid dimensions
        program_ids: Array of shape (length,) containing program identifiers
    """
    print("Generating tetro_pattern dataset...")
    print("  Length: {} tasks".format(length))
    print("  Pairs per task: {}".format(num_pairs))
    print("  Pattern types: 1 (O), 2 (T), 3 (L)")
    print("  Grid size: 5x5")
    print("  Seed: {}".format(seed))
    
    # Ensure length is divisible by 3 for perfect uniformity
    if length % 3 != 0:
        print("Warning: Length {} is not divisible by 3. Using {} tasks for perfect uniformity.".format(length, length - (length % 3)))
        length = length - (length % 3)
    
    tasks_per_pattern = length // 3
    print("  Tasks per pattern: {}".format(tasks_per_pattern))
    
    # Set random seed
    random.seed(seed)
    
    # Initialize arrays
    grids = np.zeros((length, num_pairs, 5, 5, 2), dtype=np.uint8)
    shapes = np.zeros((length, num_pairs, 2), dtype=np.uint8)
    program_ids = np.zeros(length, dtype=np.uint32)
    
    # Define pattern offsets and bounding boxes
    pattern_definitions = {
        0: {  # O tetromino (2x2) - PATTERN generator uses 0-based indexing
            'offsets': [(0, 0), (0, 1), (1, 0), (1, 1)],
            'box_h': 2, 'box_w': 2
        },
        1: {  # T tetromino (2x3 box)
            'offsets': [(0, 0), (0, 1), (0, 2), (1, 1)],
            'box_h': 2, 'box_w': 3
        },
        2: {  # L tetromino (3x2 box)
            'offsets': [(0, 0), (1, 0), (2, 0), (2, 1)],
            'box_h': 3, 'box_w': 2
        }
    }
    
    task_idx = 0
    for pattern_id in [0, 1, 2]:  # Use 0-based indexing to match PATTERN generator
        pattern_info = pattern_definitions[pattern_id]
        print("  Generating {} tasks for pattern {}...".format(tasks_per_pattern, pattern_id))
        
        for task in range(tasks_per_pattern):
            # Sample colors for this task (consistent across all pairs)
            colors = [random.randint(1, 9) for _ in range(4)]
            
            for pair in range(num_pairs):
                # Generate input grid with single anchor point
                input_grid = np.zeros((5, 5), dtype=np.uint8)
                output_grid = np.zeros((5, 5), dtype=np.uint8)
                
                # Choose random position for pattern
                max_row = 5 - pattern_info['box_h']
                max_col = 5 - pattern_info['box_w']
                top = random.randint(0, max_row)
                left = random.randint(0, max_col)
                
                # Mark anchor in input
                input_grid[top, left] = 1
                
                # Draw pattern in output
                for k, (dr, dc) in enumerate(pattern_info['offsets']):
                    output_grid[top + dr, left + dc] = colors[k % len(colors)]
                
                # Store in arrays
                grids[task_idx, pair, :, :, 0] = input_grid
                grids[task_idx, pair, :, :, 1] = output_grid
                shapes[task_idx, pair, 0] = 5  # num_rows
                shapes[task_idx, pair, 1] = 5  # num_cols
            
            # Set program ID to pattern type
            program_ids[task_idx] = pattern_id
            task_idx += 1
    
    print("\nGenerated dataset:")
    print("  Total grids shape: {} (should be ({}, {}, 5, 5, 2))".format(grids.shape, length, num_pairs))
    print("  Total shapes shape: {} (should be ({}, {}, 2))".format(shapes.shape, length, num_pairs))
    print("  Total program IDs shape: {} (should be ({},))".format(program_ids.shape, length))
    
    # Verify pattern distribution
    pattern_counts = {}
    for i in range(length):
        pattern_type = program_ids[i]
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\nPattern distribution:")
    for pattern_type, count in sorted(pattern_counts.items()):
        pattern_names = {0: "O tetromino", 1: "T tetromino", 2: "L tetromino"}
        print("  Pattern {} ({}): {} tasks".format(pattern_type, pattern_names[pattern_type], count))
    
    return grids, shapes, program_ids

def save_tetro_pattern_dataset(grids, shapes, program_ids, output_dir="src/datasets/tetro_pattern"):
    """
    Save the generated dataset to NPY files.
    
    Args:
        grids: Grid array
        shapes: Shapes array
        program_ids: Program IDs array
        output_dir: Directory to save the dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each component
    grids_path = os.path.join(output_dir, "grids.npy")
    shapes_path = os.path.join(output_dir, "shapes.npy")
    program_ids_path = os.path.join(output_dir, "program_ids.npy")
    
    np.save(grids_path, grids)
    np.save(shapes_path, shapes)
    np.save(program_ids_path, program_ids)
    
    print("\nSaved dataset to {}:".format(output_dir))
    print("  Grids: {}".format(grids_path))
    print("  Shapes: {}".format(shapes_path))
    print("  Program IDs: {}".format(program_ids_path))
    
    # Verify the saved files
    print("\nVerification:")
    print("  Grids loaded: {}".format(np.load(grids_path).shape))
    print("  Shapes loaded: {}".format(np.load(shapes_path).shape))
    print("  Program IDs loaded: {}".format(np.load(program_ids_path).shape))

def main():
    """Main function to generate and save the tetro_pattern dataset."""
    # Configuration
    LENGTH = 99      # Number of pattern tasks (divisible by 3 for perfect uniformity)
    NUM_PAIRS = 4     # Input-output pairs per task
    SEED = 42         # Random seed for reproducibility
    
    print("=" * 70)
    print("TETRO_PATTERN DATASET GENERATOR")
    print("=" * 70)
    
    try:
        # Generate the dataset
        grids, shapes, program_ids = generate_tetro_pattern_dataset(
            length=LENGTH, 
            num_pairs=NUM_PAIRS, 
            seed=SEED
        )
        
        # Save the dataset
        save_tetro_pattern_dataset(grids, shapes, program_ids)
        
        print("\n" + "=" * 70)
        print("✅ TETRO_PATTERN DATASET GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou can now use this dataset with store_latent_search.py:")
        print("python src/store_latent_search.py \\")
        print("    --wandb_artifact_path \"your_artifact_path\" \\")
        print("    --budget 2000 \\")
        print("    -d tetro_pattern \\")
        print("    --dataset-length 4 \\")
        print("    --dataset-batch-size 4 \\")
        print("    --dataset-use-hf false \\")
        print("    --dataset-seed 0 \\")
        print("    --ga_lr 0.2 \\")
        print("    --es_mutation_std 0.5 \\")
        print("    --ga_steps 1000 \\")
        print("    --es_population 100 \\")
        print("    --es_generations 20 \\")
        print("    --mutation_decay 0.9 \\")
        print("    --elite_size 10")
        
    except Exception as e:
        print("\n❌ Error generating dataset: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
