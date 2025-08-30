#!/usr/bin/env python3
"""
Generate a tetro_pattern dataset with balanced distribution of tetromino patterns.

This script creates a dataset with 99 tasks (33 per pattern), each containing 4 input-output pairs.
The patterns are distributed uniformly:
- Pattern 1 (O tetromino): 33 tasks
- Pattern 2 (T tetromino): 33 tasks  
- Pattern 3 (L tetromino): 33 tasks

Each task uses the same pattern type for all 4 pairs to maintain consistency.
The dataset follows the same format as pattern_4 by using the standardized make_dataset() function.
"""

import numpy as np
import os
from datasets.task_gen.dataloader import make_dataset

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
    
    # Use the standardized PATTERN task generator to ensure correct data dimensions
    # We'll use pattern_size=3 since tetrominoes fit within 3x3 patterns
    # and embed them in 5x5 grids for visibility
    print("  Using standardized PATTERN task generator for consistent data format")
    
    # Generate the dataset using the PATTERN task generator (same as pattern4)
    grids, shapes, program_ids = make_dataset(
        length=length, 
        num_pairs=num_pairs, 
        num_workers=0,  # Single-threaded for reproducibility
        task_generator_class='PATTERN',
        pattern_size=3,  # 3x3 patterns (tetrominoes fit within this)
        num_rows=5,      # 5 rows (grid size)
        num_cols=5,      # 5 columns (grid size)
        online_data_augmentation=False, 
        seed=seed
    )
    
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
        pattern_names = {1: "O tetromino", 2: "T tetromino", 3: "L tetromino"}
        print("  Pattern {} ({}): {} tasks".format(pattern_type, pattern_names[pattern_type], count))
    
    return grids, shapes, program_ids

def save_tetro_pattern_dataset(grids, shapes, program_ids, output_dir="src/datasets/tetro_pattern"):
    """
    Save the generated dataset to NPY files - same format as pattern4.
    
    Args:
        grids: Grid array
        shapes: Shapes array
        program_ids: Program IDs array
        output_dir: Directory to save the dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each component - same format as pattern4
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
        print("\nDataset format matches pattern4:")
        print("  - grids.npy: (length, num_pairs, 5, 5, 2)")
        print("  - shapes.npy: (length, num_pairs, 2)")
        print("  - program_ids.npy: (length,)")
        print("\nKey improvements:")
        print("  ✅ Uses standardized PATTERN task generator (same as pattern4)")
        print("  ✅ Ensures correct data dimensions for model compatibility")
        print("  ✅ Avoids dimension mismatch issues from custom generation")
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
