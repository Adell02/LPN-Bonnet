#!/usr/bin/env python3
"""
Generate a dataset with uniform distribution of pattern1, pattern2, and pattern3.

This script creates a dataset with 100 tasks, each containing 4 input-output pairs.
The patterns are distributed uniformly:
- Pattern 1 (O tetromino): ~33 tasks
- Pattern 2 (T tetromino): ~33 tasks  
- Pattern 3 (L tetromino): ~34 tasks

Each task uses the same pattern type for all 4 pairs to maintain consistency.
"""

import numpy as np
import os
import sys
sys.path.append('src')

try:
    from datasets.task_gen.dataloader import make_dataset
except ImportError:
    # Fallback for direct execution
    from src.datasets.task_gen.dataloader import make_dataset

def generate_uniform_patterns_dataset(length=100, num_pairs=4, seed=42):
    """
    Generate a dataset with uniform distribution of the three pattern types.
    
    Args:
        length: Number of pattern tasks to generate (should be divisible by 3 for perfect uniformity)
        num_pairs: Number of input-output pairs per task
        seed: Random seed for reproducibility
    
    Returns:
        grids: Array of shape (length, num_pairs, 5, 5, 2) containing the grids
        shapes: Array of shape (length, num_pairs, 2) containing the grid dimensions
        program_ids: Array of shape (length,) containing program identifiers
    """
    print("Generating uniform patterns dataset...")
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
    
    # Generate datasets for each pattern type
    all_grids = []
    all_shapes = []
    all_program_ids = []
    
    for pattern_id in [1, 2, 3]:
        print("  Generating {} tasks for pattern {}...".format(tasks_per_pattern, pattern_id))
        
        # Generate dataset for this specific pattern
        grids, shapes, program_ids = make_dataset(
            length=tasks_per_pattern,
            num_pairs=num_pairs,
            num_workers=0,  # Single-threaded for reproducibility
            task_generator_class='STRUCT_PATTERN',
            pattern=pattern_id,  # Fixed pattern for this batch
            pattern_per_task=True,  # Use same pattern for all pairs in a task
            num_rows=5,
            num_cols=5,
            online_data_augmentation=False,
            seed=seed + pattern_id  # Different seed for each pattern
        )
        
        all_grids.append(grids)
        all_shapes.append(shapes)
        all_program_ids.append(program_ids)
    
    # Concatenate all patterns
    final_grids = np.concatenate(all_grids, axis=0)
    final_shapes = np.concatenate(all_shapes, axis=0)
    final_program_ids = np.concatenate(all_program_ids, axis=0)
    
    print("\nGenerated dataset:")
    print("  Total grids shape: {} (should be ({}, {}, 5, 5, 2))".format(final_grids.shape, length, num_pairs))
    print("  Total shapes shape: {} (should be ({}, {}, 2))".format(final_shapes.shape, length, num_pairs))
    print("  Total program IDs shape: {} (should be ({},))".format(final_program_ids.shape, length))
    
    # Verify pattern distribution
    pattern_counts = {}
    for i in range(length):
        # Extract pattern type from program_id or infer from task index
        pattern_type = (i // tasks_per_pattern) + 1
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\nPattern distribution:")
    for pattern_type, count in sorted(pattern_counts.items()):
        pattern_names = {1: "O tetromino", 2: "T tetromino", 3: "L tetromino"}
        print("  Pattern {} ({}): {} tasks".format(pattern_type, pattern_names[pattern_type], count))
    
    return final_grids, final_shapes, final_program_ids

def save_uniform_patterns_dataset(grids, shapes, program_ids, output_dir="src/datasets/uniform_patterns"):
    """
    Save the generated dataset to NPZ files.
    
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
    """Main function to generate and save the uniform patterns dataset."""
    # Configuration
    LENGTH = 100      # Number of pattern tasks (will be adjusted to be divisible by 3)
    NUM_PAIRS = 4     # Input-output pairs per task
    SEED = 42         # Random seed for reproducibility
    
    print("=" * 70)
    print("UNIFORM PATTERNS DATASET GENERATOR")
    print("=" * 70)
    
    try:
        # Generate the dataset
        grids, shapes, program_ids = generate_uniform_patterns_dataset(
            length=LENGTH, 
            num_pairs=NUM_PAIRS, 
            seed=SEED
        )
        
        # Save the dataset
        save_uniform_patterns_dataset(grids, shapes, program_ids)
        
        print("\n" + "=" * 70)
        print("✅ UNIFORM PATTERNS DATASET GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou can now use this dataset with evaluate_checkpoint.py:")
        print("python src/evaluate_checkpoint.py \\")
        print("    -w \"your_artifact_path\" \\")
        print("    -d uniform_patterns \\")
        print("    --dataset-length 100 \\")
        print("    --dataset-batch-size 8 \\")
        print("    --dataset-use-hf false \\")
        print("    --dataset-seed 0 \\")
        print("    -i gradient_ascent \\")
        print("    --num-steps 20 \\")
        print("    --lr 0.1")
        
    except Exception as e:
        print("\n❌ Error generating dataset: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
