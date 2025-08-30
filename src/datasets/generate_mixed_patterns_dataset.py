#!/usr/bin/env python3
"""
Generate a mixed_patterns dataset with balanced distribution of pattern types 1, 2, and 3.

This script creates a dataset with 99 tasks (33 per pattern), each containing 4 input-output pairs.
The patterns are distributed uniformly:
- Pattern 1: 33 tasks
- Pattern 2: 33 tasks  
- Pattern 3: 33 tasks

Each task uses the same pattern type for all 4 pairs to maintain consistency.
This dataset uses the same data generation pipeline as pattern_4, ensuring compatibility
with store_latent_search.py and the evaluation pipeline.
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

def generate_mixed_patterns_dataset(length=99, num_pairs=4, seed=42):
    """
    Generate a balanced mixed-patterns dataset with all three pattern types.
    
    Args:
        length: Number of pattern tasks to generate (should be divisible by 3)
        num_pairs: Number of input-output pairs per task
        seed: Random seed for reproducibility
    
    Returns:
        grids: Array of shape (length, num_pairs, 10, 10, 2) containing the grids
        shapes: Array of shape (length, num_pairs, 2) containing the grid dimensions
        program_ids: Array of shape (length,) containing program identifiers
    """
    print("Generating mixed_patterns dataset...")
    print("  Length: {} tasks".format(length))
    print("  Pairs per task: {}".format(num_pairs))
    print("  Pattern types: 1, 2, 3")
    print("  Grid size: 10x10")
    print("  Seed: {}".format(seed))
    
    # Ensure length is divisible by 3 for perfect uniformity
    if length % 3 != 0:
        print("Warning: Length {} is not divisible by 3. Using {} tasks for perfect uniformity.".format(length, length - (length % 3)))
        length = length - (length % 3)
    
    tasks_per_pattern = length // 3
    print("  Tasks per pattern: {}".format(tasks_per_pattern))
    
    # Initialize arrays
    all_grids = []
    all_shapes = []
    all_program_ids = []
    
    # Generate datasets for each pattern type using the compatible make_dataset function
    for pattern_id in [1, 2, 3]:
        print("  Generating {} tasks for pattern {}...".format(tasks_per_pattern, pattern_id))
        
        # Generate dataset for this specific pattern using the same pipeline as pattern_4
        grids, shapes, program_ids = make_dataset(
            length=tasks_per_pattern,
            num_pairs=num_pairs,
            num_workers=0,  # Single-threaded for reproducibility
            task_generator_class='PATTERN',
            pattern=pattern_id,  # Fixed pattern for this batch
            pattern_per_task=True,  # Use same pattern for all pairs in a task
            num_rows=10,
            num_cols=10,
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
    print("  Total grids shape: {} (should be ({}, {}, 10, 10, 2))".format(final_grids.shape, length, num_pairs))
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
        print("  Pattern {}: {} tasks".format(pattern_type, count))
    
    return final_grids, final_shapes, final_program_ids

def save_mixed_patterns_dataset(grids, shapes, program_ids, output_dir="src/datasets/mixed_patterns"):
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
    """Main function to generate and save the mixed_patterns dataset."""
    # Configuration
    LENGTH = 99      # Number of pattern tasks (divisible by 3 for perfect uniformity)
    NUM_PAIRS = 4     # Input-output pairs per task
    SEED = 42         # Random seed for reproducibility
    
    print("=" * 70)
    print("MIXED PATTERNS DATASET GENERATOR")
    print("=" * 70)
    
    try:
        # Generate the dataset
        grids, shapes, program_ids = generate_mixed_patterns_dataset(
            length=LENGTH, 
            num_pairs=NUM_PAIRS, 
            seed=SEED
        )
        
        # Save the dataset
        save_mixed_patterns_dataset(grids, shapes, program_ids)
        
        print("\n" + "=" * 70)
        print("✅ MIXED PATTERNS DATASET GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou can now use this dataset with store_latent_search.py:")
        print("python src/store_latent_search.py \\")
        print("    --wandb_artifact_path \"your_artifact_path\" \\")
        print("    --budget 2000 \\")
        print("    -d mixed_patterns \\")
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
