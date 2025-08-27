#!/usr/bin/env python3
"""
Generate a pattern_4 dataset for use with store_latent_search.py

This script creates a dataset with 4x4 patterns that matches the model's training configuration:
- pattern_size: 4 (4x4 patterns)
- num_rows: 4, num_cols: 4
- Compatible with models trained on pattern_size=4
"""

import numpy as np
import os
from datasets.task_gen.dataloader import make_dataset

def generate_pattern4_dataset(length=100, num_pairs=4, seed=0):
    """
    Generate a pattern_4 dataset with 4x4 grids.
    
    Args:
        length: Number of pattern tasks to generate
        num_pairs: Number of input-output pairs per task
        seed: Random seed for reproducibility
    
    Returns:
        grids: Array of shape (length, num_pairs, 4, 4, 2) containing the grids
        shapes: Array of shape (length, num_pairs, 2) containing the grid dimensions
        program_ids: Array of shape (length,) containing program identifiers
    """
    print(f"Generating pattern_4 dataset...")
    print(f"  Length: {length} tasks")
    print(f"  Pairs per task: {num_pairs}")
    print(f"  Pattern size: 4x4")
    print(f"  Seed: {seed}")
    
    # Generate the dataset using the PATTERN task generator
    grids, shapes, program_ids = make_dataset(
        length=length, 
        num_pairs=num_pairs, 
        num_workers=0,  # Single-threaded for reproducibility
        task_generator_class='PATTERN',
        pattern_size=4,  # 4x4 patterns
        num_rows=4,      # 4 rows
        num_cols=4,      # 4 columns
        online_data_augmentation=False, 
        seed=seed
    )
    
    print(f"Generated dataset:")
    print(f"  Grids shape: {grids.shape}")
    print(f"  Shapes shape: {shapes.shape}")
    print(f"  Program IDs shape: {program_ids.shape}")
    
    return grids, shapes, program_ids

def save_pattern4_dataset(grids, shapes, program_ids, output_dir="src/datasets/pattern_4"):
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
    
    print(f"Saved dataset to {output_dir}:")
    print(f"  Grids: {grids_path}")
    print(f"  Shapes: {shapes_path}")
    print(f"  Program IDs: {program_ids_path}")
    
    # Verify the saved files
    print(f"\nVerification:")
    print(f"  Grids loaded: {np.load(grids_path).shape}")
    print(f"  Shapes loaded: {np.load(shapes_path).shape}")
    print(f"  Program IDs loaded: {np.load(program_ids_path).shape}")

def main():
    """Main function to generate and save the pattern_4 dataset."""
    # Configuration
    LENGTH = 100      # Number of pattern tasks
    NUM_PAIRS = 4     # Input-output pairs per task
    SEED = 42         # Random seed for reproducibility
    
    print("=" * 60)
    print("PATTERN_4 DATASET GENERATOR")
    print("=" * 60)
    
    try:
        # Generate the dataset
        grids, shapes, program_ids = generate_pattern4_dataset(
            length=LENGTH, 
            num_pairs=NUM_PAIRS, 
            seed=SEED
        )
        
        # Save the dataset
        save_pattern4_dataset(grids, shapes, program_ids)
        
        print("\n" + "=" * 60)
        print("✅ PATTERN_4 DATASET GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nYou can now use this dataset with store_latent_search.py:")
        print(f"python src/store_latent_search.py \\")
        print(f"    --wandb_artifact_path \"your_artifact_path\" \\")
        print(f"    --budget 20 \\")
        print(f"    -d pattern_4 \\")
        print(f"    --dataset-length 1 \\")
        print(f"    --dataset-batch-size 1 \\")
        print(f"    --dataset-use-hf false \\")
        print(f"    --dataset-seed 0")
        
    except Exception as e:
        print(f"\n❌ Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
