#!/usr/bin/env python3
"""
Combine the three existing struct_pattern datasets into one mixed dataset.

This script loads the existing:
- struct_pattern_1 (96 tasks)
- struct_pattern_2 (96 tasks) 
- struct_pattern_3 (96 tasks)

And combines them into a single dataset with 288 total tasks (96 per pattern type).
This ensures compatibility since we're using the same data format that already works.
"""

import numpy as np
import os

def combine_pattern_datasets():
    """Combine the three struct_pattern datasets into one mixed dataset."""
    print("Combining struct_pattern datasets...")
    
    # Load all three datasets
    datasets = []
    for pattern_id in [1, 2, 3]:
        dataset_dir = f"src/datasets/struct_pattern_{pattern_id}"
        print(f"  Loading {dataset_dir}...")
        
        grids = np.load(os.path.join(dataset_dir, "grids.npy"))
        shapes = np.load(os.path.join(dataset_dir, "shapes.npy"))
        program_ids = np.load(os.path.join(dataset_dir, "program_ids.npy"))
        
        print(f"    Grids: {grids.shape}")
        print(f"    Shapes: {shapes.shape}")
        print(f"    Program IDs: {program_ids.shape}")
        
        # Assign distinct pattern IDs since the original ones are all 0
        assigned_program_ids = np.full_like(program_ids, pattern_id, dtype=np.uint32)
        print(f"    Assigned pattern ID: {pattern_id}")
        
        datasets.append((grids, shapes, assigned_program_ids))
    
    # Combine all datasets
    print("\nCombining datasets...")
    combined_grids = np.concatenate([d[0] for d in datasets], axis=0)
    combined_shapes = np.concatenate([d[1] for d in datasets], axis=0)
    combined_program_ids = np.concatenate([d[2] for d in datasets], axis=0)
    
    print(f"Combined grids shape: {combined_grids.shape}")
    print(f"Combined shapes shape: {combined_shapes.shape}")
    print(f"Combined program IDs shape: {combined_program_ids.shape}")
    
    # Verify pattern distribution
    pattern_counts = {}
    for i in range(len(combined_program_ids)):
        pattern_type = combined_program_ids[i]
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\nPattern distribution:")
    for pattern_type, count in sorted(pattern_counts.items()):
        print(f"  Pattern {pattern_type}: {count} tasks")
    
    return combined_grids, combined_shapes, combined_program_ids

def save_combined_dataset(grids, shapes, program_ids, output_dir="src/datasets/mixed_patterns"):
    """Save the combined dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    grids_path = os.path.join(output_dir, "grids.npy")
    shapes_path = os.path.join(output_dir, "shapes.npy")
    program_ids_path = os.path.join(output_dir, "program_ids.npy")
    
    np.save(grids_path, grids)
    np.save(shapes_path, shapes)
    np.save(program_ids_path, program_ids)
    
    print(f"\nSaved combined dataset to {output_dir}:")
    print(f"  Grids: {grids_path}")
    print(f"  Shapes: {shapes_path}")
    print(f"  Program IDs: {program_ids_path}")
    
    # Verify
    print(f"\nVerification:")
    print(f"  Grids loaded: {np.load(grids_path).shape}")
    print(f"  Shapes loaded: {np.load(shapes_path).shape}")
    print(f"  Program IDs loaded: {np.load(program_ids_path).shape}")
    
    # Verify pattern distribution in saved file
    saved_pids = np.load(program_ids_path)
    unique_pids, counts = np.unique(saved_pids, return_counts=True)
    print(f"  Pattern distribution in saved file:")
    for pid, count in zip(unique_pids, counts):
        print(f"    Pattern {pid}: {count} tasks")

def main():
    """Main function."""
    print("=" * 60)
    print("STRUCT PATTERN DATASET COMBINER")
    print("=" * 60)
    
    try:
        # Combine the datasets
        grids, shapes, program_ids = combine_pattern_datasets()
        
        # Save the combined dataset
        save_combined_dataset(grids, shapes, program_ids)
        
        print("\n" + "=" * 60)
        print("✅ DATASETS COMBINED SUCCESSFULLY!")
        print("=" * 60)
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
        print(f"\n❌ Error combining datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
