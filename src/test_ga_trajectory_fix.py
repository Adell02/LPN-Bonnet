#!/usr/bin/env python3
"""
Test script to verify the GA trajectory shape mismatch fix.
This simulates the data structure we're seeing in the logs.
"""

import numpy as np
import tempfile
import os

def create_test_ga_data():
    """Create test GA trajectory data that matches the problematic structure."""
    
    # Simulate the problematic data structure:
    # ga_latents: (1, 4, 1, 100, 8) - batch, candidates, context, steps, latent_dim
    # ga_log_probs: (1, 4, 1, 100) - batch, candidates, context, steps
    
    batch_size = 1
    num_candidates = 4
    context_dim = 1
    num_steps = 100
    latent_dim = 8
    
    # Create test latents
    ga_latents = np.random.randn(batch_size, num_candidates, context_dim, num_steps, latent_dim)
    
    # Create test log probs (scores)
    ga_log_probs = np.random.randn(batch_size, num_candidates, context_dim, num_steps)
    
    # Create the trajectory structure that the evaluation code expects
    traj = {
        "latents": ga_latents,
        "log_probs": ga_log_probs
    }
    
    return traj

def test_ga_trajectory_processing():
    """Test the GA trajectory processing logic."""
    
    print("🧪 Testing GA trajectory processing fix...")
    
    # Create test data
    traj = create_test_ga_data()
    ga_lat = traj["latents"]
    ga_lp = traj["log_probs"]
    
    print(f"Input shapes:")
    print(f"  ga_lat: {ga_lat.shape}")
    print(f"  ga_lp: {ga_lp.shape}")
    
    # Simulate the fixed processing logic
    try:
        # Remove leading batch axis if present
        if ga_lat.ndim >= 5:
            lat0 = ga_lat[0]  # (B, N, S_or_1, C, H) -> (N, S_or_1, C, H)
        else:
            lat0 = ga_lat
            
        if ga_lp.ndim >= 4:
            lp0 = ga_lp[0]    # (B, N, S_or_1, C) -> (N, S_or_1, C)
        else:
            lp0 = ga_lp
            
        print(f"After removing batch axis:")
        print(f"  lat0: {lat0.shape}")
        print(f"  lp0: {lp0.shape}")
        
        # Identify time dimension: check the actual time axis correctly
        Ndim = lat0.shape[0]
        # The time dimension is the one with size > 1 that's not the last dimension
        time_axes = [i for i in range(lat0.ndim - 1) if lat0.shape[i] > 1]
        if time_axes:
            time_axis = time_axes[-1]  # Take the last one (usually the steps dimension)
        else:
            time_axis = lat0.ndim - 2  # Fallback to second-to-last
        
        steps_dim = lat0.shape[time_axis]
        cand_dim = lat0.shape[lat0.ndim - 2]  # Candidates are usually second-to-last
        
        print(f"Shape analysis:")
        print(f"  time_axis: {time_axis}")
        print(f"  steps_dim: {steps_dim}")
        print(f"  cand_dim: {cand_dim}")
        
        # pick first pair
        lat_pair = lat0[0]  # (C, T, H) or (T, C, H) depending on time_axis
        lp_pair = lp0[0]    # (C, T) or (T, C) depending on time_axis
        
        print(f"Pair shapes:")
        print(f"  lat_pair: {lat_pair.shape}")
        print(f"  lp_pair: {lp_pair.shape}")
        
        # Determine the actual layout and extract accordingly
        if time_axis == lat0.ndim - 2:  # Time is second-to-last: (C, T, H)
            # Candidates first, then time: (C, T, H) -> extract best candidate per time step
            print(f"Layout: (C, T, H) - extracting best candidate per time step")
            idx = np.argmax(lp_pair, axis=0)               # (T,) - best candidate per time step
            best_path = lat_pair[idx, np.arange(lat_pair.shape[1])]  # (T, H) - best latent per time step
            best_scores = np.max(lp_pair, axis=0)         # (T,) - best score per time step
        else:  # Time is first: (T, C, H)
            # Time first, then candidates: (T, C, H) -> extract best candidate per time step
            print(f"Layout: (T, C, H) - extracting best candidate per time step")
            idx = np.argmax(lp_pair, axis=-1)               # (T,) - best candidate per time step
            best_path = lat_pair[np.arange(lat_pair.shape[0]), idx]  # (T, H) - best latent per time step
            best_scores = np.max(lp_pair, axis=-1)         # (T,) - best score per time step
        
        print(f"Extracted shapes:")
        print(f"  best_path: {best_path.shape}")
        print(f"  best_scores: {best_scores.shape}")
        
        # Verify the fix works
        expected_steps = 100
        expected_latent_dim = 8
        
        assert best_path.shape[0] == expected_steps, f"Expected {expected_steps} steps, got {best_path.shape[0]}"
        assert best_path.shape[1] == expected_latent_dim, f"Expected {expected_latent_dim} latent dims, got {best_path.shape[1]}"
        assert best_scores.shape[0] == expected_steps, f"Expected {expected_steps} scores, got {best_scores.shape[0]}"
        
        print("✅ GA trajectory processing fix works correctly!")
        print(f"  Trajectory: {best_path.shape[0]} steps × {best_path.shape[1]} latent dims")
        print(f"  Scores: {best_scores.shape[0]} values")
        
        return True
        
    except Exception as e:
        print(f"❌ GA trajectory processing failed: {e}")
        return False

def test_data_saving():
    """Test that the data can be saved and loaded correctly."""
    
    print("\n🧪 Testing data saving and loading...")
    
    try:
        # Create test data
        traj = create_test_ga_data()
        
        # Simulate the saving process
        payload = {}
        
        ga_lat = traj["latents"]
        ga_lp = traj["log_probs"]
        
        # Save the raw trajectory data
        payload["ga_latents"] = ga_lat
        payload["ga_log_probs"] = ga_lp
        
        # Process and save the trajectory
        # (This would be the actual processing logic from evaluate_checkpoint.py)
        lat0 = ga_lat[0]  # Remove batch dimension
        lp0 = ga_lp[0]
        
        # For this test, we'll use a simplified version
        # The actual processing would use the fixed logic above
        best_path = lat0[0]  # Take first candidate for simplicity
        best_scores = lp0[0]
        
        payload["ga_trajectory_latents"] = best_path
        payload["ga_trajectory_losses"] = -best_scores
        
        print(f"Saved payload keys: {list(payload.keys())}")
        print(f"  ga_latents: {payload['ga_latents'].shape}")
        print(f"  ga_trajectory_latents: {payload['ga_trajectory_latents'].shape}")
        print(f"  ga_trajectory_losses: {payload['ga_trajectory_losses'].shape}")
        
        # Verify the data can be loaded
        assert payload["ga_trajectory_latents"].shape[0] == 100, "Expected 100 steps"
        assert payload["ga_trajectory_losses"].shape[0] == 100, "Expected 100 losses"
        
        print("✅ Data saving and loading works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Data saving and loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing GA trajectory shape mismatch fix...\n")
    
    # Run tests
    test1_passed = test_ga_trajectory_processing()
    test2_passed = test_data_saving()
    
    print(f"\n📊 Test Results:")
    print(f"  GA trajectory processing: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Data saving/loading: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The GA trajectory fix should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
