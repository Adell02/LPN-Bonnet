#!/usr/bin/env python3
"""
Quick parameter tuning tool for enhanced uncertainty shaping.

This script helps you quickly adjust the enhanced uncertainty shaping parameters
without manually editing the config file.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_training_with_params(params):
    """Run training with the specified parameters."""
    cmd = ["python", "src/structured_train.py", "--config-name=structured"]
    
    for key, value in params.items():
        cmd.append(f"training.{key}={value}")
    
    print(f"🚀 Running training with parameters: {params}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Tune enhanced uncertainty shaping parameters")
    
    # Add parameter arguments
    parser.add_argument("--beta-target", type=float, help="β_T for target patterns (default: 0.5)")
    parser.add_argument("--beta-other", type=float, help="β_O for other patterns (default: 1.5)")
    parser.add_argument("--prior-var-target", type=float, help="Prior variance for target patterns (default: 0.25)")
    parser.add_argument("--prior-var-other", type=float, help="Prior variance for other patterns (default: 2.0)")
    parser.add_argument("--alpha-target", type=float, help="α_T for target certainty (default: 1.0)")
    parser.add_argument("--alpha-other", type=float, help="α_O for other uncertainty (default: 0.5)")
    parser.add_argument("--uncertainty-margin", type=float, help="Minimum variance threshold (default: 1.0)")
    parser.add_argument("--entropy-gamma", type=float, help="Entropy bonus coefficient (default: 0.1)")
    parser.add_argument("--phase1a-steps", type=int, help="Phase 1A encoder-only steps (default: 200)")
    parser.add_argument("--phase1b-lr-scale", type=float, help="Phase 1B learning rate scale (default: 0.1)")
    
    # Add preset arguments
    parser.add_argument("--preset", choices=["conservative", "aggressive", "balanced", "debug"], 
                       help="Use a preset parameter configuration")
    
    args = parser.parse_args()
    
    # Define presets
    presets = {
        "conservative": {
            "beta_target": 0.7,
            "beta_other": 1.2,
            "prior_var_target": 0.5,
            "prior_var_other": 1.5,
            "alpha_target": 0.5,
            "alpha_other": 0.3,
            "entropy_gamma": 0.05,
            "phase1a_steps": 100,
            "phase1b_lr_scale": 0.2
        },
        "aggressive": {
            "beta_target": 0.2,
            "beta_other": 2.5,
            "prior_var_target": 0.1,
            "prior_var_other": 3.0,
            "alpha_target": 2.0,
            "alpha_other": 1.0,
            "entropy_gamma": 0.2,
            "phase1a_steps": 500,
            "phase1b_lr_scale": 0.05
        },
        "balanced": {
            "beta_target": 0.5,
            "beta_other": 1.5,
            "prior_var_target": 0.25,
            "prior_var_other": 2.0,
            "alpha_target": 1.0,
            "alpha_other": 0.5,
            "entropy_gamma": 0.1,
            "phase1a_steps": 200,
            "phase1b_lr_scale": 0.1
        },
        "debug": {
            "beta_target": 0.1,
            "beta_other": 3.0,
            "prior_var_target": 0.05,
            "prior_var_other": 4.0,
            "alpha_target": 3.0,
            "alpha_other": 1.5,
            "entropy_gamma": 0.3,
            "phase1a_steps": 1000,
            "phase1b_lr_scale": 0.01
        }
    }
    
    # Build parameter dictionary
    params = {}
    
    if args.preset:
        params.update(presets[args.preset])
        print(f"🎯 Using {args.preset} preset")
    
    # Override with command-line arguments
    if args.beta_target is not None:
        params["beta_target"] = args.beta_target
    if args.beta_other is not None:
        params["beta_other"] = args.beta_other
    if args.prior_var_target is not None:
        params["prior_var_target"] = args.prior_var_target
    if args.prior_var_other is not None:
        params["prior_var_other"] = args.prior_var_other
    if args.alpha_target is not None:
        params["alpha_target"] = args.alpha_target
    if args.alpha_other is not None:
        params["alpha_other"] = args.alpha_other
    if args.uncertainty_margin is not None:
        params["uncertainty_margin"] = args.uncertainty_margin
    if args.entropy_gamma is not None:
        params["entropy_gamma"] = args.entropy_gamma
    if args.phase1a_steps is not None:
        params["phase1a_steps"] = args.phase1a_steps
    if args.phase1b_lr_scale is not None:
        params["phase1b_lr_scale"] = args.phase1b_lr_scale
    
    if not params:
        print("❌ No parameters specified. Use --preset or individual parameter arguments.")
        print("\nAvailable presets:")
        for preset_name, preset_params in presets.items():
            print(f"  {preset_name}: {preset_params}")
        print("\nExample usage:")
        print("  python src/tune_uncertainty.py --preset aggressive")
        print("  python src/tune_uncertainty.py --beta-target 0.3 --alpha-target 1.5")
        return 1
    
    # Validate parameters
    if "beta_target" in params and params["beta_target"] >= 1.0:
        print("⚠️  Warning: beta_target should be < 1.0 for good specialization")
    if "beta_other" in params and params["beta_other"] < 1.0:
        print("⚠️  Warning: beta_other should be >= 1.0 for standard KL")
    if "prior_var_target" in params and params["prior_var_target"] >= 1.0:
        print("⚠️  Warning: prior_var_target should be < 1.0 for target patterns")
    if "prior_var_other" in params and params["prior_var_other"] <= 1.0:
        print("⚠️  Warning: prior_var_other should be > 1.0 for other patterns")
    
    # Run training
    success = run_training_with_params(params)
    
    if success:
        print("\n✅ Training completed! Check your logs/WandB dashboard for results.")
        print("\nKey metrics to monitor:")
        print("  - enhanced_uncertainty/avg_target_var (should be low, < 0.5)")
        print("  - enhanced_uncertainty/avg_other_var (should be high, > 1.5)")
        print("  - enhanced_uncertainty/specialization_ratio (should be high, > 3.0)")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
