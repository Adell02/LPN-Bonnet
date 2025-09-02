#!/usr/bin/env python3
"""
Verification script for enhanced uncertainty shaping implementation.

This script checks if the enhanced uncertainty shaping is properly integrated
and provides a quick way to test the parameters.
"""

import sys
import os
import yaml
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

def check_config_parameters():
    """Check if enhanced uncertainty shaping parameters are in the config."""
    print("🔍 Checking configuration parameters...")
    
    config_path = Path("src/configs/structured.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found: src/configs/structured.yaml")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_params = [
        'beta_target', 'beta_other', 'prior_var_target', 'prior_var_other',
        'alpha_target', 'alpha_other', 'uncertainty_margin', 'entropy_gamma',
        'logvar_min', 'logvar_max', 'variance_floor_other', 'phase1a_steps', 'phase1b_lr_scale'
    ]
    
    missing_params = []
    found_params = []
    
    training_config = config.get('training', {})
    for param in required_params:
        if param in training_config:
            found_params.append(f"  ✅ {param}: {training_config[param]}")
        else:
            missing_params.append(f"  ❌ {param}: MISSING")
    
    print("Enhanced Uncertainty Shaping Parameters:")
    for param in found_params:
        print(param)
    
    if missing_params:
        print("\nMissing parameters:")
        for param in missing_params:
            print(param)
        return False
    
    print(f"\n✅ All {len(required_params)} parameters found in configuration!")
    return True

def check_model_integration():
    """Check if the enhanced uncertainty shaping is integrated in the model."""
    print("\n🔍 Checking model integration...")
    
    model_path = Path("src/models/structured_lpn.py")
    if not model_path.exists():
        print("❌ Model file not found: src/models/structured_lpn.py")
        return False
    
    with open(model_path, 'r') as f:
        model_content = f.read()
    
    required_methods = [
        '_compute_enhanced_uncertainty_shaping',
        'enhanced_uncertainty_loss',
        'enhanced_uncertainty_metrics'
    ]
    
    found_methods = []
    for method in required_methods:
        if method in model_content:
            found_methods.append(f"  ✅ {method}")
        else:
            print(f"  ❌ {method}: MISSING")
    
    if len(found_methods) == len(required_methods):
        print("Enhanced uncertainty shaping methods found:")
        for method in found_methods:
            print(method)
        print("\n✅ Model integration complete!")
        return True
    else:
        print(f"\n❌ Only {len(found_methods)}/{len(required_methods)} methods found!")
        return False

def check_training_integration():
    """Check if the enhanced uncertainty shaping is integrated in training."""
    print("\n🔍 Checking training integration...")
    
    train_path = Path("src/structured_train.py")
    if not train_path.exists():
        print("❌ Training file not found: src/structured_train.py")
        return False
    
    with open(train_path, 'r') as f:
        train_content = f.read()
    
    # Check for enhanced uncertainty parameter passing
    if 'enhanced_uncertainty_kwargs' in train_content:
        print("  ✅ Enhanced uncertainty parameters are passed to model")
    else:
        print("  ❌ Enhanced uncertainty parameters not found in training")
        return False
    
    # Check for phase training methods
    phase_methods = ['_train_phase1a', '_train_phase1b']
    found_phases = []
    for method in phase_methods:
        if method in train_content:
            found_phases.append(f"  ✅ {method}")
        else:
            print(f"  ❌ {method}: MISSING")
    
    if len(found_phases) == len(phase_methods):
        print("Training phase methods found:")
        for method in found_phases:
            print(method)
        print("\n✅ Training integration complete!")
        return True
    else:
        print(f"\n❌ Only {len(found_phases)}/{len(phase_methods)} phase methods found!")
        return False

def validate_parameter_ranges():
    """Validate that parameter values are in reasonable ranges."""
    print("\n🔍 Validating parameter ranges...")
    
    config_path = Path("src/configs/structured.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    
    # Define expected ranges
    ranges = {
        'beta_target': (0.0, 1.0, "Should be < 1.0 to make small σ² cheap"),
        'beta_other': (1.0, 5.0, "Should be ≥ 1.0 to maintain standard KL"),
        'prior_var_target': (0.01, 1.0, "Should be < 1.0 for target patterns"),
        'prior_var_other': (1.0, 10.0, "Should be > 1.0 for other patterns"),
        'alpha_target': (0.1, 5.0, "Should be positive for target certainty"),
        'alpha_other': (0.1, 5.0, "Should be positive for other uncertainty"),
        'uncertainty_margin': (0.1, 5.0, "Should be positive for variance threshold"),
        'entropy_gamma': (0.0, 1.0, "Should be small positive for entropy bonus"),
        'logvar_min': (-20.0, 0.0, "Should be negative for lower bound"),
        'logvar_max': (0.0, 10.0, "Should be positive for upper bound"),
        'variance_floor_other': (0.1, 5.0, "Should be positive for floor"),
        'phase1a_steps': (0, 1000, "Should be non-negative"),
        'phase1b_lr_scale': (0.01, 1.0, "Should be between 0 and 1 for reduced LR")
    }
    
    all_valid = True
    for param, (min_val, max_val, description) in ranges.items():
        if param in training_config:
            value = training_config[param]
            if min_val <= value <= max_val:
                print(f"  ✅ {param}: {value} (valid)")
            else:
                print(f"  ❌ {param}: {value} (should be between {min_val} and {max_val})")
                print(f"      {description}")
                all_valid = False
        else:
            print(f"  ❌ {param}: MISSING")
            all_valid = False
    
    if all_valid:
        print("\n✅ All parameters are in valid ranges!")
    else:
        print("\n❌ Some parameters are outside valid ranges!")
    
    return all_valid

def check_specialization_ratio():
    """Check if the parameter combination should lead to good specialization."""
    print("\n🔍 Checking specialization potential...")
    
    config_path = Path("src/configs/structured.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    
    # Check key ratios
    prior_var_target = training_config.get('prior_var_target', 0.25)
    prior_var_other = training_config.get('prior_var_other', 2.0)
    beta_target = training_config.get('beta_target', 0.5)
    beta_other = training_config.get('beta_other', 1.5)
    
    # Calculate ratios
    prior_ratio = prior_var_other / prior_var_target
    beta_ratio = beta_other / beta_target
    
    print(f"  Prior variance ratio (other/target): {prior_ratio:.2f}")
    print(f"  Beta ratio (other/target): {beta_ratio:.2f}")
    
    # Check if ratios are good for specialization
    good_prior_ratio = prior_ratio >= 4.0
    good_beta_ratio = beta_ratio >= 2.0
    
    if good_prior_ratio and good_beta_ratio:
        print("  ✅ Good specialization potential!")
        print("    - Prior variance ratio ≥ 4.0 (creates clear variance gap)")
        print("    - Beta ratio ≥ 2.0 (asymmetric KL treatment)")
    else:
        print("  ⚠️  Suboptimal specialization potential:")
        if not good_prior_ratio:
            print(f"    - Prior variance ratio {prior_ratio:.2f} < 4.0 (try increasing prior_var_other or decreasing prior_var_target)")
        if not good_beta_ratio:
            print(f"    - Beta ratio {beta_ratio:.2f} < 2.0 (try increasing beta_other or decreasing beta_target)")
    
    return good_prior_ratio and good_beta_ratio

def main():
    """Run all verification checks."""
    print("🚀 Enhanced Uncertainty Shaping Verification")
    print("=" * 50)
    
    checks = [
        ("Configuration Parameters", check_config_parameters),
        ("Model Integration", check_model_integration),
        ("Training Integration", check_training_integration),
        ("Parameter Ranges", validate_parameter_ranges),
        ("Specialization Potential", check_specialization_ratio),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 All checks passed! Enhanced uncertainty shaping is ready to use.")
        print("\nNext steps:")
        print("1. Run training: python src/structured_train.py --config-name=structured")
        print("2. Monitor metrics in WandB dashboard")
        print("3. Check the VERIFICATION_AND_TUNING_GUIDE.md for detailed tuning instructions")
    else:
        print(f"\n⚠️  {len(results) - passed} checks failed. Please fix the issues above.")
        print("Refer to the implementation guide for details on how to fix these issues.")

if __name__ == "__main__":
    main()
