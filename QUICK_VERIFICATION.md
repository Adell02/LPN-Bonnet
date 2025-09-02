# Quick Verification Guide for Enhanced Uncertainty Shaping

Since the full verification script requires dependencies, here's a quick manual verification guide.

## 🔍 Step 1: Check Configuration

Verify that the enhanced uncertainty shaping parameters are in your config:

```bash
# Check if the parameters exist
grep -A 20 "Enhanced uncertainty shaping parameters" src/configs/structured.yaml
```

You should see:
```yaml
# Enhanced uncertainty shaping parameters
# Subset β for different pattern types
beta_target: 0.5      # β_T < 1 to make small σ² cheap for target patterns
beta_other: 1.5       # β_O ≥ 1 to maintain standard KL for other patterns

# Class-conditional prior variances
prior_var_target: 0.25  # s_T² < 1 for target patterns (σ² ≈ 0.25)
prior_var_other: 2.0    # s_O² > 1 for other patterns (σ² ≈ 2.0)

# Per-sample, per-dimension uncertainty penalties
alpha_target: 1.0     # α_T for driving target certainty: L_T = α_T * Σ_d σ_T²
alpha_other: 0.5      # α_O for inflating other uncertainty: L_O = α_O * Σ_d softplus(m - σ_O²)
uncertainty_margin: 1.0  # m: minimum variance threshold for other patterns

# Entropy bonus on other patterns
entropy_gamma: 0.1    # γ for entropy bonus: -γ * Σ_d log σ_O²

# Variance bounds and calibration
logvar_min: -10.0     # Lower bound for log variance
logvar_max: 5.0       # Upper bound for log variance
variance_floor_other: 0.5  # Floor variance for other patterns

# Training schedule parameters
phase1a_steps: 200    # Phase 1A: encoder-only training steps
phase1b_lr_scale: 0.1 # Phase 1B: reduced learning rate scale
```

## 🔍 Step 2: Check Model Integration

Verify that the enhanced uncertainty shaping method is in the model:

```bash
# Check if the method exists
grep -n "_compute_enhanced_uncertainty_shaping" src/models/structured_lpn.py
```

You should see the method definition around line 629.

## 🔍 Step 3: Check Training Integration

Verify that the enhanced uncertainty parameters are passed to the model:

```bash
# Check if parameters are passed
grep -A 5 -B 5 "enhanced_uncertainty_kwargs" src/structured_train.py
```

You should see the parameter preparation code.

## 🚀 Step 4: Run Training

Start training with the enhanced uncertainty shaping:

```bash
python src/structured_train.py --config-name=structured
```

## 📊 Step 5: Monitor Metrics

Look for these new metrics in your logs or WandB dashboard:

- `enhanced_uncertainty/kl_loss`
- `enhanced_uncertainty/L_target`
- `enhanced_uncertainty/L_other`
- `enhanced_uncertainty/L_entropy_other`
- `enhanced_uncertainty/avg_target_var`
- `enhanced_uncertainty/avg_other_var`
- `enhanced_uncertainty/specialization_ratio`

## ✅ Success Criteria

Your implementation is working if you see:

1. **Clear variance separation:**
   - `avg_target_var` < 0.5
   - `avg_other_var` > 1.5
   - `specialization_ratio` > 3.0

2. **Stable training:**
   - No NaN/inf values
   - Loss decreases smoothly
   - Metrics stabilize after ~50 steps

3. **Phase training (if enabled):**
   - Log messages about Phase 1A and Phase 1B
   - Different learning rates in different phases

## 🎛️ Quick Tuning

If the metrics don't look right, try these quick adjustments:

### Target variance too high?
```bash
python src/structured_train.py \
    --config-name=structured \
    training.beta_target=0.3 \
    training.alpha_target=1.5 \
    training.prior_var_target=0.1
```

### Other variance too low?
```bash
python src/structured_train.py \
    --config-name=structured \
    training.beta_other=2.0 \
    training.alpha_other=0.8 \
    training.entropy_gamma=0.2
```

### No specialization?
```bash
python src/structured_train.py \
    --config-name=structured \
    training.phase1a_steps=500 \
    training.beta_target=0.1 \
    training.alpha_target=3.0
```

## 🚨 Troubleshooting

### No enhanced uncertainty metrics?
- Check that the parameters are in the config
- Verify the model integration
- Look for error messages in the logs

### Training crashes?
- Check for NaN/inf values in metrics
- Reduce penalty strengths (alpha_target, alpha_other)
- Increase phase1b_lr_scale to 0.2 or 0.5

### Poor specialization?
- Increase phase1a_steps to 300-500
- Decrease beta_target to 0.1-0.3
- Increase alpha_target to 2.0-3.0

## 📈 Expected Timeline

- **Steps 0-50**: Metrics should stabilize
- **Steps 50-200**: Clear variance separation should emerge
- **Steps 200+**: Specialization should be well-established

If you don't see clear separation by step 200, adjust the parameters using the tuning guide above.
