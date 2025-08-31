# Two-Phase Training Approach for Structured LPN

## **🎯 Problem Statement**

The previous joint learning approach had critical issues:

1. **Joint learning interference**: Encoders trying to work together while also specializing
2. **Decoder interference**: Decoder changing while encoders learn patterns  
3. **Loss conflicts**: Reconstruction loss competing with specialization losses
4. **Poor specialization**: Encoders not learning distinct pattern representations

## **🚀 Solution: Two-Phase Training**

### **Phase 1: Individual Encoder Specialization**
- **Goal**: Train each encoder independently to specialize in specific patterns
- **Method**: Use original decoders from downloaded models to prevent interference
- **Data**: Complementary data subsets for each encoder
- **Loss**: Contrastive loss to reinforce certainty on "their" patterns, decrease on others
- **Duration**: `encoder_expose_steps` gradient steps per encoder

### **Phase 2: Joint Decoder Training**
- **Goal**: Train new decoder to work with specialized encoders
- **Method**: Freeze specialized encoders, train decoder jointly
- **Data**: Balanced pattern distribution for robust decoder training
- **Loss**: Only reconstruction loss (no specialization losses)
- **Duration**: Remaining training steps

## **🔧 Implementation Details**

### **Phase 1: Individual Encoder Specialization**

#### **Data Generation Strategy**
```python
# For each encoder, create specialized training data
target_pattern = enc_idx + 1  # Encoder 0 -> Pattern 1, etc.

# 70% target pattern (reinforced)
target_samples = int(total_samples * 0.7)
# 30% other patterns (reduced certainty)  
other_samples = total_samples - target_samples
```

#### **Individual Training Process**
```python
for enc_idx, enc_params in enumerate(enc_params_list):
    # Create individual model: 1 encoder + original decoder
    individual_model = StructuredLPN(
        encoders=(self.encoders[enc_idx],),
        decoder=self.decoder
    )
    
    # Train on specialized data with contrastive loss
    specialized_encoder = self._train_encoder_individually(
        enc_idx, individual_state, individual_model
    )
```

#### **Contrastive Loss Application**
- **Target pattern**: High confidence (low variance)
- **Other patterns**: Lower confidence (higher variance)
- **No decoder interference**: Original decoder parameters frozen

### **Phase 2: Joint Decoder Training**

#### **Encoder Freezing**
```python
# Phase 2: FREEZE encoders, only train decoder
if "encoders" in grads:
    zeros_enc = tree_map(lambda g: jnp.zeros_like(g), grads["encoders"])
    grads = dict(grads)
    grads["encoders"] = zeros_enc
    logging.debug(f"🔒 Phase 2: Encoder gradients zeroed (frozen)")
```

#### **Loss Configuration**
```python
# Phase 2: NO specialization losses - only reconstruction
repulsion_coeff = 0.0      # DISABLED
contrastive_coeff = 0.0    # DISABLED

# Only reconstruction losses active
prior_kl_coeff = self.cfg.training.get("prior_kl_coeff")
pairwise_kl_coeff = self.cfg.training.get("pairwise_kl_coeff")
```

## **📊 Training Flow**

### **Step-by-Step Process**

1. **Initialization**
   ```python
   # Store original parameters
   self.original_encoder_params = [enc_params.copy() for enc_params in enc_params_list]
   self.original_decoder_params = state.params["decoder"].copy()
   ```

2. **Phase 1: Individual Specialization**
   ```python
   if self.encoder_expose_steps > 0 and not self.phase1_completed:
       state = self._specialize_individual_encoders(state, enc_params_list)
       self.phase1_completed = True
   ```

3. **Phase 2: Joint Training**
   ```python
   if self.phase1_completed:
       # Phase 2: Joint decoder training (encoders frozen)
       state, metrics = self.train_n_steps_phase2(state, batches_with_patterns, train_key)
   else:
       # Phase 1: Individual encoder training
       state, metrics = self.train_n_steps(state, batches_with_patterns, train_key)
   ```

## **⚙️ Configuration**

### **Current Config**
```yaml
training:
  encoder_expose_steps: 200  # Phase 1 duration per encoder
  contrastive_kl: 0.7        # Contrastive loss coefficient
  learning_rate: 0.001       # Learning rate for both phases
  
structured:
  alphas: [0.33, 0.33, 0.34] # Balanced encoder weights
```

### **Recommended Settings**
```yaml
training:
  encoder_expose_steps: 500  # More steps for better specialization
  contrastive_kl: 1.0        # Stronger specialization signal
  batch_size: 126            # Must be divisible by 3
  
structured:
  alphas: [0.33, 0.33, 0.34] # Balanced weights
```

## **📈 Expected Results**

### **Phase 1 Outcomes**
- **Encoder 0**: Specialized in Pattern 1 (O-tetromino) - low variance
- **Encoder 1**: Specialized in Pattern 2 (T-tetromino) - low variance  
- **Encoder 2**: Specialized in Pattern 3 (L-tetromino) - low variance
- **Cross-pattern variance**: Higher variance for non-specialized patterns

### **Phase 2 Outcomes**
- **Frozen encoders**: Maintain specialization
- **Trained decoder**: Learns to work with specialized encoders
- **Joint performance**: Better reconstruction through ensemble
- **No interference**: Clean separation of concerns

## **🔍 Monitoring & Validation**

### **Phase 1 Monitoring**
```python
# Automatic validation every 100 steps
if step % 100 == 0:
    self._validate_contrastive_loss_patterns(explicit_pattern_ids, self.batch_size)

# Encoder variance tracking
Encoder 0 - Pattern 1: mean_var=0.123456, std_var=0.045678, samples=32
Encoder 1 - Pattern 2: mean_var=0.098765, std_var=0.032145, samples=32
Encoder 2 - Pattern 3: mean_var=0.156789, std_var=0.067890, samples=32
```

### **Phase 2 Monitoring**
```python
# Encoder gradients should be zero
🔒 Phase 2: Encoder gradients zeroed (frozen)

# Only decoder training active
Phase 2: Joint decoder training completed - 10 steps
   - Encoders are FROZEN (keeping specialization)
   - Decoder is TRAINABLE (reconstruction focus)
   - No specialization losses applied
```

## **🚨 Troubleshooting**

### **If Phase 1 Fails**
1. **Check encoder exposure steps**: Ensure `encoder_expose_steps > 0`
2. **Verify contrastive coefficient**: Increase `contrastive_kl` if needed
3. **Monitor pattern distribution**: Ensure balanced data generation
4. **Check individual training**: Look for encoder-specific errors

### **If Phase 2 Fails**
1. **Verify Phase 1 completion**: Check `self.phase1_completed` flag
2. **Monitor encoder freezing**: Ensure gradients are zeroed
3. **Check decoder training**: Verify reconstruction loss is active
4. **Validate PoE combination**: Ensure encoders work together

### **If Specialization is Poor**
1. **Increase exposure steps**: More steps for better specialization
2. **Strengthen contrastive loss**: Higher `contrastive_kl` coefficient
3. **Check data quality**: Ensure pattern-specific data generation
4. **Monitor variance outputs**: Verify encoders produce variable variances

## **🎯 Benefits of Two-Phase Approach**

### **✅ Advantages**
1. **Clean separation**: No interference between specialization and joint training
2. **Better specialization**: Individual training allows focused learning
3. **Stable joint training**: Frozen encoders prevent drift
4. **Clearer objectives**: Each phase has single, focused goal
5. **Easier debugging**: Issues can be isolated to specific phases

### **⚠️ Considerations**
1. **Longer total training**: Phase 1 adds to total training time
2. **More complex logic**: Two-phase approach requires careful state management
3. **Memory usage**: Storing original parameters for restoration
4. **Configuration complexity**: More parameters to tune

## **🔮 Future Improvements**

### **Potential Enhancements**
1. **Adaptive exposure**: Dynamic adjustment of encoder exposure steps
2. **Curriculum learning**: Gradually increase pattern difficulty
3. **Multi-stage specialization**: Progressive specialization refinement
4. **Ensemble validation**: Cross-validate specialized encoders before Phase 2

### **Research Directions**
1. **Optimal phase balance**: Find best ratio of Phase 1 vs Phase 2 steps
2. **Pattern complexity**: Study how pattern difficulty affects specialization
3. **Transfer learning**: Apply specialized encoders to new domains
4. **Meta-learning**: Learn optimal specialization strategies

## **📝 Summary**

The two-phase training approach addresses the fundamental issue of joint learning interference by:

1. **First specializing encoders individually** using original decoders
2. **Then training a new decoder jointly** with frozen, specialized encoders

This creates a clear separation of concerns:
- **Phase 1**: Pattern specialization (no decoder interference)
- **Phase 2**: Joint optimization (no encoder drift)

The result should be much better encoder specialization and more effective ensemble performance!

