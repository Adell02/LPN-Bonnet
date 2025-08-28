"""
Example usages:

python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i mean \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 20 \
    --lr 5e-2 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/playful-monkey-758--checkpoint:v1 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0

    
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/solar-salad-1050--checkpoint:v18 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 1.0 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}' \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/solar-salad-1050--checkpoint:v18 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 1.0 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}'





python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/playful-sun-1060--checkpoint:v1 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 0.3 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}' \
    --accumulate-gradients-decoder-pairs true \
    --random-perturbation '{"num_samples": 5, "scale": 0.1}' \
    --include-all-latents true
    

    

python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/ominous-monster-839--checkpoint:v2 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 10 \
    -i gradient_ascent \
    --num-steps 10 \
    --lr 0.1 \
    --random-perturbation '{"num_samples": 5, "scale": 0.1}' \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 0.5 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 1.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 1.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 5.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 5.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 10.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 10.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 50.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 50.0    

    
# Evaluate on the ARC json datasets (only -w, -jc, and -js are required):
## Random Search
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 1 \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0
## Gradient Ascent
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 1 \
    -i gradient_ascent \
    --num-steps 5 \
    --lr 5e-2

# Evaluate on a custom dataset (only -w and -d are required):
## Random Search
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -d storage/v0_main_fix_test \
    --dataset-length 32 \
    --dataset-batch-size 8 \
    --dataset-seed 0 \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0
## Gradient Ascent
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -d storage/v0_main_fix_test \
    --dataset-length 32 \
    --dataset-batch-size 8 \
    --dataset-seed 0 \
    -i gradient_ascent \
    --num-steps 5 \
    --lr 5e-2
"""

import argparse
import os
from typing import Optional

import chex
import wandb
import hydra
import omegaconf
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import json
import optax
from tqdm import trange
from flax.training.train_state import TrainState
from flax.serialization import from_bytes

from  models.lpn import LPN
from  evaluator import Evaluator
from  models.transformer import EncoderTransformer, DecoderTransformer
from  train import Trainer, load_datasets, instantiate_config_for_mpt
from  data_utils import make_leave_one_out, DATASETS_BASE_PATH


def instantiate_model(cfg: omegaconf.DictConfig, mixed_precision: bool, latent_dim_override: Optional[int] = None) -> LPN:
    if mixed_precision:
        encoder_config = instantiate_config_for_mpt(cfg.encoder_transformer)
        decoder_config = instantiate_config_for_mpt(cfg.decoder_transformer)
    else:
        encoder_config = hydra.utils.instantiate(cfg.encoder_transformer)
        decoder_config = hydra.utils.instantiate(cfg.decoder_transformer)
    
    # Override latent dimension if specified
    if latent_dim_override is not None:
        print(f"   🔧 Overriding latent dimension from {encoder_config.latent_dim} to {latent_dim_override}")
        encoder_config = encoder_config.replace(latent_dim=latent_dim_override)
    
    encoder = EncoderTransformer(encoder_config)
    decoder = DecoderTransformer(decoder_config)
    lpn = LPN(encoder=encoder, decoder=decoder)
    return lpn


def instantiate_train_state(lpn: LPN) -> TrainState:
    """Create a proper train state that can be loaded from checkpoint."""
    # This was the original working function - we need to restore it
    print(f"   🔍 instantiate_train_state: Starting model initialization...")
    print(f"   🔍 instantiate_train_state: lpn.encoder.config.latent_dim = {lpn.encoder.config.latent_dim}")
    print(f"   🔍 instantiate_train_state: lpn.decoder.config = {lpn.decoder.config}")
    
    key = jax.random.PRNGKey(0)
    decoder = lpn.decoder
    grids = jax.random.randint(
        key,
        (1, 3, decoder.config.max_rows, decoder.config.max_cols, 2),
        minval=0,
        maxval=decoder.config.vocab_size,
    )
    shapes = jax.random.randint(
        key,
        (1, 3, 2, 2),
        minval=1,
        maxval=min(decoder.config.max_rows, decoder.config.max_cols) + 1,
    )
    
    print(f"   🔍 instantiate_train_state: About to call lpn.init with:")
    print(f"   🔍 instantiate_train_state: - grids shape: {grids.shape}")
    print(f"   🔍 instantiate_train_state: - shapes shape: {shapes.shape}")
    print(f"   🔍 instantiate_train_state: - lpn.encoder.config.latent_dim: {lpn.encoder.config.latent_dim}")
    
    try:
        variables = lpn.init(
            key, grids, shapes, dropout_eval=False, prior_kl_coeff=0.0, pairwise_kl_coeff=0.0, mode="mean"
        )
        print(f"   ✅ instantiate_train_state: lpn.init completed successfully")
        print(f"   🔍 instantiate_train_state: variables keys: {list(variables.keys())}")
        print(f"   🔍 instantiate_train_state: params keys: {list(variables['params'].keys())}")
        
        # Check the encoder parameters specifically
        if 'encoder' in variables['params']:
            encoder_params = variables['params']['encoder']
            print(f"   🔍 instantiate_train_state: encoder params keys: {list(encoder_params.keys())}")
            if 'Dense_0' in encoder_params:
                dense_params = encoder_params['Dense_0']
                print(f"   🔍 instantiate_train_state: Dense_0 params keys: {list(dense_params.keys())}")
                if 'kernel' in dense_params:
                    kernel_shape = dense_params['kernel'].shape
                    print(f"   🔍 instantiate_train_state: Dense_0 kernel shape: {kernel_shape}")
        
    except Exception as e:
        print(f"   ❌ instantiate_train_state: lpn.init failed with error: {e}")
        print(f"   🔍 instantiate_train_state: Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

    learning_rate, linear_warmup_steps = 0, 0
    linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
        init_value=learning_rate / (linear_warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=linear_warmup_steps,
        transition_steps=1,
        end_value=learning_rate,
        decay_rate=1.0,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler))
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
    train_state = TrainState.create(apply_fn=lpn.apply, tx=optimizer, params=variables["params"])
    print(f"   ✅ instantiate_train_state: TrainState created successfully")
    return train_state


def load_model_weights(
    train_state: TrainState, artifact_dir: str, ckpt_name: str = "state.msgpack"
) -> TrainState:
    print(f"   🔍 load_model_weights: Loading checkpoint from {os.path.join(artifact_dir, ckpt_name)}")
    
    # First, let's inspect what's in the checkpoint file
    checkpoint_path = os.path.join(artifact_dir, ckpt_name)
    if os.path.exists(checkpoint_path):
        print(f"   🔍 load_model_weights: Checkpoint file exists, size: {os.path.getsize(checkpoint_path)} bytes")
        
        # Try to peek at the checkpoint structure without loading
        try:
            with open(checkpoint_path, "rb") as data_file:
                byte_data = data_file.read()
            
            print(f"   🔍 load_model_weights: About to inspect checkpoint structure...")
            
            # Create a minimal state structure to inspect the checkpoint
            from flax.training.train_state import TrainState
            import optax
            
            # Create a dummy model with the same structure but empty params
            dummy_lpn = train_state.apply_fn
            dummy_optimizer = optax.sgd(0.0)
            dummy_state = TrainState.create(
                apply_fn=dummy_lpn,
                tx=dummy_optimizer,
                params={}  # Empty params
            )
            
            # Try to load just the structure
            try:
                loaded_dummy = from_bytes(dummy_state, byte_data)
                print(f"   🔍 load_model_weights: Successfully loaded checkpoint structure")
                
                # Inspect the loaded structure
                if hasattr(loaded_dummy, 'params') and loaded_dummy.params:
                    print(f"   🔍 load_model_weights: Checkpoint contains params with keys: {list(loaded_dummy.params.keys())}")
                    if 'encoder' in loaded_dummy.params:
                        encoder_params = loaded_dummy.params['encoder']
                        print(f"   🔍 load_model_weights: Encoder params keys: {list(encoder_params.keys())}")
                        if 'Dense_0' in encoder_params:
                            dense_params = encoder_params['Dense_0']
                            if 'kernel' in dense_params:
                                kernel_shape = dense_params['kernel'].shape
                                print(f"   🔍 load_model_weights: Checkpoint Dense_0 kernel shape: {kernel_shape}")
                                print(f"   🔍 load_model_weights: Expected Dense_0 kernel shape: (96, 2)")
                                if kernel_shape != (96, 2):
                                    print(f"   ⚠️  SHAPE MISMATCH: Checkpoint has {kernel_shape}, but model expects (96, 2)")
                                    print(f"   ⚠️  This checkpoint was trained with {kernel_shape[1]}-dimensional latents!")
                                    print(f"   ⚠️  SOLUTION: Use a checkpoint with 2D latents or modify model to accept {kernel_shape[1]}D latents")
                            else:
                                print(f"   🔍 load_model_weights: Dense_0 has no kernel parameter")
                        else:
                            print(f"   🔍 load_model_weights: No Dense_0 in encoder params")
                    else:
                        print(f"   🔍 load_model_weights: No encoder in params")
                else:
                    print(f"   🔍 load_model_weights: No params found in checkpoint")
                    
            except Exception as inspect_e:
                print(f"   ❌ load_model_weights: Failed to inspect checkpoint structure: {inspect_e}")
                print(f"   🔍 load_model_weights: This suggests the checkpoint format is incompatible")
            
        except Exception as e:
            print(f"   ❌ load_model_weights: Failed to read checkpoint file: {e}")
    
    # Now load into the actual train state
    with open(checkpoint_path, "rb") as data_file:
        byte_data = data_file.read()
    
    print(f"   🔍 load_model_weights: Loading checkpoint into actual train state...")
    loaded_state = from_bytes(train_state, byte_data)
    print(f"   ✅ load_model_weights: Checkpoint loaded successfully")
    return loaded_state


def build_generate_output_batch_to_be_pmapped(
    model: LPN, eval_inference_mode: str, eval_inference_mode_kwargs: dict, return_info: bool = False
) -> callable:
    def generate_output_batch_to_be_pmapped(
        params, leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes, keys
    ) -> dict[str, chex.Array]:
        grids_inputs, labels_grids_outputs = dataset_grids[..., 0], dataset_grids[..., 1]
        shapes_inputs, labels_shapes_outputs = dataset_shapes[..., 0], dataset_shapes[..., 1]
        generated_grids_outputs, generated_shapes_outputs, info = model.apply(
            {"params": params},
            leave_one_out_grids,
            leave_one_out_shapes,
            grids_inputs,
            shapes_inputs,
            keys,
            dropout_eval=True,
            mode=eval_inference_mode,
            **eval_inference_mode_kwargs,
            method=model.generate_output,
        )

        correct_shapes = jnp.all(generated_shapes_outputs == labels_shapes_outputs, axis=-1)
        batch_ndims = len(grids_inputs.shape[:-2])

        row_arange_broadcast = jnp.arange(grids_inputs.shape[-2]).reshape(
            (*batch_ndims * (1,), grids_inputs.shape[-2])
        )
        input_row_mask = row_arange_broadcast < labels_shapes_outputs[..., :1]
        col_arange_broadcast = jnp.arange(grids_inputs.shape[-1]).reshape(
            (*batch_ndims * (1,), grids_inputs.shape[-1])
        )
        input_col_mask = col_arange_broadcast < labels_shapes_outputs[..., 1:]
        input_mask = input_row_mask[..., None] & input_col_mask[..., None, :]

        pixels_equal = jnp.where(
            input_mask & correct_shapes[..., None, None],
            (generated_grids_outputs == labels_grids_outputs),
            False,
        )
        pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / labels_shapes_outputs.prod(axis=-1)
        accuracy = pixels_equal.sum(axis=(-1, -2)) == labels_shapes_outputs.prod(axis=-1)

        metrics = {
            "correct_shapes": jnp.mean(correct_shapes),
            "pixel_correctness": jnp.mean(pixel_correctness),
            "accuracy": jnp.mean(accuracy),
        }
        
        # Extract loss values from gradient ascent process if available
        if eval_inference_mode == "gradient_ascent" and info is not None:
            try:
                # For gradient ascent, extract loss from optimization trajectory
                if hasattr(info, 'get') and callable(info.get):
                    if "optimization_trajectory" in info:
                        trajectory = info["optimization_trajectory"]
                        if isinstance(trajectory, dict):
                            # Try to extract final loss from trajectory
                            if "log_probs" in trajectory:
                                log_probs = trajectory["log_probs"]
                                if isinstance(log_probs, jnp.ndarray):
                                    # Get the final step log probabilities (best across candidates)
                                    if log_probs.ndim >= 2:
                                        # Take the last step and best candidate
                                        final_step_log_probs = log_probs[..., -1, :]  # Last step
                                        best_final_log_probs = jnp.max(final_step_log_probs, axis=-1)  # Best candidate
                                        final_losses = -best_final_log_probs  # Convert to positive loss
                                        metrics["total_final_loss"] = jnp.mean(final_losses)
                                    else:
                                        metrics["total_final_loss"] = None
                                else:
                                    metrics["total_final_loss"] = None
                            elif "losses" in trajectory:
                                # Direct access to losses if available
                                losses = trajectory["losses"]
                                if isinstance(losses, jnp.ndarray):
                                    # Get final loss (last step)
                                    if losses.ndim >= 1:
                                        final_losses = losses[..., -1]  # Last step
                                        metrics["total_final_loss"] = jnp.mean(final_losses)
                                    else:
                                        metrics["total_final_loss"] = None
                                else:
                                    metrics["total_final_loss"] = None
                            else:
                                metrics["total_final_loss"] = None
                        else:
                            metrics["total_final_loss"] = None
                    else:
                        metrics["total_final_loss"] = None
                else:
                    metrics["total_final_loss"] = None
            except Exception as e:
                # If any error occurs during loss extraction, set to None
                metrics["total_final_loss"] = None
        else:
            metrics["total_final_loss"] = None
        
        if return_info:
            return {"metrics": metrics, "info": info}
        return metrics

    return generate_output_batch_to_be_pmapped


def evaluate_json(
    train_state: TrainState,
    evaluator: Evaluator,
    json_challenges_file: str,
    json_solutions_file: str,
    only_n_tasks: Optional[int],
    random_search_seed: int,
) -> dict:
    print(f"Evaluating the model on {json_challenges_file.rstrip().split('/')[-1]}...")
    metrics, fig = Trainer.test_json_submission(
        train_state,
        evaluator,
        json_challenges_file=os.path.join(DATASETS_BASE_PATH, json_challenges_file),
        json_solutions_file=os.path.join(DATASETS_BASE_PATH, json_solutions_file),
        test_name="",
        key=jax.random.PRNGKey(random_search_seed),
        only_n_tasks=only_n_tasks,  # 'None' to run on all tasks
        progress_bar=True,
        num_tasks_to_show=0,
    )
    metrics = {k.split("/")[-1]: v for k, v in metrics.items()}
    metrics["fig"] = fig
    return metrics


def evaluate_custom_dataset(
    train_state: TrainState,
    evaluator: Evaluator,
    dataset_folder: str,
    dataset_length: Optional[int],
    dataset_batch_size: int,
    dataset_use_hf: bool,
    dataset_seed: int,
    random_search_seed: int,
) -> dict:
    print(f"Evaluating the model on the {dataset_folder.rstrip().split('/')[-1]} dataset...")

    # Load data
    grids, shapes, _ = load_datasets([dataset_folder], use_hf=dataset_use_hf)[0]
    if dataset_length is not None:
        key = jax.random.PRNGKey(dataset_seed)
        indices = jax.random.permutation(key, len(grids))[:dataset_length]
        grids, shapes = grids[indices], shapes[indices]
    # Drop the last batch if it's smaller than the batch size
    num_batches = grids.shape[0] // dataset_batch_size
    grids, shapes = grids[: num_batches * dataset_batch_size], shapes[: num_batches * dataset_batch_size]

    leave_one_out_grids = make_leave_one_out(grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(shapes, axis=-3)

    num_devices = len(evaluator.devices)
    # Split the dataset onto devices.
    assert grids.shape[0] % num_devices == 0
    leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
        lambda x: x.reshape((num_devices, x.shape[0] // num_devices, *x.shape[1:])),
        (leave_one_out_grids, leave_one_out_shapes, grids, shapes),
    )
    # Split the dataset into batches for each device.
    if dataset_batch_size is None:
        dataset_batch_size = grids.shape[0]
    batch_size_per_device = dataset_batch_size // num_devices
    assert grids.shape[1] % batch_size_per_device == 0
    leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
        lambda x: x.reshape(
            (x.shape[0], x.shape[1] // batch_size_per_device, batch_size_per_device, *x.shape[2:])
        ),
        (leave_one_out_grids, leave_one_out_shapes, grids, shapes),
    )
    keys = jax.random.split(
        jax.random.PRNGKey(random_search_seed), (num_devices, grids.shape[1])
    )  # (num_devices, num_batches)

    pmap_dataset_generate_output_batch = jax.pmap(
        build_generate_output_batch_to_be_pmapped(
            model=evaluator.model,
            eval_inference_mode=evaluator.inference_mode,
            eval_inference_mode_kwargs=evaluator.inference_mode_kwargs,
        ),
        axis_name="num_devices",
    )
    metrics_list = [
        pmap_dataset_generate_output_batch(
            train_state.params,
            leave_one_out_grids[:, i],
            leave_one_out_shapes[:, i],
            grids[:, i],
            shapes[:, i],
            keys[:, i],
        )
        for i in trange(grids.shape[1], desc="Generating solutions")
    ]
    # Aggregate the metrics over the devices and the batches (robust to None values)
    metrics = {}
    for k in metrics_list[0].keys():
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if len(vals) == 0:
            # No valid values; set to NaN to keep key presence without crashing
            metrics[k] = jnp.nan
        else:
            metrics[k] = jnp.stack(vals).mean()

    # If storing latents, also collect per-sample metrics across the whole evaluated dataset
    per_sample_shape_acc = None
    per_sample_pixel_corr = None
    per_sample_acc = None

    try:
        store_path_flag = evaluator.inference_mode_kwargs.get("store_latents_path", None)
    except Exception:
        store_path_flag = None

    if store_path_flag:
        # Build a pmap that returns per-sample metrics (no averaging)
        def build_generate_output_batch_per_sample(model: LPN, eval_inference_mode: str, eval_inference_mode_kwargs: dict) -> callable:
            def generate_output_batch_per_sample(params, leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes, keys) -> dict[str, chex.Array]:
                grids_inputs, labels_grids_outputs = dataset_grids[..., 0], dataset_grids[..., 1]
                shapes_inputs, labels_shapes_outputs = dataset_shapes[..., 0], dataset_shapes[..., 1]
                generated_grids_outputs, generated_shapes_outputs, _ = model.apply(
                    {"params": params},
                    leave_one_out_grids,
                    leave_one_out_shapes,
                    grids_inputs,
                    shapes_inputs,
                    keys,
                    dropout_eval=True,
                    mode=eval_inference_mode,
                    **eval_inference_mode_kwargs,
                    method=model.generate_output,
                )
                correct_shapes = jnp.all(generated_shapes_outputs == labels_shapes_outputs, axis=-1)
                batch_ndims = len(grids_inputs.shape[:-2])
                row_arange_broadcast = jnp.arange(grids_inputs.shape[-2]).reshape((*batch_ndims * (1,), grids_inputs.shape[-2]))
                input_row_mask = row_arange_broadcast < labels_shapes_outputs[..., :1]
                col_arange_broadcast = jnp.arange(grids_inputs.shape[-1]).reshape((*batch_ndims * (1,), grids_inputs.shape[-1]))
                input_col_mask = col_arange_broadcast < labels_shapes_outputs[..., 1:]
                input_mask = input_row_mask[..., None] & input_col_mask[..., None, :]
                pixels_equal = jnp.where(
                    input_mask & correct_shapes[..., None, None],
                    (generated_grids_outputs == labels_grids_outputs),
                    False,
                )
                pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / labels_shapes_outputs.prod(axis=-1)
                accuracy = pixels_equal.sum(axis=(-1, -2)) == labels_shapes_outputs.prod(axis=-1)
                return {
                    "per_sample_shape_accuracy": correct_shapes.astype(jnp.float32),
                    "per_sample_pixel_correctness": pixel_correctness.astype(jnp.float32),
                    "per_sample_accuracy": accuracy.astype(jnp.float32),
                }
            return generate_output_batch_per_sample

        pmap_dataset_generate_output_batch_per_sample = jax.pmap(
            build_generate_output_batch_per_sample(
                model=evaluator.model,
                eval_inference_mode=evaluator.inference_mode,
                eval_inference_mode_kwargs=evaluator.inference_mode_kwargs,
            ),
            axis_name="num_devices",
        )

        # Collect per-sample metrics across batches
        per_shape_list = []
        per_pixel_list = []
        per_acc_list = []
        for i in trange(grids.shape[1], desc="Collecting per-sample metrics"):
            out = pmap_dataset_generate_output_batch_per_sample(
                train_state.params,
                leave_one_out_grids[:, i],
                leave_one_out_shapes[:, i],
                grids[:, i],
                shapes[:, i],
                keys[:, i],
            )
            host = jax.device_get(out)
            per_shape_list.append(host["per_sample_shape_accuracy"].reshape(-1))
            per_pixel_list.append(host["per_sample_pixel_correctness"].reshape(-1))
            per_acc_list.append(host["per_sample_accuracy"].reshape(-1))

        try:
            import numpy as np
            per_sample_shape_acc = np.concatenate(per_shape_list, axis=0)
            per_sample_pixel_corr = np.concatenate(per_pixel_list, axis=0)
            per_sample_acc = np.concatenate(per_acc_list, axis=0)
            print(f"[per-sample] shape_acc: {per_sample_shape_acc.shape}, pixel: {per_sample_pixel_corr.shape}, acc: {per_sample_acc.shape}")
        except Exception as _ps_e:
            print(f"[per-sample] Failed to assemble per-sample arrays: {_ps_e!r}")

    # Optionally store latents/trajectory for a single representative batch if requested via kwargs
    store_path = None
    try:
        store_path = evaluator.inference_mode_kwargs.get("store_latents_path", None)
    except Exception:
        store_path = None
    if store_path:
        try:
            import numpy as np
            # Collect per-sample loss curves for GA/ES over all batches when available
            ga_losses_rows = []
            es_losses_rows = []
            ga_steps_len = None
            es_gen_len = None
            es_pop_size = evaluator.inference_mode_kwargs.get("population_size", None)

            # Build a variant that returns info
            pmap_with_info = jax.pmap(
                build_generate_output_batch_to_be_pmapped(
                    model=evaluator.model,
                    eval_inference_mode=evaluator.inference_mode,
                    eval_inference_mode_kwargs=evaluator.inference_mode_kwargs,
                    return_info=True,
                ),
                axis_name="num_devices",
            )
            # Iterate all batches to collect per-sample losses
            per_batch_infos = []
            for i in trange(grids.shape[1], desc="Collecting loss trajectories"):
                result_with_info = pmap_with_info(
                    train_state.params,
                    leave_one_out_grids[:, i],
                    leave_one_out_shapes[:, i],
                    grids[:, i],
                    shapes[:, i],
                    keys[:, i],
                )
                per_batch_infos.append(result_with_info)
            # Use the first batch on each device for the representative payload as before
            result_with_info = per_batch_infos[0]
            # Collect info from first device across the pytree
            info_tree = result_with_info.get("info", None)
            if info_tree is None:
                print("[store_latents] No 'info' key returned from pmapped function.")
                info0 = None
            else:
                def _first_device(x):
                    try:
                        return x[0]
                    except Exception:
                        return x
                info0 = jax.tree_util.tree_map(_first_device, info_tree)
                # Bring to host (avoid device-backed arrays in payload)
                try:
                    info0 = jax.device_get(info0)
                except Exception:
                    pass
            print(f"[store_latents] info0 type: {type(info0)}")
            if isinstance(info0, dict):
                print(f"[store_latents] info0 keys: {list(info0.keys())}")
            payload = {}

            # GA trajectory -> save as ga_latents, ga_log_probs, derive ga_path, ga_scores/ga_losses
            if isinstance(info0, dict) and "optimization_trajectory" in info0 and info0["optimization_trajectory"]:
                traj = info0["optimization_trajectory"]
                try:
                    traj = jax.device_get(traj)
                except Exception:
                    pass
                if isinstance(traj, dict):
                    print(f"[store_latents] GA traj keys: {list(traj.keys())}")
                    if "latents" in traj:
                        ga_lat = np.array(traj["latents"])  # (*B, steps, C, H)
                        print(f"[store_latents] ga_latents shape: {getattr(ga_lat, 'shape', None)}")
                        payload["ga_latents"] = ga_lat
                    if "log_probs" in traj:
                        ga_lp = np.array(traj["log_probs"])  # could be (*B, N, steps?, C) or similar
                        print(f"[store_latents] ga_log_probs shape: {getattr(ga_lp, 'shape', None)}")
                        payload["ga_log_probs"] = ga_lp
                        # Derive a 2D GA path and scores robustly for batch 0, pair 0
                        try:
                            lat = ga_lat
                            lp = ga_lp
                            # Bring to canonical shape: (N, S, C, H) and (N, S, C)
                            # Remove leading batch axis if present
                            if lat.ndim >= 5:
                                # (B, N, S_or_1, C, H)
                                lat0 = lat[0]
                            else:
                                lat0 = lat
                            if lp.ndim >= 4:
                                lp0 = lp[0]
                            else:
                                lp0 = lp
                            # Ensure we have explicit dims
                            # Identify time dimension: if lat0.shape[-3] > 1 use that as steps; else use candidate axis as time
                            Ndim = lat0.shape[0]
                            steps_dim = lat0.shape[-3]
                            cand_dim = lat0.shape[-2]
                            # pick first pair
                            lat_pair = lat0[0]  # (S, C, H) or (1, C, H)
                            lp_pair = lp0[0]    # (S, C) or (1, C)
                            if steps_dim > 1 and lat_pair.ndim == 3:
                                # Standard case: time along axis 0, candidates along axis 1
                                idx = np.argmax(lp_pair, axis=-1)               # (S,)
                                best_path = lat_pair[np.arange(lat_pair.shape[0]), idx]  # (S, H)
                                best_scores = np.max(lp_pair, axis=-1)         # (S,)
                            else:
                                # Steps collapsed to 1 and candidates represent iterations
                                # Use candidates as time
                                # transpose (1, C, H) -> (C, H)
                                best_path = lat_pair.reshape(-1, lat_pair.shape[-1])  # (C, H)
                                best_scores = np.max(lp_pair, axis=0).reshape(-1) if lp_pair.ndim == 2 else lp_pair.reshape(-1)
                            if best_path.shape[-1] == 2:
                                payload["ga_path"] = best_path
                            payload["ga_scores"] = best_scores
                            payload["ga_losses"] = -best_scores
                        except Exception as _pe:
                            print(f"[store_latents] GA path/score derivation failed: {_pe!r}")

            # ES trajectory -> save best_latents_per_generation, per-generation losses, and populations
            if isinstance(info0, dict) and "evolutionary_trajectory" in info0 and info0["evolutionary_trajectory"]:
                traj = info0["evolutionary_trajectory"]
                try:
                    traj = jax.device_get(traj)
                except Exception:
                    pass
                if isinstance(traj, dict):
                    print(f"[store_latents] ES traj keys: {list(traj.keys())}")
                    if "best_latents_per_generation" in traj and traj["best_latents_per_generation"] is not None:
                        es_lat = np.array(traj["best_latents_per_generation"])  # (*B, G, H)
                        print(f"[store_latents] es_best_latents_per_generation shape: {getattr(es_lat, 'shape', None)}")
                        payload["es_best_latents_per_generation"] = es_lat
                    # Handle generation metrics: prefer positive losses
                    if "losses_per_generation" in traj:
                        es_gen_losses = np.array(traj["losses_per_generation"])  # (*B, G) or (G,)
                        payload["es_generation_losses"] = es_gen_losses
                        try:
                            payload["es_best_losses_per_generation"] = es_gen_losses.reshape(-1)
                        except Exception:
                            pass
                    elif "final_best_loss" in traj or "generation_fitness" in traj:
                        # Backward compatibility: if only fitness provided, convert to losses
                        if "generation_fitness" in traj:
                            gen_fit = np.array(traj["generation_fitness"])  # fitness = -loss
                            es_gen_losses = -gen_fit
                            payload["es_generation_losses"] = es_gen_losses
                            try:
                                payload["es_best_losses_per_generation"] = es_gen_losses.reshape(-1)
                            except Exception:
                                pass
                        if "final_best_loss" in traj:
                            payload["es_final_best_loss"] = np.array(traj["final_best_loss"])  # positive
                    elif "generation_accuracies" in traj:  # Fallback for old key names
                        es_gen_acc = np.array(traj["generation_accuracies"])  # (G, *B) or (*B, G)
                        payload["es_generation_accuracies"] = es_gen_acc
                        try:
                            payload["es_best_scores_per_generation"] = es_gen_acc.reshape(-1)
                        except Exception:
                            pass
                    
                    # Handle population losses per generation
                    if "losses_per_generation" in traj:
                        es_pop_losses = np.array(traj["losses_per_generation"])  # (*B, G, C)
                        payload["es_losses_per_generation"] = es_pop_losses
                        # Flatten for background heatmap
                        payload["es_all_losses"] = es_pop_losses.reshape(-1)
                    
                    # Optional: populations per generation for plotting cloud
                    if "populations_per_generation" in traj:
                        pop = np.array(traj["populations_per_generation"])  # (*B, G, C, H)
                        payload["es_all_latents"] = pop.reshape(-1, pop.shape[-1])
                        # generation indices for coloring
                        G = pop.shape[-3]
                        payload["es_generation_idx"] = np.repeat(np.arange(G), pop.shape[-2])
                    
                    # Handle fitness scores (fallback for old key names)
                    if "fitness_per_generation" in traj:
                        fit = np.array(traj["fitness_per_generation"])      # (*B, G, C)
                        payload["es_all_scores"] = fit.reshape(-1)
                    
                    # Handle final best metrics
                    if "final_best_loss" in traj:
                        payload["es_final_best_loss"] = np.array(traj["final_best_loss"])  # positive
                    if "final_best_fitness" in traj:
                        payload["es_final_best_fitness"] = np.array(traj["final_best_fitness"])  # negative of loss
                    elif "final_best_accuracy" in traj:  # Fallback for old key names
                        payload["es_final_best_accuracy"] = np.array(traj["final_best_accuracy"])    

            # Save compressed
            if not payload:
                payload["note"] = np.array(["no_trajectory_available"], dtype=object)
            # Attach per-sample metrics if available
            if per_sample_shape_acc is not None:
                payload["per_sample_shape_accuracy"] = np.asarray(per_sample_shape_acc)
            if per_sample_pixel_corr is not None:
                payload["per_sample_pixel_correctness"] = np.asarray(per_sample_pixel_corr)
            if per_sample_acc is not None:
                payload["per_sample_accuracy"] = np.asarray(per_sample_acc)

            # Build per-sample loss curves from all batches when possible
            try:
                for result_with_info in per_batch_infos:
                    info_tree = result_with_info.get("info", None)
                    if info_tree is None:
                        continue
                    def _first_device(x):
                        try:
                            return x[0]
                        except Exception:
                            return x
                    info0_batch = jax.tree_util.tree_map(_first_device, info_tree)
                    try:
                        info0_batch = jax.device_get(info0_batch)
                    except Exception:
                        pass
                    # GA per-sample losses per step (reduce over candidate axis, keep steps axis)
                    if isinstance(info0_batch, dict) and "optimization_trajectory" in info0_batch and info0_batch["optimization_trajectory"]:
                        t = info0_batch["optimization_trajectory"]
                        if isinstance(t, dict) and "log_probs" in t:
                            lp = np.array(t["log_probs"])  # expected shape like (B, ..., steps and candidates axes)
                            # Try to infer steps axis using latents when available
                            steps_from_latents = None
                            try:
                                if "latents" in t:
                                    lat_arr = np.array(t["latents"])  # (..., steps, H) or (B, steps, C, H)
                                    if lat_arr.ndim >= 3:
                                        steps_from_latents = lat_arr.shape[-2]
                            except Exception:
                                steps_from_latents = None

                            # Squeeze any leading batch/device axes except keep one batch axis if present
                            while lp.ndim > 3:
                                lp = lp[0]
                            if lp.ndim == 2:
                                # (X, Y) -> treat as (1, X, Y)
                                lp = lp[None, ...]

                            # At this point lp is (B, A, B2). One of A or B2 is steps, the other is candidates.
                            # Identify steps axis robustly, preferring match with latents-derived steps length.
                            steps_axis = None
                            cand_axis = None
                            if steps_from_latents is not None:
                                if lp.shape[-1] == steps_from_latents:
                                    steps_axis, cand_axis = -1, -2
                                elif lp.shape[-2] == steps_from_latents:
                                    steps_axis, cand_axis = -2, -1
                            # Fallback: if one axis equals 1, that is candidates
                            if steps_axis is None:
                                if lp.shape[-2] == 1 and lp.shape[-1] > 1:
                                    steps_axis, cand_axis = -1, -2
                                elif lp.shape[-1] == 1 and lp.shape[-2] > 1:
                                    steps_axis, cand_axis = -2, -1
                            # Last fallback: choose larger axis as steps
                            if steps_axis is None:
                                if lp.shape[-2] >= lp.shape[-1]:
                                    steps_axis, cand_axis = -2, -1
                                else:
                                    steps_axis, cand_axis = -1, -2

                            # If steps are on the last axis, reduce over the other; else transpose to (B, steps, cand)
                            if steps_axis == -1:
                                # lp: (B, cand, steps) assumed -> transpose to (B, steps, cand)
                                lp_bc = np.swapaxes(lp, -1, -2)
                            elif steps_axis == -2:
                                # lp: (B, steps, cand) already
                                lp_bc = lp
                            else:
                                lp_bc = lp

                            # Reduce over candidates (last axis) to best score per step
                            ga_scores_bs = lp_bc.max(axis=-1)  # (B, steps)
                            ga_losses_bs = -ga_scores_bs
                            if ga_steps_len is None:
                                ga_steps_len = ga_losses_bs.shape[-1]
                            ga_losses_rows.extend([row.reshape(-1) for row in ga_losses_bs])
                    # ES per-sample losses per generation
                    if isinstance(info0_batch, dict) and "evolutionary_trajectory" in info0_batch and info0_batch["evolutionary_trajectory"]:
                        t = info0_batch["evolutionary_trajectory"]
                        if isinstance(t, dict):
                            if "losses_per_generation" in t:
                                L = np.array(t["losses_per_generation"])  # (B, G) or (B, G, C)
                                if L.ndim == 3:
                                    # Reduce across population (min loss per generation)
                                    L = L.min(axis=-1)
                                if L.ndim == 1:
                                    L = L[None, ...]
                                if es_gen_len is None:
                                    es_gen_len = L.shape[-1]
                                es_losses_rows.extend([row.reshape(-1) for row in L])
                            elif "generation_fitness" in t:
                                F = np.array(t["generation_fitness"])  # (B, G)
                                if F.ndim == 1:
                                    F = F[None, ...]
                                L = -F
                                if es_gen_len is None:
                                    es_gen_len = L.shape[-1]
                                es_losses_rows.extend([row.reshape(-1) for row in L])
            except Exception as _gpe:
                print(f"[store_latents] Failed to gather per-sample loss curves: {_gpe!r}")

            # Attach per-sample loss arrays and budgets if collected
            if ga_losses_rows and ga_steps_len is not None:
                ga_losses_per_sample = np.vstack(ga_losses_rows).astype(np.float32)
                payload["ga_losses_per_sample"] = ga_losses_per_sample
                payload["ga_budget"] = (2 * np.arange(1, ga_steps_len + 1)).astype(np.int32)
            if es_losses_rows and es_gen_len is not None:
                es_losses_per_sample = np.vstack(es_losses_rows).astype(np.float32)
                payload["es_generation_losses_per_sample"] = es_losses_per_sample
                # Budget per generation (start from gen1)
                if es_pop_size is None:
                    es_pop_size = 1
                payload["es_budget"] = (np.arange(1, es_gen_len + 1) * int(es_pop_size)).astype(np.int32)
            os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
            print(f"[store_latents] Saving to {store_path} with keys: {list(payload.keys())}")
            np.savez_compressed(store_path, **payload)
            print(f"Saved latent search data to {store_path}")
        except Exception as _e:
            import traceback
            print(f"Failed to store latents to {store_path}: {_e!r}")
            traceback.print_exc()
    return metrics


def pretty_print(metrics: dict) -> None:
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (jnp.ndarray, float, int)):
            if k == "total_final_loss":
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: not a scalar")


def main(
    artifact_path: str,
    json_challenges_file: Optional[str],
    json_solutions_file: Optional[str],
    only_n_tasks: Optional[int],
    dataset_folder: Optional[str],
    dataset_length: Optional[int],
    dataset_batch_size: Optional[int],
    dataset_use_hf: bool,
    dataset_seed: int,
    inference_mode: str,
    inference_mode_kwargs: dict,
    random_search_seed: int,
    mixed_precision: bool,
    no_wandb_run: bool = False,
    latent_dim_override: Optional[int] = None,
) -> None:
    print("Downloading the model artifact...")
    # Download the artifact and save the config file without creating separate W&B runs if requested
    if no_wandb_run:
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")
    else:
        os.environ["WANDB_MODE"] = "run"
        run = wandb.init()
        artifact = run.use_artifact(artifact_path, type="model")
        run.finish()
    cfg = omegaconf.OmegaConf.create(artifact.metadata)
    artifact_dir = artifact.download()
    omegaconf.OmegaConf.save(config=cfg, f=os.path.join(artifact_dir, "config.yaml"))

    print("Instantiating the model and the train state...")
    print(f"   - Config type: {type(cfg)}")
    print(f"   - Encoder config: {cfg.encoder_transformer}")
    print(f"   - Decoder config: {cfg.decoder_transformer}")
    if latent_dim_override is not None:
        print(f"   - Latent dimension override: {latent_dim_override}")
    lpn = instantiate_model(cfg, mixed_precision, latent_dim_override)
    print(f"   - Model created with encoder config: {lpn.encoder.config}")
    print(f"   - Model created with decoder config: {lpn.decoder.config}")
    train_state = instantiate_train_state(lpn)
    evaluator = Evaluator(
        lpn,
        inference_mode=inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        devices=None,
    )

    # Load the model weights
    print("Loading the model weights...")
    try:
        train_state = load_model_weights(train_state, artifact_dir)
        print(f"✅ Checkpoint loaded successfully")
        print(f"   - Checkpoint path: {artifact_dir}")
        print(f"   - Model config: latent_dim={cfg.encoder_transformer.get('latent_dim', 'N/A')}")
        print(f"   - Params structure: {list(train_state.params.keys()) if hasattr(train_state, 'params') else 'No params'}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"   - Checkpoint path: {artifact_dir}")
        print(f"   - Available files: {os.listdir(artifact_dir) if os.path.exists(artifact_dir) else 'Directory not found'}")
        raise

    # Put the train state on the device(s)
    train_state = jax.device_put_replicated(train_state, evaluator.devices)

    # Evaluate the model
    print(f"Inference mode: {evaluator.inference_mode}")
    kwargs = {k: v for k, v in evaluator.inference_mode_kwargs.items() if v is not None}
    if kwargs:
        print(f"Inference mode kwargs: {kwargs}")
    if json_challenges_file and json_solutions_file:
        metrics = evaluate_json(
            train_state,
            evaluator,
            json_challenges_file,
            json_solutions_file,
            only_n_tasks,
            random_search_seed,
        )
        pretty_print(metrics)
    if dataset_folder:
        metrics = evaluate_custom_dataset(
            train_state,
            evaluator,
            dataset_folder,
            dataset_length,
            dataset_batch_size,
            dataset_use_hf,
            dataset_seed,
            random_search_seed,
        )
        pretty_print(metrics)

    # If store-latents was requested but only JSON path was used, run one small pseudo-batch
    # from JSON to extract a representative trajectory for saving.
    try:
        store_path = evaluator.inference_mode_kwargs.get("store_latents_path", None)
    except Exception:
        store_path = None
    if store_path and not dataset_folder and json_challenges_file and json_solutions_file:
        print("[store_latents] Running a single-batch save pass for JSON mode.")
        # Reuse evaluate_json loader to get tasks, but here we just call the pmapped function once would be larger change.
        # For now, warn the user that custom dataset mode is recommended for latent saving.
        print("[store_latents] Note: latent saving is fully supported in dataset mode. Consider using -d for saving NPZ.")


def true_or_false_from_arg(arg: str) -> bool:
    if arg.lower() == "true":
        return True
    if arg.lower() == "false":
        return False
    raise ValueError(f"Invalid boolean argument '{arg}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model checkpoint on either the ARC json datasets or custom datasets."
            "Must provide arguments for -w, and, either -jc and -js, or -d."
        )
    )
    parser.add_argument(
        "-w",
        "--wandb-artifact-path",
        type=str,
        required=True,
        help="WandB path to the desired artifact. E.g. 'TheThinker/ARC/faithful-dawn-316--checkpoint:v76'.",
    )
    parser.add_argument(
        "-jc",
        "--json-challenges-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC challenges. E.g. 'json/arc-agi_training_challenges.json'.",
    )
    parser.add_argument(
        "-js",
        "--json-solutions-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC solutions. E.g. 'json/arc-agi_training_solutions.json'.",
    )
    parser.add_argument(
        "--only-n-tasks",
        type=int,
        required=False,
        default=None,
        help="Number of tasks to evaluate the model on. 'None' to run on all tasks.",
    )
    parser.add_argument(
        "-d",
        "--dataset-folder",
        type=str,
        required=False,
        default=None,
        help="Path to the folder with the custom dataset. E.g. 'storage/v0_main_fix_test'.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        required=False,
        default=None,
        help="Number of examples to evaluate the model on. 'None' to run on all examples.",
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the custom dataset evaluation. 'None' to use the length of the dataset.",
    )
    parser.add_argument(
        "--dataset-use-hf",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use Hugging Face to load the datasets (otherwise loads locally).",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        required=False,
        default=0,
        help="Seed to sample a subset of the custom dataset for evaluation.",
    )
    parser.add_argument(
        "-i",
        "--inference-mode",
        type=str,
        required=False,
        default="mean",
        help="Inference mode to use, choose from ['mean', 'first', 'random_search', 'gradient_ascent'].",
    )
    parser.add_argument(
        "--random-search-seed",
        type=int,
        required=False,
        default=0,
        help="Seed for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=None,
        help="Scale for the random noise added during the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=None,
        help="Number of steps for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=None,
        help="Learning rate for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to use a cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule-exponent",
        type=float,
        required=False,
        default=None,
        help="Exponent for the cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default=None,
        help="Optimizer to use for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=json.loads,
        required=False,
        default=None,
        help="Optimizer kwargs for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--accumulate-gradients-decoder-pairs",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to accumulate gradients for the decoder pairs in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--scan-gradients-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to scan gradients for the latents in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-mean-latent",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include the mean latent in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-all-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include all latents in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--random-perturbation",
        type=json.loads,
        required=False,
        default=None,
        help="Random perturbation kwargs. Requires 'num_samples' and 'scale' keys.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use mixed precision for inference.",
    )
    parser.add_argument(
        "--no-wandb-run",
        dest="no_wandb_run",
        type=true_or_false_from_arg,
        required=False,
        default=False,
        help="If True, do not create a W&B run; download artifact via API instead.",
    )
    parser.add_argument(
        "--store-latents",
        type=str,
        required=False,
        default=None,
        help="If set, save latent search trajectories to this .npz path (enables track_progress).",
    )
    # Evolutionary search parameters
    parser.add_argument(
        "--population-size",
        type=int,
        required=False,
        default=None,
        help="Population size for evolutionary search mode.",
    )
    parser.add_argument(
        "--num-generations", 
        type=int,
        required=False,
        default=None,
        help="Number of generations for evolutionary search mode.",
    )
    parser.add_argument(
        "--mutation-std",
        type=float,
        required=False,
        default=None,
        help="Mutation standard deviation for evolutionary search mode.",
    )
    
    # Subspace evolutionary search parameters
    parser.add_argument(
        "--use-subspace-mutation",
        action="store_true",
        required=False,
        help="Enable subspace mutation for evolutionary search (mutate in low-dimensional subspace).",
    )
    parser.add_argument(
        "--subspace-dim",
        type=int,
        required=False,
        default=32,
        help="Subspace dimension for evolutionary search (default: 32).",
    )
    parser.add_argument(
        "--ga-step-length",
        type=float,
        required=False,
        default=0.5,
        help="Target GA step length for automatic sigma scaling (default: 0.5).",
    )
    parser.add_argument(
        "--trust-region-radius",
        type=float,
        required=False,
        default=None,
        help="Trust region radius for evolutionary search (default: None).",
    )
    
    # Model configuration override
    parser.add_argument(
        "--latent-dim",
        type=int,
        required=False,
        default=None,
        help="Override the latent dimension from the checkpoint config. Use this when the checkpoint has a different latent dimension than expected.",
    )

    args = parser.parse_args()
    if (
        args.json_challenges_file is None
        and args.json_solutions_file is not None
        or args.json_challenges_file is not None
        and args.json_solutions_file is None
    ):
        parser.error("Must provide both the json challenges (-jc) and solutions (-js) files.")
    if args.json_challenges_file is None and args.dataset_folder is None:
        parser.error(
            "Must provide either the json challenges (-jc) and solutions (-js) files or the dataset folder (-d)."
        )
    if args.inference_mode not in ["mean", "first", "random_search", "gradient_ascent", "evolutionary_search"]:
        parser.error(
            "Invalid inference mode. Choose from ['mean', 'first', 'random_search', 'gradient_ascent', 'evolutionary_search']."
        )
    if args.inference_mode == "random_search":
        if args.num_samples is None:
            parser.error("The 'random_search' inference mode requires the --num-samples argument.")
        if args.scale is None:
            parser.error("The 'random_search' inference mode requires the --scale argument.")
    if args.inference_mode == "gradient_ascent":
        if args.num_steps is None:
            parser.error("The 'gradient_ascent' inference mode requires the --num-steps argument.")
        if args.lr is None:
            parser.error("The 'gradient_ascent' inference mode requires the --lr argument.")
    if args.inference_mode == "evolutionary_search":
        for arg in ["population_size", "num_generations", "mutation_std"]:
            if getattr(args, arg) is None:
                parser.error(f"The 'evolutionary_search' inference mode requires the --{arg} argument.")
    inference_mode_kwargs = {
        "num_samples": args.num_samples,
        "scale": args.scale,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "population_size": args.population_size,
        "num_generations": args.num_generations,
        "mutation_std": args.mutation_std,
        "use_subspace_mutation": args.use_subspace_mutation,
        "subspace_dim": args.subspace_dim,
        "ga_step_length": args.ga_step_length,
        "trust_region_radius": args.trust_region_radius,
    }
    for arg in [
        "scan_batch_size",
        "include_mean_latent",
        "include_all_latents",
        "lr_schedule",
        "lr_schedule_exponent",
        "optimizer",
        "optimizer_kwargs",
        "scan_gradients_latents",
        "accumulate_gradients_decoder_pairs",
        "random_perturbation",
    ]:
        if getattr(args, arg) is not None:
            inference_mode_kwargs[arg] = getattr(args, arg)
    
    # If storing latents, force track_progress and pass path down for dataset evaluation
    if args.store_latents is not None:
        inference_mode_kwargs["track_progress"] = True
        # We will use this key to signal saving in dataset eval path
        inference_mode_kwargs["store_latents_path"] = args.store_latents
    
    # For gradient ascent, always enable track_progress to capture loss information
    if args.inference_mode == "gradient_ascent":
        inference_mode_kwargs["track_progress"] = True
    main(
        artifact_path=args.wandb_artifact_path,
        json_challenges_file=args.json_challenges_file,
        json_solutions_file=args.json_solutions_file,
        only_n_tasks=args.only_n_tasks,
        dataset_folder=args.dataset_folder,
        dataset_length=args.dataset_length,
        dataset_batch_size=args.dataset_batch_size,
        dataset_use_hf=args.dataset_use_hf,
        dataset_seed=args.dataset_seed,
        inference_mode=args.inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        random_search_seed=args.random_search_seed,
        mixed_precision=args.mixed_precision,
        no_wandb_run=args.no_wandb_run,
        latent_dim_override=args.latent_dim,
    )
