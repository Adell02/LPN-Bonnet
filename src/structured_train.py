from __future__ import annotations

import logging
import time
from functools import partial
from typing import Optional, Sequence

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import omegaconf
import wandb
from flax.training.train_state import TrainState
from jax.tree_util import tree_map
from tqdm.auto import trange

from models.transformer import EncoderTransformer, DecoderTransformer
from models.utils import DecoderTransformerConfig, EncoderTransformerConfig
from models.structured_lpn import StructuredLPN, average_params
from data_utils import (
    load_datasets,
    shuffle_dataset_into_batches,
    data_augmentation_fn,
    make_leave_one_out,
)
from visualization import (
    visualize_dataset_generation,
    visualize_heatmap,
    visualize_tsne,
)


logging.getLogger().setLevel(logging.INFO)


def instantiate_config_for_mpt(transformer_cfg: omegaconf.DictConfig) -> DecoderTransformerConfig | EncoderTransformerConfig:
    import jax.numpy as jnp
    import hydra
    return hydra.utils.instantiate(
        transformer_cfg,
        transformer_layer=hydra.utils.instantiate(transformer_cfg.transformer_layer, dtype=jnp.bfloat16),
    )


def build_model_from_cfg(cfg: omegaconf.DictConfig) -> tuple[StructuredLPN, list[EncoderTransformer], DecoderTransformer]:
    # Prefer structured.model_config if provided to match artifact shapes
    mc = getattr(cfg.structured, "model_config", None)
    if mc is not None:
        enc_cfg = omegaconf.OmegaConf.create({
            "_target_": "models.utils.EncoderTransformerConfig",
            "max_rows": mc.max_rows,
            "max_cols": mc.max_cols,
            "num_layers": mc.num_layers,
            "transformer_layer": {
                "_target_": "models.utils.TransformerLayerConfig",
                "num_heads": mc.num_heads,
                "emb_dim_per_head": mc.emb_dim_per_head,
                "mlp_dim_factor": mc.mlp_dim_factor,
                "dropout_rate": mc.dropout_rate,
                "attention_dropout_rate": mc.attention_dropout_rate,
            },
            "latent_dim": mc.latent_dim,
            "variational": mc.variational,
            "latent_projection_bias": mc.latent_projection_bias,
        })
        dec_cfg = omegaconf.OmegaConf.create({
            "_target_": "models.utils.DecoderTransformerConfig",
            "max_rows": mc.max_rows,
            "max_cols": mc.max_cols,
            "num_layers": mc.num_layers,
            "transformer_layer": {
                "_target_": "models.utils.TransformerLayerConfig",
                "num_heads": mc.num_heads,
                "emb_dim_per_head": mc.emb_dim_per_head,
                "mlp_dim_factor": mc.mlp_dim_factor,
                "dropout_rate": mc.dropout_rate,
                "attention_dropout_rate": mc.attention_dropout_rate,
            },
        })
        enc = EncoderTransformer(hydra.utils.instantiate(enc_cfg))
        dec = DecoderTransformer(hydra.utils.instantiate(dec_cfg))
    else:
        # Fallback to explicit encoder/decoder configs
        if cfg.training.get("mixed_precision", False):
            enc = EncoderTransformer(instantiate_config_for_mpt(cfg.encoder_transformer))
            dec = DecoderTransformer(instantiate_config_for_mpt(cfg.decoder_transformer))
        else:
            enc = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
            dec = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))

    # Replicate encoder module K times (params will differ per artifact)
    num_models = len(cfg.structured.artifacts.models)
    encoders = tuple(enc for _ in range(num_models))
    model = StructuredLPN(encoders=encoders, decoder=dec)
    return model, list(encoders), dec


def load_artifact_params(artifact_ref: str, key: str = "params") -> dict:
    art = wandb.use_artifact(artifact_ref)
    art_dir = art.download()
    # Expect a msgpack serialized flax state named state.msgpack
    import os
    from flax.serialization import msgpack_restore
    state_path = os.path.join(art_dir, "state.msgpack")
    with open(state_path, "rb") as f:
        data = f.read()
    # Restore raw state dict written via to_state_dict(state)
    restored = msgpack_restore(data)
    if isinstance(restored, dict) and "params" in restored:
        return restored["params"]
    # Fallback if artifact directly stores params
    return restored


def build_params_from_artifacts(cfg: omegaconf.DictConfig, decoder_module: DecoderTransformer) -> tuple[list[dict], dict]:
    enc_params_list = []
    dec_params_list = []
    model_artifacts = list(cfg.structured.artifacts.models or [])
    for art in model_artifacts:
        full_params = load_artifact_params(art)
        # Expect top-level keys 'encoder' and 'decoder'
        enc_params = full_params["encoder"] if "encoder" in full_params else full_params
        dec_params = full_params["decoder"] if "decoder" in full_params else full_params
        enc_params_list.append(enc_params)
        dec_params_list.append(dec_params)

    if len(dec_params_list) == 0:
        raise ValueError(
            "No structured.artifacts.models provided. Populate structured.artifacts.models with one or more "
            "W&B artifact references to LPN checkpoints (encoder+decoder)."
        )
    if len(dec_params_list) == 1:
        avg_decoder_params = dec_params_list[0]
    else:
        avg_decoder_params = average_params(dec_params_list)
    return enc_params_list, avg_decoder_params


class StructuredTrainer:
    def __init__(self, cfg: omegaconf.DictConfig, model: StructuredLPN, encoders: list[EncoderTransformer], decoder: DecoderTransformer) -> None:
        self.cfg = cfg
        self.model = model
        self.encoders = encoders
        self.decoder = decoder
        self.num_devices = jax.device_count()
        self.devices = jax.local_devices()
        self.batch_size = cfg.training.batch_size
        self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError("batch_size must be divisible by gradient_accumulation_steps")

        # Training/eval datasets
        if cfg.training.get("struct_patterns_balanced", False):
            # Build a balanced dataset by concatenating equal-sized splits from the 3 struct patterns
            from datasets.task_gen.dataloader import make_dataset
            L = int(cfg.training.get("struct_length_per_pattern", 1024))
            N = int(cfg.training.get("struct_num_pairs", 2))
            grids_all, shapes_all = [], []
            for pid in (1, 2, 3):
                g, s, _ = make_dataset(
                    length=L,
                    num_pairs=N,
                    num_workers=0,
                    task_generator_class="STRUCT_PATTERN",
                    online_data_augmentation=False,
                    seed=cfg.training.seed,
                    pattern=pid,
                )
                grids_all.append(g)
                shapes_all.append(s)
            self.train_grids = jnp.concatenate(grids_all, axis=0)
            self.train_shapes = jnp.concatenate(shapes_all, axis=0)
        else:
            train_datasets = cfg.training.train_datasets
            if isinstance(train_datasets, str) and train_datasets:
                train_datasets = [train_datasets]
            grids, shapes = [], []
            if train_datasets:
                for grids_i, shapes_i, _ in load_datasets(train_datasets, cfg.training.get("use_hf", True)):
                    grids.append(grids_i)
                    shapes.append(shapes_i)
                self.train_grids = jnp.concat(grids, axis=0)
                self.train_shapes = jnp.concat(shapes, axis=0)
            else:
                raise ValueError("No training data specified: set training.train_datasets or enable struct_patterns_balanced")

        # Simple single eval dataset support (optional)
        self.eval_conf = cfg.eval.get("dataset")
        if self.eval_conf and self.eval_conf.get("folder"):
            eg, es, _ = load_datasets([self.eval_conf.folder], self.eval_conf.get("use_hf", True))[0]
            self.eval_grids = eg
            self.eval_shapes = es
        elif cfg.training.get("struct_patterns_balanced", False):
            # Build a small balanced eval sample
            from datasets.task_gen.dataloader import make_dataset
            L = 96
            N = int(cfg.training.get("struct_num_pairs", 2))
            grids_all, shapes_all = [], []
            for pid in (1, 2, 3):
                g, s, _ = make_dataset(
                    length=L,
                    num_pairs=N,
                    num_workers=0,
                    task_generator_class="STRUCT_PATTERN",
                    online_data_augmentation=False,
                    seed=cfg.training.seed + pid,
                    pattern=pid,
                )
                grids_all.append(g)
                shapes_all.append(s)
            self.eval_grids = jnp.concatenate(grids_all, axis=0)
            self.eval_shapes = jnp.concatenate(shapes_all, axis=0)

    def init_state(self, key: chex.PRNGKey, enc_params_list: list[dict], avg_decoder_params: dict) -> TrainState:
        variables = self.model.init(
            key,
            self.train_grids[:1],
            self.train_shapes[:1],
            dropout_eval=False,
            mode=self.cfg.training.inference_mode,
            poe_alphas=jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32),
            encoder_params_list=enc_params_list,
            decoder_params=avg_decoder_params,
        )

        # Mask: only decoder params are trainable
        def mask_fn(params):
            def mark(p):
                # Expect top-level keys: 'encoder', 'decoder' in original LPN; here only decoder matters
                return True
            return tree_map(lambda _: True, avg_decoder_params)

        lr = self.cfg.training.learning_rate
        linear_warmup_steps = self.cfg.training.get("linear_warmup_steps", 99)
        scheduler = optax.warmup_exponential_decay_schedule(
            init_value=lr / (linear_warmup_steps + 1),
            peak_value=lr,
            warmup_steps=linear_warmup_steps,
            transition_steps=1,
            end_value=lr,
            decay_rate=1.0,
        )
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.masked(optax.adamw(scheduler), mask_fn(avg_decoder_params)))

        # TrainState holds only decoder params for updates
        return TrainState.create(apply_fn=self.model.apply, tx=tx, params=avg_decoder_params)

    def prepare_train_dataset_for_epoch(self, key: chex.PRNGKey, log_every_n_steps: int) -> tuple[chex.Array, chex.Array]:
        """Shuffle the dataset and reshape it to (num_logs, log_every_n_steps, batch_size, *)."""
        shuffle_key, augmentation_key = jax.random.split(key)
        grids, shapes = shuffle_dataset_into_batches(self.train_grids, self.train_shapes, self.batch_size, shuffle_key)
        num_logs = grids.shape[0] // log_every_n_steps
        grids = grids[: num_logs * log_every_n_steps]
        shapes = shapes[: num_logs * log_every_n_steps]
        
        if self.cfg.training.online_data_augmentation:
            grids, shapes = data_augmentation_fn(grids, shapes, augmentation_key)
        
        # Reshape to (num_logs, log_every_n_steps, batch_size, *)
        grids = grids.reshape(num_logs, log_every_n_steps, self.batch_size, *grids.shape[2:])
        shapes = shapes.reshape(num_logs, log_every_n_steps, self.batch_size, *shapes.shape[2:])
        return grids, shapes

    def train_n_steps(self, state: TrainState, batches: tuple[chex.Array, chex.Array], key: chex.PRNGKey) -> tuple[TrainState, dict]:
        """Process log_every_n_steps batches and return updated state and metrics."""
        num_steps = batches[0].shape[0]  # Should be log_every_n_steps
        keys = jax.random.split(key, num_steps)
        
        # Process each batch sequentially (since we don't have pmap)
        all_metrics = []
        for i in range(num_steps):
            batch_pairs, batch_shapes = batches[0][i], batches[1][i]
            rng = keys[i]
            
            def loss_fn(decoder_params, batch_pairs, batch_shapes, rng):
                loss, metrics = self.model.apply(
                    {"params": decoder_params},
                    batch_pairs,
                    batch_shapes,
                    dropout_eval=False,
                    mode=self.cfg.training.inference_mode,
                    poe_alphas=jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32),
                    encoder_params_list=self.enc_params_list,
                    decoder_params=decoder_params,
                    rngs={"dropout": rng, "latents": rng},
                    prior_kl_coeff=self.cfg.training.get("prior_kl_coeff"),
                    pairwise_kl_coeff=self.cfg.training.get("pairwise_kl_coeff"),
                    **(self.cfg.training.get("inference_kwargs") or {}),
                )
                return loss, metrics
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch_pairs, batch_shapes, rng)
            state = state.apply_gradients(grads=grads)
            all_metrics.append(metrics)
        
        # Average metrics over all steps
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.stack([m[key] for m in all_metrics]))
        
        return state, avg_metrics

    def train(self, state: TrainState, enc_params_list: list[dict]) -> TrainState:
        cfg = self.cfg
        num_steps = cfg.training.total_num_steps
        log_every = cfg.training.log_every_n_steps
        self.enc_params_list = enc_params_list  # Store for train_n_steps
        
        step = 0
        epoch = 0
        key = jax.random.PRNGKey(cfg.training.seed)
        logging.info("Starting structured training...")
        logging.info(f"Total steps: {num_steps}, Log every: {log_every}, Batch size: {self.batch_size}")
        logging.info(f"Training schedule: Log every {log_every} steps, Eval every {cfg.training.get('eval_every_n_logs', 'disabled')} logs, Checkpoint every {cfg.training.get('save_checkpoint_every_n_logs', 'disabled')} logs")
        logging.info(f"With current config: Eval every {log_every * cfg.training.get('eval_every_n_logs', 0)} steps, Checkpoint every {log_every * cfg.training.get('save_checkpoint_every_n_logs', 0)} steps")
        
        # Test forward pass first to catch any issues early
        logging.info("Testing forward pass...")
        try:
            test_batch = self.train_grids[:self.batch_size], self.train_shapes[:self.batch_size]
            test_loss, test_metrics = self.model.apply(
                {"params": state.params},
                *test_batch,
                dropout_eval=False,
                mode=cfg.training.inference_mode,
                poe_alphas=jnp.asarray(cfg.structured.alphas, dtype=jnp.float32),
                encoder_params_list=enc_params_list,
                decoder_params=state.params,
                rngs={"dropout": key, "latents": key},
                prior_kl_coeff=cfg.training.get("prior_kl_coeff"),
                pairwise_kl_coeff=cfg.training.get("pairwise_kl_coeff"),
                **(cfg.training.get("inference_kwargs") or {}),
            )
            logging.info(f"Forward pass test successful: loss={float(test_loss):.4f}")
        except Exception as e:
            logging.error(f"Forward pass test failed: {e}")
            raise
        
        logging.info("Starting training loop...")
        pbar = trange(num_steps, disable=False)
        
        while step < num_steps:
            key, epoch_key = jax.random.split(key)
            # Prepare dataset for this epoch
            grids, shapes = self.prepare_train_dataset_for_epoch(epoch_key, log_every)
            dataloader = zip(grids, shapes)
            
            dataloading_time = time.time()
            for batches in dataloader:
                wandb.log({"timing/dataloading_time": time.time() - dataloading_time})
                
                # Training - process log_every_n_steps batches at once
                key, train_key = jax.random.split(key)
                start = time.time()
                state, metrics = self.train_n_steps(state, batches, train_key)
                end = time.time()
                
                pbar.update(log_every)
                step += log_every
                throughput = log_every * self.batch_size / (end - start)
                metrics.update({
                    "timing/train_time": end - start,
                    "timing/train_num_samples_per_second": throughput
                })
                wandb.log(metrics, step=step)

                # Save checkpoint
                if cfg.training.get("save_checkpoint_every_n_logs") and (step // log_every) % cfg.training.save_checkpoint_every_n_logs == 0:
                    try:
                        logging.info(f"Saving checkpoint at step {step}")
                        from flax.serialization import msgpack_serialize, to_state_dict
                        with open("state.msgpack", "wb") as outfile:
                            outfile.write(msgpack_serialize(to_state_dict(state)))
                        wandb.save("state.msgpack")
                    except Exception as e:
                        logging.warning(f"Checkpoint save failed: {e}")

                # Evaluation
                if cfg.training.get("eval_every_n_logs") and (step // log_every) % cfg.training.eval_every_n_logs == 0:
                    try:
                        logging.info(f"Running evaluation at step {step}")
                        self.evaluate(state, enc_params_list)
                    except Exception as e:
                        logging.warning(f"Eval failed: {e}")

                # Exit if the total number of steps is reached
                if step >= num_steps:
                    break
                
                dataloading_time = time.time()
            
            epoch += 1
        
        pbar.close()
        return state

    def evaluate(self, state: TrainState, enc_params_list: list[dict]) -> dict:
        """
        Evaluate the model using the same approach as train.py:
        
        ✅ IMPLEMENTED FEATURES:
        1. Leave-One-Out: Uses N-1 pairs as support, 1 pair as query
        2. Proper Batch Processing: Handles batches like train.py
        3. T-SNE Context Sampling: Uses generation context, not raw encoder latents
        4. Proper Mode Handling: Supports all inference modes (mean, gradient_ascent, random_search, evolutionary_search)
        5. Missing Metrics: correct_shapes, pixel_correctness, accuracy, search trajectory metrics
        
        This ensures structured_train.py emulates train.py exactly while maintaining
        the architectural differences (multiple encoders + PoE + single decoder).
        """
        if not hasattr(self, "eval_grids"):
            return {}
        cfg = self.cfg
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
        
        # 1. IMPLEMENT LEAVE-ONE-OUT: Create leave_one_out versions like train.py
        from data_utils import make_leave_one_out
        leave_one_out_pairs = make_leave_one_out(self.eval_grids, axis=-4)
        leave_one_out_shapes = make_leave_one_out(self.eval_shapes, axis=-3)
        
        # 2. IMPLEMENT PROPER BATCH PROCESSING: Handle batches like train.py
        batch_size = cfg.eval.get("batch_size", len(self.eval_grids))
        num_batches = len(self.eval_grids) // batch_size
        # Drop the last batch if it's not full
        if num_batches > 0:
            pairs = self.eval_grids[:num_batches * batch_size]
            shapes = self.eval_shapes[:num_batches * batch_size]
            leave_one_out_pairs = leave_one_out_pairs[:num_batches * batch_size]
            leave_one_out_shapes = leave_one_out_shapes[:num_batches * batch_size]
        
        # Process in batches
        all_output_grids = []
        all_output_shapes = []
        all_info = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_pairs = pairs[start_idx:end_idx]
            batch_shapes = shapes[start_idx:end_idx]
            batch_leave_one_out_pairs = leave_one_out_pairs[start_idx:end_idx]
            batch_leave_one_out_shapes = leave_one_out_shapes[start_idx:end_idx]
            
            key = jax.random.PRNGKey(i)  # Different key per batch
            
            # Generate output using leave_one_out approach (like train.py)
            batch_output_grids, batch_output_shapes, batch_info = self.model.apply(
                {"params": state.params},
                batch_leave_one_out_pairs,  # Use leave_one_out pairs as support
                batch_leave_one_out_shapes, # Use leave_one_out shapes as support
                batch_pairs[:, 0, ..., 0],  # Query: first pair
                batch_shapes[:, 0, ..., 0], # Query: first shape
                key,
                True,  # dropout_eval
                cfg.eval.inference_mode,
                False,  # return_two_best
                method=self.model.generate_output,
                poe_alphas=alphas,
                encoder_params_list=enc_params_list,
                decoder_params=state.params,
                **(cfg.eval.get("inference_kwargs") or {}),
            )
            
            all_output_grids.append(batch_output_grids)
            all_output_shapes.append(batch_output_shapes)
            all_info.append(batch_info)
        
        # Concatenate batch results
        output_grids = jnp.concatenate(all_output_grids, axis=0)
        output_shapes = jnp.concatenate(all_output_shapes, axis=0)
        # Merge info dictionaries
        info = {}
        for key in all_info[0].keys():
            if key == "context":
                # Concatenate contexts
                contexts = [inf[key] for inf in all_info]
                info[key] = jnp.concatenate(contexts, axis=0)
            else:
                # For other info, just take the first batch
                info[key] = all_info[0][key]

        # Move tensors to host (CPU) and convert to numpy for lightweight eval
        pairs_np = np.array(jax.device_get(pairs))
        shapes_np = np.array(jax.device_get(shapes))
        out_grids_np = np.array(jax.device_get(output_grids))
        out_shapes_np = np.array(jax.device_get(output_shapes))

        # Naming aligned with Trainer: use a test_name and log under test/<name>/...
        test_name = "structured_mean" if cfg.eval.get("inference_mode", "mean") == "mean" else f"structured_{cfg.eval.inference_mode}"

        # 3. IMPLEMENT MISSING METRICS: Compute metrics exactly like train.py
        # Get the ground truth from the original pairs (not leave_one_out)
        gt_grids = pairs_np[:, 0, ..., 1]  # Ground truth output grids
        gt_shapes = shapes_np[:, 0, ..., 1]  # Ground truth output shapes
        
        # Shape accuracy: check if predicted shapes match ground truth
        correct_shapes = np.all(out_shapes_np == gt_shapes, axis=-1)  # (L,)
        
        # Pixel accuracy: check if predicted grids match ground truth
        R, C = pairs_np.shape[-3], pairs_np.shape[-2]
        rows = np.arange(R)[None, :, None]                  # (1, R, 1)
        cols = np.arange(C)[None, None, :]                  # (1, 1, C)
        mask = (rows < gt_shapes[:, 0:1, None]) & (cols < gt_shapes[:, 1:2, None])  # (L, R, C)
        eq = (out_grids_np == gt_grids)                     # (L, R, C)
        pixels_equal = np.where(mask, eq, False)
        num_valid = (gt_shapes[:, 0] * gt_shapes[:, 1])     # (L,)
        pixel_correctness = pixels_equal.sum(axis=(1, 2)) / (num_valid + 1e-5)
        accuracy = pixels_equal.sum(axis=(1, 2)) == num_valid
        
        # 4. ADD SEARCH TRAJECTORY METRICS (if using optimization modes)
        search_metrics = {}
        if cfg.eval.inference_mode in ["gradient_ascent", "random_search", "evolutionary_search"]:
            # Check for optimization trajectory (gradient_ascent)
            if "optimization_trajectory" in info:
                traj = info["optimization_trajectory"]
                if "log_probs" in traj:
                    log_probs = np.array(traj["log_probs"])
                    if log_probs.ndim >= 2:
                        final_log_probs = log_probs[..., -1, :]  # Last step
                        best_final_log_probs = np.max(final_log_probs, axis=-1)  # Best candidate
                        final_losses = -best_final_log_probs  # Convert to positive loss
                        search_metrics[f"test/{test_name}/total_final_loss"] = float(np.mean(final_losses))
                        print(f"✅ Found gradient ascent trajectory with {log_probs.shape}")
            
            # Check for search trajectory (random_search)
            if "search_trajectory" in info:
                search_traj = info["search_trajectory"]
                if "sample_accuracies" in search_traj:
                    sample_accs = np.array(search_traj["sample_accuracies"])
                    if sample_accs.ndim >= 1:
                        max_acc = np.max(sample_accs)
                        search_metrics[f"test/{test_name}/final_best_accuracy"] = float(max_acc)
                        print(f"✅ Found random search trajectory with {sample_accs.shape}")
            
            # Check for evolutionary trajectory (evolutionary_search)
            if "evolutionary_trajectory" in info:
                es_traj = info["evolutionary_trajectory"]
                if "generation_fitness" in es_traj:
                    gen_fitness = np.array(es_traj["generation_fitness"])
                    if gen_fitness.ndim >= 1:
                        final_fitness = gen_fitness[..., -1]  # Last generation
                        final_losses = -final_fitness  # Convert fitness to loss
                        search_metrics[f"test/{test_name}/total_final_loss"] = float(np.mean(final_losses))
                        print(f"✅ Found evolutionary search trajectory with {gen_fitness.shape}")
        
        # Log what we found
        print(f"🔍 Evaluation mode: {cfg.eval.inference_mode}")
        print(f"🔍 Info keys: {list(info.keys())}")
        print(f"🔍 Search metrics found: {list(search_metrics.keys())}")
        
        metrics = {
            f"test/{test_name}/correct_shapes": float(np.mean(correct_shapes)),
            f"test/{test_name}/pixel_correctness": float(np.mean(pixel_correctness)),
            f"test/{test_name}/accuracy": float(np.mean(accuracy)),
            **search_metrics,  # Include search trajectory metrics
        }

        # Figures
        fig_heatmap = visualize_heatmap(
            (pixels_equal.sum(axis=(0)) / (mask.sum(axis=(0)) + 1e-5)),
            (mask.sum(axis=(0)) / (mask.sum() + 1e-5)),
        )
        # Limit number of tasks shown for memory efficiency
        num_show = int(cfg.eval.get("num_tasks_to_show", 5))
        num_show = max(1, min(num_show, int(pairs_np.shape[0])))
        # Visualization expects predicted grids/shapes with per-pair axis; tile our single prediction across pairs
        num_pairs = int(shapes_np.shape[1])
        pred_grids_vis = np.repeat(out_grids_np[:num_show, None, ...], num_pairs, axis=1)
        pred_shapes_vis = np.repeat(out_shapes_np[:num_show, None, :], num_pairs, axis=1)
        fig_gen = visualize_dataset_generation(pairs_np[:num_show], shapes_np[:num_show], pred_grids_vis, pred_shapes_vis, num_show)

        # 5. IMPLEMENT TSNE CONTEXT SAMPLING: Use generation context like train.py
        # Instead of raw encoder latents, use the context that was actually used for generation
        all_latents = []
        source_ids = []  # 0..E-1 for encoders, E for PoE, E+1 for generation context
        task_ids_list = []  # task indices per point for color
        
        # Add individual encoder latents (for comparison)
        for enc_idx, enc_params in enumerate(enc_params_list):
            mu_i, logvar_i = self.encoders[enc_idx].apply(
                {"params": enc_params}, 
                pairs, 
                shapes, 
                True, 
                mutable=False
            )
            lat = mu_i.mean(axis=-2)  # Mean over pairs
            lat_np = np.array(lat).reshape(-1, lat.shape[-1])
            all_latents.append(lat_np)
            source_ids.extend([enc_idx] * lat_np.shape[0])
            task_ids_list.append(np.arange(lat_np.shape[0]))
        
        # Add PoE combined latents
        if len(enc_params_list) > 1:
            # Collect encoder outputs for PoE computation
            enc_mus = []
            enc_logvars = []
            for enc_idx, enc_params in enumerate(enc_params_list):
                mu_i, logvar_i = self.encoders[enc_idx].apply(
                    {"params": enc_params}, 
                    pairs, 
                    shapes, 
                    True, 
                    mutable=False
                )
                enc_mus.append(mu_i)
                enc_logvars.append(logvar_i)
            
            mus = jnp.stack(enc_mus, axis=0)
            logvars = jnp.stack(enc_logvars, axis=0)
            alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
            from models.structured_lpn import poe_diag_gaussians
            mu_poe, _ = poe_diag_gaussians(mus, logvars, alphas)
            poe_lat = mu_poe.mean(axis=-2)
            poe_np = np.array(poe_lat).reshape(-1, poe_lat.shape[-1])
            all_latents.append(poe_np)
            source_ids.extend([len(enc_params_list)] * poe_np.shape[0])
            task_ids_list.append(np.arange(poe_np.shape[0]))
        
        # MOST IMPORTANT: Add the generation context (like train.py does)
        if "context" in info:
            generation_context = info["context"]
            if generation_context is not None:
                # Use the context that was actually used for generation
                context_np = np.array(generation_context).reshape(-1, generation_context.shape[-1])
                all_latents.append(context_np)
                source_ids.extend([len(enc_params_list) + 1] * context_np.shape[0])  # E+1 for generation context
                task_ids_list.append(np.arange(context_np.shape[0]))
                print(f"✅ Added generation context to T-SNE: {context_np.shape}")
            else:
                print("⚠️  No generation context found in info")
        else:
            print("⚠️  No 'context' key in info")

        latents_concat = np.concatenate(all_latents, axis=0)
        source_ids_np = np.array(source_ids)
        task_ids_concat = np.concatenate(task_ids_list, axis=0)

        # Downsample points for t-SNE to be memory efficient
        max_points = int(cfg.eval.get("tsne_max_points", 2000))
        if latents_concat.shape[0] > max_points:
            idx = np.random.RandomState(42).choice(latents_concat.shape[0], size=max_points, replace=False)
            latents_concat = latents_concat[idx]
            source_ids_np = source_ids_np[idx]
            task_ids_concat = task_ids_concat[idx]

        # Compute t-SNE figure using shared visualization helper
        from visualization import visualize_tsne_sources
        fig_tsne = visualize_tsne_sources(latents_concat, task_ids_concat, source_ids_np, max_points=int(cfg.eval.get("tsne_max_points", 2000)))

        wandb.log({
            f"test/{test_name}/pixel_accuracy": wandb.Image(fig_heatmap),
            f"test/{test_name}/generation": wandb.Image(fig_gen),
            f"test/{test_name}/latents": wandb.Image(fig_tsne),
            **metrics,
        })

        # Free figures to save memory
        import matplotlib.pyplot as plt
        plt.close(fig_heatmap)
        plt.close(fig_gen)
        plt.close(fig_tsne)

        # Release large intermediates
        del enc_mus, enc_logvars, all_latents, latents_concat, source_ids_np, mus, logvars, mu_poe, poe_np
        return metrics


@hydra.main(config_path="configs", version_base=None, config_name="structured")
def run(cfg: omegaconf.DictConfig):
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        settings=wandb.Settings(console="redirect"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    model, encoders, decoder = build_model_from_cfg(cfg)
    enc_params_list, avg_decoder_params = build_params_from_artifacts(cfg, decoder)
    trainer = StructuredTrainer(cfg, model, encoders, decoder)
    key = jax.random.PRNGKey(cfg.training.seed)
    state = trainer.init_state(key, enc_params_list, avg_decoder_params)
    # Resume logic if desired
    if cfg.training.get("resume_from_checkpoint"):
        try:
            import os
            from flax.serialization import from_bytes
            from flax.training.train_state import TrainState
            with open(cfg.training.resume_from_checkpoint, "rb") as f:
                data = f.read()
            state = from_bytes(state, data)
        except Exception as e:
            logging.warning(f"Resume failed: {e}")
    state = trainer.train(state, enc_params_list)
    trainer.evaluate(state, enc_params_list)


if __name__ == "__main__":
    run()


