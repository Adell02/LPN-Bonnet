"""
Structured LPN Training Script

This script implements structured training that is EQUIVALENT to regular training (train.py)
but with the architectural difference of using multiple frozen encoders + PoE + single trainable decoder.

KEY EQUIVALENCE FEATURES:
1. **Same Data Loading**: Uses task generator with STRUCT_PATTERN class (pattern=0) to mix all 3 patterns uniformly
2. **Same Training Loop**: Processes data in the same way as train.py
3. **Same Batch Processing**: Uses the same batch sizes and logging frequencies
4. **Same Evaluation**: Implements the same evaluation metrics and visualization

The only difference is the model architecture: instead of training both encoder and decoder,
this trains only the decoder while using multiple pre-trained encoders via Product of Experts (PoE).

This eliminates the data size mismatch that was causing training to get stuck.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

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

# For clustering metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

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
    visualize_tsne_sources,  # For different markers (encoders vs context)
)


logging.getLogger().setLevel(logging.INFO)


def compute_modularity_q(embeddings, labels, k=5):
    """
    Compute Modularity Q metric for clustering quality.
    
    Args:
        embeddings: [N, D] array of embeddings
        labels: [N] array of cluster labels
        k: number of neighbors for k-NN graph
        
    Returns:
        float: Modularity Q score (higher is better)
    """
    try:
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Create adjacency matrix (remove self-loops)
        N = len(embeddings)
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(1, k+1):  # Skip first neighbor (self)
                A[i, indices[i, j]] = 1.0
                A[indices[i, j], i] = 1.0  # Undirected graph
        
        # Compute degrees
        k_i = np.sum(A, axis=1)
        m = np.sum(A) / 2  # Total edge weight
        
        # Compute modularity Q
        Q = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    expected_edges = (k_i[i] * k_i[j]) / (2 * m)
                    actual_edges = A[i, j]
                    same_cluster = int(labels[i] == labels[j])
                    Q += (actual_edges - expected_edges) * same_cluster
        
        Q = Q / (2 * m)
        return float(Q)
        
    except Exception as e:
        logging.warning(f"Modularity Q computation failed: {e}")
        return 0.0


def compute_adjusted_rand_index(embeddings, true_labels, k=5):
    """
    Compute Adjusted Rand Index (ARI) for clustering quality.
    
    Args:
        embeddings: [N, D] array of embeddings
        true_labels: [N] array of true class labels
        k: number of neighbors for k-NN graph
        
    Returns:
        float: ARI score [-1, 1] (higher is better)
    """
    try:
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Create adjacency matrix
        N = len(embeddings)
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(1, k+1):  # Skip first neighbor (self)
                A[i, indices[i, j]] = 1.0
                A[indices[i, j], i] = 1.0  # Undirected graph
        
        # Use KMeans clustering to get predicted clusters
        # Determine number of clusters based on unique true labels
        n_clusters = len(np.unique(true_labels))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(embeddings)
        else:
            # If only one true label, assign all to same cluster
            predicted_labels = np.zeros(len(embeddings), dtype=int)
        
        # Compute ARI
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return float(ari)
        
    except Exception as e:
        logging.warning(f"ARI computation failed: {e}")
        return 0.0


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
            total_train_length = 1024  # Total training samples
            samples_per_pattern = total_train_length // 3  # ~341 samples per pattern
            N = int(cfg.training.get("struct_num_pairs", 4))  # Use 4 pairs like config
            
            grids_all, shapes_all = [], []
            for pid in (1, 2, 3):  # Generate from all 3 patterns (O, T, L tetrominos)
                g, s, _ = make_dataset(
                    length=samples_per_pattern,  # ~341 samples per pattern
                    num_pairs=N,  # 4 pairs per task
                    num_workers=cfg.training.get("num_workers", 4),
                    task_generator_class="STRUCT_PATTERN",
                    online_data_augmentation=cfg.training.online_data_augmentation,
                    seed=cfg.training.seed + pid,  # Different seed per pattern
                    pattern=pid,  # pattern 1, 2, 3 for O, T, L tetrominos
                )
                grids_all.append(g)
                shapes_all.append(s)
            
            # Concatenate to get balanced dataset: ~341 + ~341 + ~341 = 1024 total samples
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
            # Build a small balanced eval sample with equal representation from all 3 patterns
            from datasets.task_gen.dataloader import make_dataset
            total_eval_length = 96  # Total evaluation samples
            samples_per_pattern = total_eval_length // 3  # 32 samples per pattern
            N = int(cfg.training.get("struct_num_pairs", 4))  # Use 4 pairs like training
            
            grids_all, shapes_all = [], []
            for pid in (1, 2, 3):  # Generate from all 3 patterns (O, T, L tetrominos)
                g, s, _ = make_dataset(
                    length=samples_per_pattern,  # 32 samples per pattern
                    num_pairs=N,  # 4 pairs per task
                    num_workers=0,
                    task_generator_class="STRUCT_PATTERN",
                    online_data_augmentation=False,
                    seed=cfg.training.seed + pid,  # Different seed per pattern
                    pattern=pid,  # pattern 1, 2, 3 for O, T, L tetrominos
                )
                grids_all.append(g)
                shapes_all.append(s)
            
                    # Concatenate to get balanced dataset: 32 + 32 + 32 = 96 total samples
            self.eval_grids = jnp.concatenate(grids_all, axis=0)
            self.eval_shapes = jnp.concatenate(shapes_all, axis=0)
            
            # DEBUG: Log evaluation dataset info
            logging.info(f"Generated balanced evaluation dataset:")
            logging.info(f"  - Total samples: {self.eval_grids.shape[0]}")
            logging.info(f"  - Samples per pattern: {samples_per_pattern}")
            logging.info(f"  - Grids shape: {self.eval_grids.shape}")
            logging.info(f"  - Shapes shape: {self.eval_shapes.shape}")
            
        else:
            # Fallback: create a small balanced eval dataset even if struct_patterns_balanced=False
            from datasets.task_gen.dataloader import make_dataset
            total_eval_length = 96  # Total evaluation samples
            samples_per_pattern = total_eval_length // 3  # 32 samples per pattern
            N = int(cfg.training.get("struct_num_pairs", 4))  # Use 4 pairs like training
            
            grids_all, shapes_all = [], []
            for pid in (1, 2, 3):  # Generate from all 3 patterns (O, T, L tetrominos)
                g, s, _ = make_dataset(
                    length=samples_per_pattern,  # 32 samples per pattern
                    num_pairs=N,  # 4 pairs per task
                    num_workers=0,
                    task_generator_class="STRUCT_PATTERN",
                    online_data_augmentation=False,
                    seed=cfg.training.seed + pid,  # Different seed per pattern
                    pattern=pid,  # pattern 1, 2, 3 for O, T, L tetrominos
                )
                grids_all.append(g)
                shapes_all.append(s)
            
            # Concatenate to get balanced dataset: 32 + 32 + 32 = 96 total samples
            self.eval_grids = jnp.concatenate(grids_all, axis=0)
            self.eval_shapes = jnp.concatenate(shapes_all, axis=0)
            
            # DEBUG: Log evaluation dataset info
            logging.info(f"Generated fallback balanced evaluation dataset:")
            logging.info(f"  - Total samples: {self.eval_grids.shape[0]}")
            logging.info(f"  - Samples per pattern: {samples_per_pattern}")
            logging.info(f"  - Grids shape: {self.eval_grids.shape}")
            logging.info(f"  - Shapes shape: {self.eval_shapes.shape}")
        
        # Load test datasets for comprehensive evaluation (like train.py)
        self.test_datasets = []
        for i, dict_ in enumerate(cfg.eval.test_datasets or []):
            if dict_.get("generator", False):
                for arg in ["num_pairs", "length"]:
                    assert arg in dict_, f"Each test generator dataset must have arg '{arg}'."
                num_pairs, length = dict_["num_pairs"], dict_["length"]
                default_dataset_name = dict_["generator"]
                task_generator_kwargs = dict_.get("task_generator_kwargs") or {}
                grids, shapes, program_ids = make_dataset(
                    length,
                    num_pairs,
                    num_workers=0,  # No workers for evaluation
                    task_generator_class=dict_["generator"],
                    online_data_augmentation=False,
                    seed=cfg.training.seed + i,  # Different seed per test dataset
                    **task_generator_kwargs,
                )
            else:
                for arg in ["folder", "length"]:
                    assert arg in dict_, f"Each test dataset must have arg '{arg}'."
                folder, length = dict_["folder"], dict_["length"]
                default_dataset_name = folder.rstrip().split("/")[-1]
                grids, shapes, program_ids = load_datasets([folder], dict_.get("use_hf", True))[0]
            
            if length is not None:
                key = jax.random.PRNGKey(dict_.get("seed", cfg.training.seed + i))
                indices = jax.random.permutation(key, len(grids))[:length]
                grids, shapes, program_ids = grids[indices], shapes[indices], program_ids[indices]
            
            batch_size = dict_.get("batch_size", len(grids))
            # Drop the last batch if it's not full
            num_batches = len(grids) // batch_size
            grids, shapes, program_ids = (
                grids[: num_batches * batch_size],
                shapes[: num_batches * batch_size],
                program_ids[: num_batches * batch_size],
            )
            
            inference_mode = dict_.get("inference_mode", "mean")
            # Fix the test name construction
            if dict_.get("name"):
                # If explicit name is provided, use it directly with inference mode
                test_name = dict_["name"] + "_" + inference_mode
            else:
                # If no name provided, use default_dataset_name + inference_mode
                test_name = default_dataset_name + "_" + inference_mode
            
            # Remove the duplicate generator prefix if it exists
            if test_name.startswith("generator_generator"):
                test_name = test_name.replace("generator_generator", "generator", 1)
            
            inference_kwargs = dict_.get("inference_kwargs", {})
            
            # DEBUG: Log test dataset info
            logging.info(f"Generated test dataset '{test_name}':")
            logging.info(f"  - Grids shape: {grids.shape}")
            logging.info(f"  - Shapes shape: {shapes.shape}")
            logging.info(f"  - Program IDs: {np.unique(program_ids) if program_ids is not None else 'None'}")
            logging.info(f"  - Task generator kwargs: {task_generator_kwargs}")
            
            self.test_datasets.append({
                "test_name": test_name,
                "dataset_grids": grids,
                "dataset_shapes": shapes,
                "batch_size": batch_size,
                "num_tasks_to_show": dict_.get("num_tasks_to_show", 5),
                "program_ids": program_ids,
                "inference_mode": inference_mode,
                "inference_kwargs": inference_kwargs,
            })

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
                        
                        # Test datasets evaluation (like train.py)
                        if hasattr(self, 'test_datasets') and self.test_datasets:
                            for dataset_dict in self.test_datasets:
                                try:
                                    start = time.time()
                                    test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress = self.test_dataset_submission(
                                        state, enc_params_list, **dataset_dict
                                    )
                                    test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                                    
                                    # Upload all figures
                                    for fig, name in [
                                        (fig_grids, "generation"),
                                        (fig_heatmap, "pixel_accuracy"),
                                        (fig_latents, "latents"),
                                        (fig_latents_samples, "latents_samples"),
                                        (fig_search_progress, "search_progress"),
                                    ]:
                                        if fig is not None:
                                            test_metrics[f"test/{dataset_dict['test_name']}/{name}"] = wandb.Image(fig)
                                    
                                    wandb.log(test_metrics, step=step)
                                    plt.close('all')  # Close all figures to prevent memory leaks
                                    
                                except Exception as e:
                                    logging.warning(f"Test dataset {dataset_dict['test_name']} failed: {e}")
                        
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
        # The issue is that make_leave_one_out is adding an extra dimension instead of reducing it
        # We need to manually create the leave_one_out data with correct shapes
        from data_utils import make_leave_one_out
        
        # DEBUG: Log what we're evaluating
        logging.info(f"Evaluation dataset info:")
        logging.info(f"  - eval_grids shape: {self.eval_grids.shape}")
        logging.info(f"  - eval_shapes shape: {self.eval_shapes.shape}")
        logging.info(f"  - Total samples: {self.eval_grids.shape[0]}")
        logging.info(f"  - Expected: 96 samples (32 per pattern)")
        
        # Create leave_one_out data with correct shapes
        # For pairs: (L, N, R, C, 2) -> (L, N-1, R, C, 2) where N=4, so N-1=3
        # For shapes: (L, N, 2, 2) -> (L, N-1, 2, 2) where N=4, so N-1=3
        
        # The make_leave_one_out function is adding an extra dimension, so we need to fix it
        # Original: (288, 4, 5, 5, 2) -> make_leave_one_out -> (288, 4, 3, 5, 5, 2)
        # We want: (288, 3, 5, 5, 2) - remove the extra 4 dimension
        raw_leave_one_out_pairs = make_leave_one_out(self.eval_grids, axis=-4)
        raw_leave_one_out_shapes = make_leave_one_out(self.eval_shapes, axis=-3)
        
        # Fix the shapes by removing the extra dimension
        # From (288, 4, 3, 5, 5, 2) -> (288, 3, 5, 5, 2)
        # From (288, 4, 3, 2, 2) -> (288, 3, 2, 2)
        if raw_leave_one_out_pairs.shape[1] == 4:  # If the extra dimension is there
            leave_one_out_pairs = raw_leave_one_out_pairs[:, 0, :, :, :, :]  # Take first slice
            leave_one_out_shapes = raw_leave_one_out_shapes[:, 0, :, :, :]   # Take first slice
        else:
            # If no extra dimension, use as is
            leave_one_out_pairs = raw_leave_one_out_pairs
            leave_one_out_shapes = raw_leave_one_out_shapes
        
        # The leave_one_out should reduce the N dimension from 4 to 3
        # So pairs: (L, 3, R, C, 2) and shapes: (L, 3, 2, 2)
        expected_pairs_shape = (self.eval_grids.shape[0], 3, 5, 5, 2)
        expected_shapes_shape = (self.eval_shapes.shape[0], 3, 2, 2)
        
        assert leave_one_out_pairs.shape == expected_pairs_shape, f"Leave_one_out_pairs shape mismatch: got {leave_one_out_pairs.shape}, expected {expected_pairs_shape}"
        assert leave_one_out_shapes.shape == expected_shapes_shape, f"Leave_one_out_shapes shape mismatch: got {leave_one_out_shapes.shape}, expected {expected_shapes_shape}"
        
        # 2. IMPLEMENT PROPER BATCH PROCESSING: Handle batches like train.py
        batch_size = cfg.eval.get("batch_size", len(self.eval_grids))
        num_batches = len(self.eval_grids) // batch_size
        # Drop the last batch if it's not full
        if num_batches > 0:
            pairs = self.eval_grids[:num_batches * batch_size]
            shapes = self.eval_shapes[:num_batches * batch_size]
            leave_one_out_pairs = leave_one_out_pairs[:num_batches * batch_size]
            leave_one_out_shapes = leave_one_out_shapes[:num_batches * batch_size]
            
            # Ensure all batched data has consistent shapes
            assert pairs.shape[:-4] == shapes.shape[:-3], f"Batched shape mismatch: pairs={pairs.shape}, shapes={shapes.shape}"
            assert leave_one_out_pairs.shape[:-4] == leave_one_out_shapes.shape[:-3], f"Batched leave_one_out shape mismatch: pairs={leave_one_out_pairs.shape}, shapes={leave_one_out_shapes.shape}"
        
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
            
            # Ensure all batch data has consistent shapes
            assert batch_pairs.shape[:-4] == batch_shapes.shape[:-3], f"Batch {i} shape mismatch: pairs={batch_pairs.shape}, shapes={batch_shapes.shape}"
            assert batch_leave_one_out_pairs.shape[:-4] == batch_leave_one_out_shapes.shape[:-3], f"Batch {i} leave_one_out shape mismatch: pairs={batch_pairs.shape}, shapes={batch_shapes.shape}"
            
            key = jax.random.PRNGKey(i)  # Different key per batch
            
            # Generate output using leave_one_out approach (like train.py)
            try:
                # Ensure all encoders receive exactly the same input data with the same shapes
                expected_support_pairs_shape = (96, 3, 5, 5, 2)
                expected_support_shapes_shape = (96, 3, 2, 2)
                
                assert batch_leave_one_out_pairs.shape == expected_support_pairs_shape, f"Support pairs shape mismatch: got {batch_leave_one_out_pairs.shape}, expected {expected_support_pairs_shape}"
                assert batch_leave_one_out_shapes.shape == expected_support_shapes_shape, f"Support shapes shape mismatch: got {batch_leave_one_out_shapes.shape}, expected {expected_support_shapes_shape}"
                
                # Use generate_output method for evaluation (like train.py does)
                # Use apply with method parameter and pass all arguments as keyword arguments
                batch_output_grids, batch_output_shapes, batch_info = self.model.apply(
                    {"params": state.params},
                    method=self.model.generate_output,
                    pairs=batch_leave_one_out_pairs,  # support pairs
                    grid_shapes=batch_leave_one_out_shapes, # support shapes
                    input=batch_pairs[:, 0, ..., 0],  # query pair
                    input_grid_shape=batch_shapes[:, 0, ..., 0], # query shape
                    key=key,  # RNG key
                    dropout_eval=True,  # dropout_eval
                    mode=cfg.eval.inference_mode,  # mode
                    return_two_best=False,  # return_two_best
                    poe_alphas=alphas,  # poe_alphas
                    encoder_params_list=enc_params_list,  # encoder_params_list
                    decoder_params=state.params,  # decoder_params
                )
                
                all_output_grids.append(batch_output_grids)
                all_output_shapes.append(batch_output_shapes)
                all_info.append(batch_info)
                
            except Exception as e:
                logging.error(f"Batch {i} failed: {e}")
                logging.error(f"Batch {i} input shapes: pairs={batch_pairs.shape}, shapes={batch_shapes.shape}")
                continue
        
        # Handle empty results
        if not all_output_grids:
            logging.error("No successful generations - evaluation failed")
            return {}
        
        logging.info(f"Successfully processed {len(all_output_grids)} batches")
        
        # Concatenate batch results
        try:
            output_grids = jnp.concatenate(all_output_grids, axis=0)
            output_shapes = jnp.concatenate(all_output_shapes, axis=0)
            logging.info(f"Final concatenated shapes: grids={output_grids.shape}, shapes={output_shapes.shape}")
        except Exception as e:
            logging.error(f"Failed to concatenate outputs: {e}")
            logging.error(f"Output shapes: {[g.shape for g in all_output_grids]}")
            return {}

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
                        # Check for search trajectory (random_search)
            if "search_trajectory" in info:
                search_traj = info["search_trajectory"]
                if "sample_accuracies" in search_traj:
                    sample_accs = np.array(search_traj["sample_accuracies"])
                    if sample_accs.ndim >= 1:
                        max_acc = np.max(sample_accs)
                        search_metrics[f"test/{test_name}/final_best_accuracy"] = float(max_acc)
            
            # Check for evolutionary trajectory (evolutionary_search)
            if "evolutionary_trajectory" in info:
                es_traj = info["evolutionary_trajectory"]
                if "generation_fitness" in es_traj:
                    gen_fitness = np.array(es_traj["generation_fitness"])
                    if gen_fitness.ndim >= 1:
                        final_fitness = gen_fitness[..., -1]  # Last generation
                        final_losses = -final_fitness  # Convert fitness to loss
                        search_metrics[f"test/{test_name}/total_final_loss"] = float(np.mean(final_losses))
        

        
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

        # 5. IMPLEMENT TSNE WITH ENCODERS + CONTEXT: Show both with different markers
        # We want to see: encoder outputs vs final generation context, both colored by sample index
        # This allows us to compare how the SAME sample is represented across different sources
        all_latents = []
        source_ids = []  # 0, 1, 2 for encoders, 3 for generation context
        sample_ids_list = []  # sample indices (0-95) for each point - same across all sources
        
        # Create sample indices (0-95) - each sample gets a unique ID
        num_samples = self.eval_grids.shape[0]  # Should be 96
        sample_sequence = np.arange(num_samples)  # [0, 1, 2, ..., 95]
        
        # Add individual encoder latents (unique source_id per encoder)
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
            source_ids.extend([enc_idx] * lat_np.shape[0])  # enc_idx for each encoder (0, 1, 2)
            sample_ids_list.append(sample_sequence)  # Same sample sequence for each encoder
        
        # Add the generation context (source_id = num_encoders)
        if "context" in info:
            generation_context = info["context"]
            if generation_context is not None:
                # Reshape like train.py does
                context_np = np.array(generation_context).reshape(-1, generation_context.shape[-1])
                all_latents.append(context_np)
                source_ids.extend([len(enc_params_list)] * context_np.shape[0])  # num_encoders for generation context
                sample_ids_list.append(sample_sequence)  # Same sample sequence for context
        
        if all_latents:
            latents_concat = np.concatenate(all_latents, axis=0)
            source_ids_np = np.array(source_ids)
            sample_ids_concat = np.concatenate(sample_ids_list, axis=0)
            
            # Downsample points for t-SNE to be memory efficient
            max_points = int(cfg.eval.get("tsne_max_points", 500))
            if latents_concat.shape[0] > max_points:
                idx = np.random.RandomState(42).choice(latents_concat.shape[0], size=max_points, replace=False)
                latents_concat = latents_concat[idx]
                source_ids_np = source_ids_np[idx]
                sample_ids_concat = sample_ids_concat[idx]
            
            # Use visualize_tsne_sources to show different markers for encoders vs context
            fig_tsne = visualize_tsne_sources(
                latents=latents_concat,
                program_ids=sample_ids_concat,  # Sample indices (0-95) for colors
                source_ids=source_ids_np,        # 0 for encoders, 1 for context
                max_points=max_points,
                random_state=42
            )
            
            # COMPUTE CLUSTERING METRICS AND UPLOAD TO WANDB
            try:
                # Compute metrics for different k values to check sensitivity
                k_values = [3, 5, 10]
                clustering_metrics = {}
                
                for k in k_values:
                    # Modularity Q (no labels needed)
                    modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"clustering/modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index (using sample indices as ground truth)
                    ari_score = compute_adjusted_rand_index(latents_concat, sample_ids_concat, k=k)
                    clustering_metrics[f"clustering/ari_k{k}"] = ari_score
                
                # Log clustering metrics to WandB
                wandb.log(clustering_metrics, step=step if 'step' in locals() else None)
                logging.info(f"Clustering metrics computed: {clustering_metrics}")
                
            except Exception as e:
                logging.warning(f"Clustering metrics computation failed: {e}")
        else:
            fig_tsne = None

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
        del all_latents, latents_concat, source_ids_np, sample_ids_concat
        return metrics

    def test_dataset_submission(
        self,
        state: TrainState,
        enc_params_list: list[dict],
        test_name: str,
        dataset_grids: chex.Array,
        dataset_shapes: chex.Array,
        program_ids: Optional[chex.Array],
        batch_size: int,
        num_tasks_to_show: int = 5,
        inference_mode: str = "mean",
        inference_kwargs: dict = None,
    ) -> tuple[dict[str, float], Optional[plt.Figure], plt.Figure, Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Test dataset submission method for structured training (similar to train.py).
        Generates outputs using leave-one-out approach and computes metrics.
        """
        if inference_kwargs is None:
            inference_kwargs = {}
            
        # Create leave_one_out data
        leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
        leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
        
        # Process in batches
        all_output_grids = []
        all_output_shapes = []
        all_info = []
        
        num_batches = len(dataset_grids) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_grids = dataset_grids[start_idx:end_idx]
            batch_shapes = dataset_shapes[start_idx:end_idx]
            batch_leave_one_out_grids = leave_one_out_grids[start_idx:end_idx]
            batch_leave_one_out_shapes = leave_one_out_shapes[start_idx:end_idx]
            
            key = jax.random.PRNGKey(i)
            
            try:
                # Generate output using leave_one_out approach
                batch_output_grids, batch_output_shapes, batch_info = self.model.apply(
                    {"params": state.params},
                    method=self.model.generate_output,
                    pairs=batch_leave_one_out_grids,
                    grid_shapes=batch_leave_one_out_shapes,
                    input=batch_grids[:, 0, ..., 0],
                    input_grid_shape=batch_shapes[:, 0, ..., 0],
                    key=key,
                    dropout_eval=True,
                    mode=inference_mode,
                    return_two_best=False,
                    poe_alphas=jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32),
                    encoder_params_list=enc_params_list,
                    decoder_params=state.params,
                )
                
                all_output_grids.append(batch_output_grids)
                all_output_shapes.append(batch_output_shapes)
                all_info.append(batch_info)
                
            except Exception as e:
                logging.error(f"Test batch {i} failed: {e}")
                continue
        
        if not all_output_grids:
            logging.error("No successful test generations")
            return {}, None, None, None, None, None
        
        # Concatenate results
        output_grids = jnp.concatenate(all_output_grids, axis=0)
        output_shapes = jnp.concatenate(all_output_shapes, axis=0)
        
        # Merge info
        info = {}
        for key in all_info[0].keys():
            if key == "context":
                contexts = [inf[key] for inf in all_info]
                info[key] = jnp.concatenate(contexts, axis=0)
            else:
                info[key] = all_info[0][key]
        
        # Convert to numpy for evaluation
        grids_np = np.array(jax.device_get(dataset_grids))
        shapes_np = np.array(jax.device_get(dataset_shapes))
        out_grids_np = np.array(jax.device_get(output_grids))
        out_shapes_np = np.array(jax.device_get(output_shapes))
        
        # Compute metrics
        gt_grids = grids_np[:, 0, ..., 1]
        gt_shapes = shapes_np[:, 0, ..., 1]
        
        correct_shapes = np.all(out_shapes_np == gt_shapes, axis=-1)
        
        # Pixel accuracy
        R, C = grids_np.shape[-3], grids_np.shape[-2]
        rows = np.arange(R)[None, :, None]
        cols = np.arange(C)[None, None, :]
        mask = (rows < gt_shapes[:, 0:1, None]) & (cols < gt_shapes[:, 1:2, None])
        eq = (out_grids_np == gt_grids)
        pixels_equal = np.where(mask, eq, False)
        num_valid = (gt_shapes[:, 0] * gt_shapes[:, 1])
        pixel_correctness = pixels_equal.sum(axis=(1, 2)) / (num_valid + 1e-5)
        accuracy = pixels_equal.sum(axis=(1, 2)) == num_valid
        
        metrics = {
            f"test/{test_name}/correct_shapes": float(np.mean(correct_shapes)),
            f"test/{test_name}/pixel_correctness": float(np.mean(pixel_correctness)),
            f"test/{test_name}/accuracy": float(np.mean(accuracy)),
        }
        
        # Create visualizations
        fig_heatmap = visualize_heatmap(
            (pixels_equal.sum(axis=(0)) / (mask.sum(axis=(0)) + 1e-5)),
            (mask.sum(axis=(0)) / (mask.sum() + 1e-5)),
        )
        
        # Generation visualization
        num_show = max(1, min(num_tasks_to_show, int(grids_np.shape[0])))
        num_pairs = int(shapes_np.shape[1])
        pred_grids_vis = np.repeat(out_grids_np[:num_show, None, ...], num_pairs, axis=1)
        pred_shapes_vis = np.repeat(out_shapes_np[:num_show, None, :], num_pairs, axis=1)
        fig_gen = visualize_dataset_generation(grids_np[:num_show], shapes_np[:num_show], pred_grids_vis, pred_shapes_vis, num_show)
        
        # T-SNE visualization - Show encoders + context with different markers
        fig_latents = None
        fig_search_progress = None
        
        if "context" in info and program_ids is not None:
            context = info["context"]
            if context is not None:
                # Show both encoder outputs and generation context
                all_latents = []
                source_ids = []
                sample_ids_list = []
                
                # Add encoder outputs (unique source_id per encoder)
                for enc_idx, enc_params in enumerate(enc_params_list):
                    mu_i, logvar_i = self.encoders[enc_idx].apply(
                        {"params": enc_params}, 
                        dataset_grids, 
                        dataset_shapes, 
                        True, 
                        mutable=False
                    )
                    lat = mu_i.mean(axis=-2)  # Mean over pairs
                    lat_np = np.array(lat).reshape(-1, lat.shape[-1])
                    all_latents.append(lat_np)
                    source_ids.extend([enc_idx] * lat_np.shape[0])  # enc_idx for each encoder (0, 1, 2)
                    sample_ids_list.append(program_ids)  # Sample IDs for each encoder
                
                # Add generation context (source_id = num_encoders)
                context_np = np.array(context).reshape(-1, context.shape[-1])
                all_latents.append(context_np)
                source_ids.extend([len(enc_params_list)] * context_np.shape[0])  # num_encoders for context
                sample_ids_list.append(program_ids)  # Sample IDs for context
                
                if all_latents:
                    latents_concat = np.concatenate(all_latents, axis=0)
                    source_ids_np = np.array(source_ids)
                    sample_ids_concat = np.concatenate(sample_ids_list, axis=0)
                    
                    # Use visualize_tsne_sources for different markers
                    fig_latents = visualize_tsne_sources(
                        latents=latents_concat,
                        program_ids=sample_ids_concat,
                        source_ids=source_ids_np,
                        max_points=500,
                        random_state=42
                    )
                    
                    # COMPUTE CLUSTERING METRICS FOR TEST DATASETS
                    try:
                        # Compute metrics for different k values
                        k_values = [3, 5, 10]
                        test_clustering_metrics = {}
                        
                        for k in k_values:
                            # Modularity Q (no labels needed)
                            modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                            test_clustering_metrics[f"clustering/{test_name}/modularity_q_k{k}"] = modularity_q
                            
                            # Adjusted Rand Index (using sample indices as ground truth)
                            ari_score = compute_adjusted_rand_index(latents_concat, sample_ids_concat, k=k)
                            test_clustering_metrics[f"clustering/{test_name}/ari_k{k}"] = ari_score
                        
                        # Add clustering metrics to the main metrics dict
                        metrics.update(test_clustering_metrics)
                        logging.info(f"Test clustering metrics computed: {test_clustering_metrics}")
                        
                    except Exception as e:
                        logging.warning(f"Test clustering metrics computation failed: {e}")
        
        return metrics, fig_gen, fig_heatmap, fig_latents, None, fig_search_progress


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


