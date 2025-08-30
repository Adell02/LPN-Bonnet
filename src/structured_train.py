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

# from __future__ import annotations  # Not supported in Python 3.6

import logging
import matplotlib.pyplot as plt
import time
from functools import partial
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np

# Try to import sklearn for clustering metrics, but make it optional
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Clustering metrics will be disabled.")

import chex
import hydra
import jax
import jax.numpy as jnp
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
from datasets.task_gen.dataloader import make_task_gen_dataloader
from visualization import (
    visualize_dataset_generation,
    visualize_heatmap,
    visualize_tsne,
    visualize_tsne_sources,  # For different markers (encoders vs context)
    visualize_struct_confidence_panel,
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
    if not SKLEARN_AVAILABLE:
        logging.warning("sklearn not available, skipping Modularity Q computation")
        return 0.0
        
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
    if not SKLEARN_AVAILABLE:
        logging.warning("sklearn not available, skipping ARI computation")
        return 0.0
        
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
        if not getattr(mc, "variational", False):
            raise ValueError(
                "Encoders must be variational; set structured.model_config.variational=true."
            )
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
        if not getattr(cfg.encoder_transformer, "variational", False):
            raise ValueError(
                "Encoders must be variational; set encoder_transformer.variational=true."
            )
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
        # Optional: expose/unfreeze encoders for the first N gradient steps
        self.encoder_expose_steps = int(cfg.training.get("encoder_expose_steps", 0) or 0)

        # Training/eval datasets - Use task generator like train.py for on-the-fly generation
        if cfg.training.get("struct_patterns_balanced", False):
            # Use task generator for on-the-fly sample generation (like train.py)
            logging.info("Using task generator for on-the-fly sample generation (like train.py)")
            self.task_generator = True
            self.task_generator_kwargs = {
                "num_workers": cfg.training.get("num_workers", 4),
                "num_pairs": int(cfg.training.get("struct_num_pairs", 4)),
                "class": "STRUCT_PATTERN",
                "pattern": 0,  # pattern=0 mixes all 3 patterns uniformly
                "pattern_per_task": True,
                "num_rows": 5,
                "num_cols": 5,
                "online_data_augmentation": cfg.training.online_data_augmentation,
            }
            
            # Initialize dummy grids/shapes for model initialization (like train.py)
            num_pairs = self.task_generator_kwargs["num_pairs"]
            num_rows, num_cols = 5, 5  # Default grid size
            self.init_grids = jnp.zeros((1, num_pairs, num_rows, num_cols, 2), jnp.uint8)
            self.init_shapes = jnp.ones((1, num_pairs, 2, 2), jnp.uint8)
            
            # No fixed dataset - samples generated on-the-fly
            self.train_grids = None
            self.train_shapes = None
            self.shuffled_pattern_ids = None
            
            # CRITICAL: Configure uniform pattern distribution
            self.batch_size = cfg.training.batch_size
            self.samples_per_pattern_per_batch = self.batch_size // 3  # Ensure divisible by 3
            if self.batch_size % 3 != 0:
                logging.warning(f"Batch size {self.batch_size} not divisible by 3, adjusting for uniform pattern distribution")
                self.batch_size = (self.batch_size // 3) * 3
                logging.info(f"Adjusted batch size to {self.batch_size} for uniform pattern distribution")
            
            logging.info(f"Task generator configured: {self.task_generator_kwargs}")
            logging.info(f"Uniform pattern distribution: {self.samples_per_pattern_per_batch} samples per pattern per batch")
        else:
            # Fallback to fixed datasets if specified
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
                self.task_generator = False
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
        # Use appropriate initialization data based on whether we have fixed dataset or task generator
        if hasattr(self, 'task_generator') and self.task_generator:
            # Use init_grids/shapes for task generator (these are properly initialized)
            init_grids = self.init_grids
            init_shapes = self.init_shapes
        else:
            # Use train_grids/shapes for fixed dataset
            init_grids = self.train_grids[:1]
            init_shapes = self.train_shapes[:1]
        
        variables = self.model.init(
            key,
            init_grids,
            init_shapes,
            dropout_eval=False,
            mode=self.cfg.training.inference_mode,
            poe_alphas=jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32),
            encoder_params_list=enc_params_list,
            decoder_params=avg_decoder_params,
        )

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
        # Standard optimizer over full param tree; we will zero encoder grads manually after exposure
        # STABILIZATION: Increase gradient clipping for contrastive loss stability
        gradient_clip_norm = 5.0  # Increased from 1.0 to handle contrastive loss better
        tx = optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optax.adamw(scheduler))

        # Compose params for decoder and encoders
        combined_params = {
            "decoder": avg_decoder_params,
            "encoders": tuple(enc_params_list),
        }
        return TrainState.create(apply_fn=self.model.apply, tx=tx, params=combined_params)

    def prepare_train_dataset_for_epoch(self, key: chex.PRNGKey, log_every_n_steps: int) -> tuple[chex.Array, chex.Array]:
        """Shuffle the dataset and reshape it to (num_logs, log_every_n_steps, batch_size, *)."""
        # This method is only used for fixed datasets, not task generators
        if not hasattr(self, 'train_grids') or self.train_grids is None:
            raise ValueError("prepare_train_dataset_for_epoch called but no fixed dataset available. Use task generator instead.")
        
        shuffle_key, augmentation_key = jax.random.split(key)
        grids, shapes = shuffle_dataset_into_batches(
            self.train_grids, self.train_shapes, self.batch_size, shuffle_key
        )

        # Reset the batch index counter for proper pattern tracking
        # This ensures pattern_ids are correctly aligned with the actual data patterns
        self._current_batch_start_idx = 0

        num_batches = grids.shape[0]
        if num_batches < log_every_n_steps:
            raise ValueError(
                "Dataset provides only "
                f"{num_batches} batches but log_every_n_steps={log_every_n_steps}. "
                "Increase dataset size or reduce log_every_n_steps to avoid stalling."
            )

        num_logs = num_batches // log_every_n_steps
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
            
            def loss_fn(full_params, batch_pairs, batch_shapes, rng):
                # CRITICAL FIX: Extract TRUE pattern IDs from actual data content
                # The task generator creates random patterns, so we must analyze the data
                batch_size = batch_pairs.shape[0]
                
                # Extract true pattern IDs by analyzing the actual output grids
                # This ensures pattern_ids match the REAL data, not assumed ordering
                pattern_ids = self._extract_true_pattern_ids_from_data(batch_pairs, batch_shapes)
                
                # Validate pattern distribution
                unique_patterns, counts = jnp.unique(pattern_ids, return_counts=True)
                # Convert JAX arrays to Python types for safe logging
                unique_patterns_py = [int(p) for p in unique_patterns]
                counts_py = [int(c) for c in counts]
                pattern_distribution = dict(zip(unique_patterns_py, counts_py))
                logging.debug(f"True pattern distribution: {pattern_distribution}")
                
                # Ensure we have multiple patterns for contrastive loss to work
                if len(unique_patterns) < 2:
                    logging.warning(f"Only {len(unique_patterns)} unique patterns found. Contrastive loss may be ineffective.")
                
                loss, metrics = self.model.apply(
                    {"params": full_params["decoder"]},
                    batch_pairs,
                    batch_shapes,
                    dropout_eval=False,
                    mode=self.cfg.training.inference_mode,
                    poe_alphas=jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32),
                    encoder_params_list=full_params["encoders"],
                    decoder_params=full_params["decoder"],
                    rngs={"dropout": rng, "latents": rng},
                    prior_kl_coeff=self.cfg.training.get("prior_kl_coeff"),
                    pairwise_kl_coeff=self.cfg.training.get("pairwise_kl_coeff"),
                    repulsion_kl_coeff=self.cfg.training.get("repulsion_kl"),
                    contrastive_kl_coeff=self.cfg.training.get("contrastive_kl"),  # ADD CONTRASTIVE LOSS
                    pattern_ids=pattern_ids,  # ADD PATTERN IDS FOR CONTRASTIVE LOSS
                    **(self.cfg.training.get("inference_kwargs") or {}),
                )
                return loss, metrics
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch_pairs, batch_shapes, rng)
            # Zero encoder grads after exposure window - FIX LOGIC
            if self.encoder_expose_steps <= 0 and "encoders" in grads:
                zeros_enc = tree_map(lambda g: jnp.zeros_like(g), grads["encoders"])
                grads = dict(grads)
                grads["encoders"] = zeros_enc
                # Note: step logging moved to main training loop for better visibility
            elif self.encoder_expose_steps > 0 and "encoders" in grads:
                # Note: step logging moved to main training loop for better visibility
                pass
            state = state.apply_gradients(grads=grads)
            all_metrics.append(metrics)
        
        # Average metrics over all steps
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.stack([m[key] for m in all_metrics]))
        
        # Log repulsion loss if present
        if "repulsion_loss" in avg_metrics:
            # Convert JAX arrays to Python types for safe logging
            repulsion_loss_val = float(np.array(avg_metrics['repulsion_loss']))
            repulsion_loss_weighted_val = float(np.array(avg_metrics.get('repulsion_loss_weighted', 0)))
            logging.info(f"Repulsion loss: {repulsion_loss_val:.6f} (weighted: {repulsion_loss_weighted_val:.6f})")
        
        # Log contrastive loss if present
        if "contrastive_loss" in avg_metrics:
            # Convert JAX arrays to Python types for safe logging
            contrastive_loss_val = float(np.array(avg_metrics['contrastive_loss']))
            contrastive_loss_weighted_val = float(np.array(avg_metrics.get('contrastive_loss_weighted', 0)))
            logging.info(f"Contrastive loss: {contrastive_loss_val:.6f} (weighted: {contrastive_loss_weighted_val:.6f})")
            
            # STABILIZATION: Adaptive coefficient adjustment suggestion
            if abs(contrastive_loss_val) > 50.0:
                logging.warning(f"Contrastive loss is large ({contrastive_loss_val:.2f}). Consider reducing contrastive_kl coefficient.")
            elif abs(contrastive_loss_val) < 0.01:
                logging.info(f"Contrastive loss is very small ({contrastive_loss_val:.6f}). Consider increasing contrastive_kl coefficient.")
            
            if "contrastive_kl_mean" in avg_metrics:
                kl_mean_val = float(np.array(avg_metrics['contrastive_kl_mean']))
                logging.info(f"  - KL mean: {kl_mean_val:.6f}")
            if "contrastive_sign_mean" in avg_metrics:
                sign_mean_val = float(np.array(avg_metrics['contrastive_sign_mean']))
                logging.info(f"  - Sign mean: {sign_mean_val:.6f}")
            
            # NEW: Debug pattern ID effectiveness
            if "contrastive_sign_mean" in avg_metrics:
                sign_mean = float(np.array(avg_metrics['contrastive_sign_mean']))
                if abs(sign_mean) < 0.1:
                    logging.warning(f"Contrastive sign mean is very small ({sign_mean:.6f}). This suggests pattern IDs may not be effective.")
                    logging.warning(f"  - Check if pattern_ids match actual data patterns")
                    logging.warning(f"  - Verify encoder variance outputs are different")
                elif abs(sign_mean) > 0.9:
                    logging.info(f"Contrastive sign mean is strong ({sign_mean:.6f}). Pattern IDs appear effective ✓")
                else:
                    logging.info(f"Contrastive sign mean is moderate ({sign_mean:.6f}). Pattern IDs may need improvement.")
            
            # NEW: Debug encoder specialization progress
            if "contrastive_kl_mean" in avg_metrics:
                kl_mean = float(np.array(avg_metrics['contrastive_kl_mean']))
                if kl_mean < 0.01:
                    logging.warning(f"Contrastive KL mean is very small ({kl_mean:.6f}). Encoders may not be specializing.")
                    logging.warning(f"  - Consider increasing contrastive_kl coefficient")
                    logging.warning(f"  - Check encoder variance outputs")
                elif kl_mean > 1.0:
                    logging.info(f"Contrastive KL mean is large ({kl_mean:.6f}). Encoders are actively specializing ✓")
                else:
                    logging.info(f"Contrastive KL mean is moderate ({kl_mean:.6f}). Encoders showing some specialization.")
        
        # Log pattern distribution for this training step (for debugging)
        # With task generator, each batch contains a mix of all patterns
        logging.debug(f"Training step completed: {num_steps} steps, batch size: {self.batch_size}")
        logging.debug(f"Pattern distribution: Each batch contains samples from all 3 patterns (O, T, L tetrominos)")
        
        # Decrement exposure counter by number of gradient steps completed
        self.encoder_expose_steps = max(0, self.encoder_expose_steps - num_steps)
        return state, avg_metrics

    def _create_balanced_pattern_batch(self, batch_size: int, samples_per_pattern: int) -> tuple[list, list]:
        """
        Create a balanced batch with equal representation from all 3 patterns.
        
        This ensures each batch contains exactly the same number of samples from each pattern,
        which is crucial for proper contrastive loss computation.
        
        Args:
            batch_size: Total batch size (must be divisible by 3)
            samples_per_pattern: Number of samples per pattern per batch
            
        Returns:
            Tuple of (grids_list, shapes_list) with balanced pattern distribution
        """
        if batch_size % 3 != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by 3 for uniform pattern distribution")
        
        # Generate samples for each pattern
        grids_list = []
        shapes_list = []
        
        for pattern_id in [1, 2, 3]:  # O-tetromino, T-tetromino, L-tetromino
            # Generate samples_per_pattern samples for this pattern
            # Use make_task_gen_dataloader directly since make_dataset doesn't support STRUCT_PATTERN
            from datasets.task_gen.dataloader import make_task_gen_dataloader
            
            # Create dataloader for this specific pattern
            dataloader = make_task_gen_dataloader(
                batch_size=1,
                log_every_n_steps=1,
                num_workers=0,  # No workers for single batch generation
                task_generator_class="STRUCT_PATTERN",
                num_pairs=self.task_generator_kwargs["num_pairs"],
                online_data_augmentation=self.cfg.training.online_data_augmentation,
                seed=self.cfg.training.seed + pattern_id + (self._batch_counter if hasattr(self, '_batch_counter') else 0),
                pattern=pattern_id,  # Specific pattern
                pattern_per_task=True,
                num_rows=self.task_generator_kwargs.get("num_rows", 5),
                num_cols=self.task_generator_kwargs.get("num_cols", 5),
            )
            
            # Generate samples using the dataloader
            grids_list_pattern = []
            shapes_list_pattern = []
            for i, ((grids, shapes), _) in enumerate(zip(dataloader, range(samples_per_pattern))):
                # The dataloader returns (log_every_n_steps, batch_size, ...) format
                # Since we set batch_size=1 and log_every_n_steps=1, extract the actual data
                # grids shape: (1, 1, num_pairs, max_rows, max_cols, 2) -> (num_pairs, max_rows, max_cols, 2)
                # shapes shape: (1, 1, num_pairs, 2, 2) -> (num_pairs, 2, 2)
                grids_list_pattern.append(grids[0, 0])  # Extract from batch format
                shapes_list_pattern.append(shapes[0, 0])  # Extract from batch format
            
            # Stack the samples for this pattern
            g = jnp.stack(grids_list_pattern, axis=0)
            s = jnp.stack(shapes_list_pattern, axis=0)
            
            # DEBUG: Log the actual shapes returned by direct dataloader
            logging.debug(f"Pattern {pattern_id} - grids shape: {g.shape}, shapes shape: {s.shape}")
            
            grids_list.append(g)
            shapes_list.append(s)
        
        # Concatenate all patterns to create balanced batch
        balanced_grids = jnp.concatenate(grids_list, axis=0)
        balanced_shapes = jnp.concatenate(shapes_list, axis=0)
        
        # DEBUG: Log the final concatenated shapes
        logging.debug(f"Final balanced batch - grids shape: {balanced_grids.shape}, shapes shape: {balanced_shapes.shape}")
        
        # Increment batch counter for different seeds
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1
        
        return balanced_grids, balanced_shapes

    def _create_balanced_dataloader(self, log_every_n_steps: int):
        """
        Create a dataloader that generates balanced batches with uniform pattern distribution.
        
        Args:
            log_every_n_steps: Number of steps to log
            
        Returns:
            Generator that yields balanced batches in the expected format
        """
        # Generate all batches for this epoch
        all_grids = []
        all_shapes = []
        
        for step in range(log_every_n_steps):
            # Generate a balanced batch for this step
            balanced_grids, balanced_shapes = self._create_balanced_pattern_batch(
                self.batch_size, 
                self.samples_per_pattern_per_batch
            )
            all_grids.append(balanced_grids)
            all_shapes.append(balanced_shapes)
        
        # Stack all batches to create the expected format: (log_every_n_steps, batch_size, ...)
        stacked_grids = jnp.stack(all_grids, axis=0)  # (log_every_n_steps, batch_size, ...)
        stacked_shapes = jnp.stack(all_shapes, axis=0)  # (log_every_n_steps, batch_size, ...)
        
        # Yield the stacked batches in the expected format
        yield (stacked_grids, stacked_shapes)

    def _extract_true_pattern_ids_from_data(self, batch_pairs: chex.Array, batch_shapes: chex.Array) -> chex.Array:
        """
        CRITICAL: Extract true pattern IDs by analyzing the actual data content.
        
        This method analyzes the output grids to determine the actual tetromino pattern
        for each sample, ensuring pattern_ids match the REAL data.
        
        Args:
            batch_pairs: Shape (batch_size, num_pairs, rows, cols, 2) - input/output grids
            batch_shapes: Shape (batch_size, num_pairs, 2) - grid dimensions
            
        Returns:
            pattern_ids: Shape (batch_size,) - true pattern IDs (1=O, 2=T, 3=L)
        """
        batch_size = batch_pairs.shape[0]
        pattern_ids = []
        
        for i in range(batch_size):
            # Get the output grid for this sample (use first pair as representative)
            output_grid = batch_pairs[i, 0, :, :, 1]  # Shape: (rows, cols)
            
            # Analyze the pattern by counting active pixels and their distribution
            active_pixels = jnp.where(output_grid > 0, 1, 0)
            num_active = jnp.sum(active_pixels)
            
            # Tetrominos always have exactly 4 active pixels
            if num_active != 4:
                logging.warning(f"Sample {i} has {num_active} active pixels, expected 4. Using fallback pattern ID.")
                pattern_ids.append(1)  # Fallback to O-tetromino
                continue
            
            # Find the bounding box of active pixels
            active_coords = jnp.where(active_pixels == 1)
            if len(active_coords[0]) == 0:
                pattern_ids.append(1)  # Fallback
                continue
                
            min_row, max_row = jnp.min(active_coords[0]), jnp.max(active_coords[0])
            min_col, max_col = jnp.min(active_coords[1]), jnp.max(active_coords[1])
            
            # Calculate dimensions of the bounding box
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            # Pattern classification based on bounding box dimensions and pixel distribution
            if height == 2 and width == 2:
                # 2x2 box: O-tetromino
                pattern_id = 1
            elif (height == 2 and width == 3) or (height == 3 and width == 2):
                # 2x3 or 3x2 box: T or L tetromino
                # Further classify by checking if it's T (centered) or L (corner)
                if height == 2:  # 2x3 box
                    # Check if middle column has pixel (T-tetromino characteristic)
                    middle_col = min_col + 1
                    if jnp.any(active_coords[1] == middle_col):
                        pattern_id = 2  # T-tetromino
                    else:
                        pattern_id = 3  # L-tetromino
                else:  # 3x2 box
                    # Check if bottom row has 2 pixels (L-tetromino characteristic)
                    bottom_row = max_row
                    bottom_pixels = jnp.sum(active_coords[1] == bottom_row)
                    if bottom_pixels == 2:
                        pattern_id = 3  # L-tetromino
                    else:
                        pattern_id = 2  # T-tetromino
            else:
                # Unexpected dimensions, use fallback
                logging.warning(f"Sample {i} has unexpected bounding box {height}x{width}. Using fallback pattern ID.")
                pattern_id = 1  # Fallback to O-tetromino
            
            pattern_ids.append(pattern_id)
        
        return jnp.array(pattern_ids, dtype=jnp.int32)

    def _validate_encoder_variance_outputs(self, state: TrainState, test_batch: tuple) -> None:
        """
        CRITICAL: Validate that encoders are outputting proper variance terms.
        
        This ensures that the contrastive loss can drive encoder specialization
        through different certainty levels.
        
        Args:
            state: Current training state
            test_batch: Test batch for validation
        """
        logging.info("Validating encoder variance outputs...")
        
        try:
            # Test encoder outputs on a small batch
            test_pairs, test_shapes = test_batch
            test_batch_size = min(4, test_pairs.shape[0])  # Use small batch for validation
            test_pairs_small = test_pairs[:test_batch_size]
            test_shapes_small = test_shapes[:test_batch_size]
            
            # Check each encoder's output
            for enc_idx, enc_params in enumerate(state.params["encoders"]):
                try:
                    # Get encoder outputs
                    mu_i, logvar_i = self.encoders[enc_idx].apply(
                        {"params": enc_params}, 
                        test_pairs_small, 
                        test_shapes_small, 
                        True,  # training mode
                        mutable=False
                    )
                    
                    # Check shapes
                    logging.info(f"Encoder {enc_idx} outputs:")
                    logging.info(f"  - mu_i shape: {mu_i.shape}")
                    logging.info(f"  - logvar_i shape: {logvar_i.shape}")
                    
                    # Check variance values
                    var_i = jnp.exp(logvar_i)
                    min_var = float(jnp.min(var_i))
                    max_var = float(jnp.max(var_i))
                    mean_var = float(jnp.mean(var_i))
                    
                    logging.info(f"  - Variance range: [{min_var:.6f}, {max_var:.6f}]")
                    logging.info(f"  - Mean variance: {mean_var:.6f}")
                    
                    # Check for fixed variance (indicating non-variational behavior)
                    if max_var - min_var < 1e-6:
                        logging.warning(f"Encoder {enc_idx} has nearly fixed variance! This will prevent specialization.")
                        logging.warning(f"  - All variance values are approximately {mean_var:.6f}")
                        logging.warning(f"  - Consider checking encoder configuration or training history")
                    else:
                        logging.info(f"  - Encoder {enc_idx} has variable variance ✓")
                        
                except Exception as e:
                    logging.error(f"Failed to validate encoder {enc_idx}: {e}")
                    continue
            
            logging.info("Encoder variance validation completed")
            
        except Exception as e:
            logging.error(f"Encoder variance validation failed: {e}")

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
        eval_every_n_logs = cfg.training.get('eval_every_n_logs')
        save_checkpoint_every_n_logs = cfg.training.get('save_checkpoint_every_n_logs')
        
        logging.info(f"Training schedule: Log every {log_every} steps, Eval every {eval_every_n_logs or 'disabled'} logs, Checkpoint every {save_checkpoint_every_n_logs or 'disabled'} logs")
        
        if eval_every_n_logs is not None:
            logging.info(f"With current config: Eval every {log_every * eval_every_n_logs} steps")
        else:
            logging.info("With current config: Evaluation disabled")
            
        if save_checkpoint_every_n_logs is not None:
            logging.info(f"With current config: Checkpoint every {log_every * save_checkpoint_every_n_logs} steps")
        else:
            logging.info("With current config: Checkpointing disabled")
        logging.info(f"Encoder exposure period: {self.encoder_expose_steps} steps (encoders trainable during this period)")
        logging.info(f"Repulsion KL coefficient: {cfg.training.get('repulsion_kl', 'disabled')}")
        logging.info(f"Contrastive KL coefficient: {cfg.training.get('contrastive_kl', 'disabled')}")
        logging.info(f"Training with {len(cfg.structured.artifacts.models)} encoders for pattern specialization")
        
        # Test forward pass first to catch any issues early
        logging.info("Testing forward pass...")
        try:
            # Generate a balanced test batch with uniform pattern distribution
            if hasattr(self, 'task_generator') and self.task_generator:
                test_grids, test_shapes = self._create_balanced_pattern_batch(
                    self.batch_size, 
                    self.samples_per_pattern_per_batch
                )
                test_batch = test_grids, test_shapes
            else:
                # Fallback to fixed dataset
                if hasattr(self, 'train_grids') and self.train_grids is not None:
                    test_batch = self.train_grids[:self.batch_size], self.train_shapes[:self.batch_size]
                else:
                    # No fixed dataset available, create a minimal test batch
                    logging.warning("No fixed dataset available for test forward pass, creating minimal test batch")
                    num_pairs = self.task_generator_kwargs["num_pairs"]
                    test_grids = jnp.zeros((self.batch_size, num_pairs, 5, 5, 2), jnp.uint8)
                    test_shapes = jnp.ones((self.batch_size, num_pairs, 2, 2), jnp.uint8)
                    test_batch = test_grids, test_shapes
            
            # CRITICAL FIX: Extract TRUE pattern IDs from actual test data
            test_batch_size = test_batch[0].shape[0]
            
            # Extract true pattern IDs by analyzing the actual data content
            test_pattern_ids = self._extract_true_pattern_ids_from_data(test_batch[0], test_batch[1])
            
            # Validate pattern_ids
            logging.debug(f"Pattern IDs validation:")
            logging.debug(f"  - Shape: {test_pattern_ids.shape}")
            logging.debug(f"  - Dtype: {test_pattern_ids.dtype}")
            logging.debug(f"  - Min value: {int(jnp.min(test_pattern_ids))}")
            logging.debug(f"  - Max value: {int(jnp.max(test_pattern_ids))}")
            logging.debug(f"  - Unique values: {jnp.unique(test_pattern_ids)}")
            
            # Log true pattern distribution from actual data analysis
            unique_patterns, counts = jnp.unique(test_pattern_ids, return_counts=True)
            # Convert JAX arrays to Python types for safe dictionary creation
            unique_patterns_py = [int(p) for p in unique_patterns]
            counts_py = [int(c) for c in counts]
            pattern_distribution = dict(zip(unique_patterns_py, counts_py))
            logging.info(f"Test forward pass: Using TRUE pattern extraction from data")
            logging.info(f"  - Batch size: {test_batch_size}")
            logging.info(f"  - True pattern distribution: {pattern_distribution}")
            logging.info(f"  - Pattern IDs: {[int(p) for p in test_pattern_ids[:10]]}... (first 10)")
            
            # CRITICAL: Validate encoder variance outputs before training
            self._validate_encoder_variance_outputs(state, test_batch)
            
            test_loss, test_metrics = self.model.apply(
                {"params": state.params["decoder"]},
                *test_batch,
                dropout_eval=False,
                mode=cfg.training.inference_mode,
                poe_alphas=jnp.asarray(cfg.structured.alphas, dtype=jnp.float32),
                encoder_params_list=state.params["encoders"],
                decoder_params=state.params["decoder"],
                rngs={"dropout": key, "latents": key},
                prior_kl_coeff=cfg.training.get("prior_kl_coeff"),
                pairwise_kl_coeff=cfg.training.get("pairwise_kl_coeff"),
                repulsion_kl_coeff=cfg.training.get("repulsion_kl"),
                contrastive_kl_coeff=cfg.training.get("contrastive_kl"),  # ADD CONTRASTIVE LOSS
                pattern_ids=test_pattern_ids,  # ADD PATTERN IDS FOR CONTRASTIVE LOSS
                **(cfg.training.get("inference_kwargs") or {}),
            )
            logging.info(f"Forward pass test successful: loss={float(test_loss):.4f}")
        except Exception as e:
            logging.error(f"Forward pass test failed: {e}")
            logging.error(f"Error details: {type(e).__name__}: {str(e)}")
            logging.error(f"Test batch shapes: grids={test_batch[0].shape}, shapes={test_batch[1].shape}")
            logging.error(f"Pattern IDs shape: {test_pattern_ids.shape}, dtype: {test_pattern_ids.dtype}")
            raise
        
        logging.info("Starting training loop...")
        pbar = trange(num_steps, disable=False)
        
        # Run evaluation at step 0 (first step)
        if cfg.training.get("eval_every_n_logs"):
            try:
                logging.info(f"Running evaluation at step 0 (first step)")
                self.evaluate(state, enc_params_list)
                
                # Test datasets evaluation at first step
                if hasattr(self, 'test_datasets') and self.test_datasets:
                    for dataset_dict in self.test_datasets:
                        try:
                            start = time.time()
                            test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress, fig_tsne_context, fig_tsne_encoders_list = self.test_dataset_submission(
                                state, dataset_dict
                            )
                            test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                            
                            # Upload all figures
                            for fig, name in [
                                (fig_grids, "generation"),
                                (fig_heatmap, "pixel_accuracy"),
                                (fig_latents, "latents"),
                                (fig_latents_samples, "latents_samples"),
                                (fig_search_progress, "search_progress"),
                                (fig_tsne_context, "latents_context_only"),
                            ]:
                                if fig is not None:
                                    test_metrics[f"test/{dataset_dict['test_name']}/{name}"] = wandb.Image(fig)
                            
                            # Upload all pattern-specific T-SNE plots
                            pattern_names = {1: "O-tetromino", 2: "T-tetromino", 3: "L-tetromino"}
                            for pattern_idx, fig_tsne_encoders_single in enumerate(fig_tsne_encoders_list, 1):
                                if fig_tsne_encoders_single is not None:
                                    test_metrics[f"test/{dataset_dict['test_name']}/latents_encoders_pattern{pattern_idx}"] = wandb.Image(fig_tsne_encoders_single)
                                    logging.info(f"Logged T-SNE for pattern {pattern_idx} ({pattern_names[pattern_idx]})")
                                else:
                                    logging.warning(f"No T-SNE plot available for pattern {pattern_idx}")
                            
                            wandb.log(test_metrics)
                            plt.close('all')  # Close all figures to prevent memory leaks
                            # Explicitly close additional T-SNE figures
                            if fig_tsne_context is not None:
                                plt.close(fig_tsne_context)
                            # Close all pattern-specific T-SNE figures
                            for fig_tsne_encoders_single in fig_tsne_encoders_list:
                                if fig_tsne_encoders_single is not None:
                                    plt.close(fig_tsne_encoders_single)
                            
                        except Exception as e:
                            logging.warning(f"Test dataset {dataset_dict['test_name']} failed at step 0: {e}")
                            
            except Exception as e:
                logging.warning(f"Initial evaluation failed: {e}")
        
        while step < num_steps:
            key, epoch_key = jax.random.split(key)
            
            # Prepare dataset for this epoch - Use balanced pattern generation
            if hasattr(self, 'task_generator') and self.task_generator:
                # Use balanced pattern generation for uniform distribution
                logging.info(f"Using balanced pattern generation for epoch {epoch}")
                # Create a simple dataloader that generates balanced batches
                dataloader = self._create_balanced_dataloader(log_every)
            else:
                # Fallback to fixed dataset (if specified)
                grids, shapes = self.prepare_train_dataset_for_epoch(epoch_key, log_every)
                dataloader = zip(grids, shapes)
                logging.info(f"Using fixed dataset for epoch {epoch}")
            
            # Log current step and encoder status
            if step % 100 == 0:
                encoder_status = "TRAINABLE" if self.encoder_expose_steps > 0 else "FROZEN"
                logging.info(f"Step {step}/{num_steps}: Encoders {encoder_status} (exposure: {self.encoder_expose_steps} steps remaining)")
            
            dataloading_time = time.time()
            for batches in dataloader:
                wandb.log({"timing/dataloading_time": time.time() - dataloading_time})
                
                # Training - process log_every_n_steps batches at once
                key, train_key = jax.random.split(key)
                start = time.time()
                
                # Log pattern distribution for current batch (safe to do outside JAX context)
                batch_size = batches[0].shape[1] if hasattr(batches[0], 'shape') and len(batches[0].shape) > 1 else len(batches[0])
                samples_per_pattern = batch_size // 3
                logging.debug(f"Processing balanced batch with {batch_size} samples")
                logging.debug(f"Pattern distribution: {samples_per_pattern} samples per pattern (O, T, L tetrominos)")
                logging.debug(f"Pattern structure: [O-tetromino x{samples_per_pattern}, T-tetromino x{samples_per_pattern}, L-tetromino x{samples_per_pattern}]")
                
                state, metrics = self.train_n_steps(state, batches, train_key)
                end = time.time()
                
                pbar.update(log_every)
                step += log_every
                
                # Log encoder status after step update
                if step % 100 == 0:
                    encoder_status = "TRAINABLE" if self.encoder_expose_steps > 0 else "FROZEN"
                    logging.info(f"Step {step}/{num_steps}: Encoders {encoder_status} (exposure: {self.encoder_expose_steps} steps remaining)")
                throughput = log_every * self.batch_size / (end - start)
                metrics.update({
                    "timing/train_time": end - start,
                    "timing/train_num_samples_per_second": throughput
                })
                
                # Add contrastive loss to Charts section for better visualization
                if "contrastive_loss" in metrics:
                    # STABILIZATION: Monitor contrastive loss magnitude and clip if needed
                    try:
                        contrastive_loss_val = float(np.array(metrics["contrastive_loss"]))
                        contrastive_loss_weighted_val = float(np.array(metrics["contrastive_loss_weighted"]))
                        
                        # Safety check: if contrastive loss is exploding, log warning
                        if abs(contrastive_loss_val) > 100.0:
                            logging.warning(f"Contrastive loss is very large: {contrastive_loss_val:.2f}. Consider reducing contrastive_kl coefficient.")
                    except Exception as e:
                        logging.warning(f"Could not monitor contrastive loss magnitude: {e}")
                    
                    metrics["Charts/contrastive_loss"] = metrics["contrastive_loss"]
                    metrics["Charts/contrastive_loss_weighted"] = metrics["contrastive_loss_weighted"]
                    # Add additional contrastive loss metrics to Charts
                    if "contrastive_kl_mean" in metrics:
                        metrics["Charts/contrastive_kl_mean"] = metrics["contrastive_kl_mean"]
                    if "contrastive_sign_mean" in metrics:
                        metrics["Charts/contrastive_sign_mean"] = metrics["contrastive_sign_mean"]
                
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

                # Evaluation - More frequent during encoder exposure period
                eval_interval = 5 if self.encoder_expose_steps > 0 else cfg.training.get("eval_every_n_logs", 0)
                if eval_interval and (step // log_every) % eval_interval == 0:
                    try:
                        logging.info(f"Running evaluation at step {step}")
                        self.evaluate(state, enc_params_list)
                        
                        # Test datasets evaluation (like train.py)
                        if hasattr(self, 'test_datasets') and self.test_datasets:
                            for dataset_dict in self.test_datasets:
                                try:
                                    start = time.time()
                                    test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress, fig_tsne_context, fig_tsne_encoders_list = self.test_dataset_submission(
                                        state, dataset_dict
                                    )
                                    test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                                    
                                    # Upload all figures
                                    for fig, name in [
                                        (fig_grids, "generation"),
                                        (fig_heatmap, "pixel_accuracy"),
                                        (fig_latents, "latents"),
                                        (fig_latents_samples, "latents_samples"),
                                        (fig_search_progress, "search_progress"),
                                        (fig_tsne_context, "latents_context_only"),
                                    ]:
                                        if fig is not None:
                                            test_metrics[f"test/{dataset_dict['test_name']}/{name}"] = wandb.Image(fig)
                                    
                                    # Upload all pattern-specific T-SNE plots
                                    pattern_names = {1: "O-tetromino", 2: "T-tetromino", 3: "L-tetromino"}
                                    for pattern_idx, fig_tsne_encoders_single in enumerate(fig_tsne_encoders_list, 1):
                                        if fig_tsne_encoders_single is not None:
                                            test_metrics[f"test/{dataset_dict['test_name']}/latents_encoders_pattern{pattern_idx}"] = wandb.Image(fig_tsne_encoders_single)
                                            logging.info(f"Logged T-SNE for pattern {pattern_idx} ({pattern_names[pattern_idx]})")
                                        else:
                                            logging.warning(f"No T-SNE plot available for pattern {pattern_idx}")
                                    
                                    wandb.log(test_metrics, step=step)
                                    plt.close('all')  # Close all figures to prevent memory leaks
                                    # Explicitly close additional T-SNE figures
                                    if fig_tsne_context is not None:
                                        plt.close(fig_tsne_context)
                                    # Close all pattern-specific T-SNE figures
                                    for fig_tsne_encoders_single in fig_tsne_encoders_list:
                                        if fig_tsne_encoders_single is not None:
                                            plt.close(fig_tsne_encoders_single)
                                    
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

    def evaluate(self, state: TrainState, enc_params_list: list[dict] = None) -> dict:
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
        # Use current encoder weights from state.params, not the original artifact weights
        current_enc_params_list = state.params["encoders"]
        logging.info(f"Evaluation using current encoder weights from training state (step {getattr(state, 'step', 'unknown')})")
        if not hasattr(self, "eval_grids"):
            return {}
        cfg = self.cfg
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
        
        # 1. IMPLEMENT LEAVE-ONE-OUT: Create leave_one_out versions like train.py
        # The issue is that make_leave_one_out is adding an extra dimension instead of reducing it
        # We need to manually create the leave_one_out data with correct shapes
        
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
                    {"params": state.params["decoder"]},
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
                    encoder_params_list=state.params["encoders"],  # encoder_params_list
                    decoder_params=state.params["decoder"],  # decoder_params
                    repulsion_kl_coeff=self.cfg.training.get("repulsion_kl"),  # repulsion_kl_coeff
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
        logging.info(f"Available info keys: {list(all_info[0].keys())}")
        for key in all_info[0].keys():
            if key == "context":
                # Concatenate contexts
                contexts = [inf[key] for inf in all_info]
                info[key] = jnp.concatenate(contexts, axis=0)
                logging.info(f"Context shape after concatenation: {info[key].shape}")
            else:
                # For other info, just take the first batch
                info[key] = all_info[0][key]
                logging.info(f"Info key '{key}' shape: {info[key].shape}")

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
        # Limit number of tasks shown for memory efficiency, but STRATIFY across patterns (1,2,3)
        num_show = int(cfg.eval.get("num_tasks_to_show", 5))
        num_show = max(1, min(num_show, int(pairs_np.shape[0])))
        # Build a local pattern sequence (assumes ordering by pattern with equal blocks)
        total_sets = int(pairs_np.shape[0])
        spp = max(1, total_sets // 3)
        pattern_sequence = np.concatenate([
            np.ones(spp, dtype=int),
            np.ones(spp, dtype=int) * 2,
            np.ones(total_sets - 2 * spp, dtype=int) * 3,
        ])
        # Determine per-pattern counts (at least 1 from each if possible)
        per_pat = max(1, num_show // 3)
        # Build indices for each pattern in a round-robin up to num_show
        pat_idxs = {1: [], 2: [], 3: []}
        for pid in (1, 2, 3):
            candidates = np.where(pattern_sequence == pid)[0]
            pat_idxs[pid] = list(candidates[:per_pat])
        selected = pat_idxs[1] + pat_idxs[2] + pat_idxs[3]
        # If we still need more (e.g., num_show not divisible by 3), append next available across patterns
        if len(selected) < num_show:
            extra_needed = num_show - len(selected)
            # Concatenate remaining candidates
            remaining = np.concatenate([
                np.where(pattern_sequence == 1)[0][per_pat:],
                np.where(pattern_sequence == 2)[0][per_pat:],
                np.where(pattern_sequence == 3)[0][per_pat:],
            ])
            if remaining.size > 0:
                selected.extend(list(remaining[:extra_needed]))
        selected = np.array(selected[:num_show], dtype=int)

        # Visualization expects predicted grids/shapes with per-pair axis; tile our single prediction across pairs
        num_pairs = int(shapes_np.shape[1])
        pred_grids_vis = np.repeat(out_grids_np[selected, None, ...], num_pairs, axis=1)
        pred_shapes_vis = np.repeat(out_shapes_np[selected, None, :], num_pairs, axis=1)
        fig_gen = visualize_dataset_generation(pairs_np[selected], shapes_np[selected], pred_grids_vis, pred_shapes_vis, len(selected))

        # 5. IMPLEMENT TSNE WITH ENCODERS + CONTEXT: Show both with different markers
        # We want to see: encoder outputs vs final generation context, both colored by PATTERN TYPE
        # This allows us to compare how the SAME PATTERN is represented across different sources
        all_latents = []
        source_ids = []  # 0, 1, 2 for encoders, 3 for generation context
        pattern_ids_list = []  # pattern types (1, 2, 3) for each point - same across all sources
        task_ids_list = []  # task indices so we can label points from the same task
        
        # Create pattern-based coloring: 32 samples per pattern (O, T, L tetrominos)
        num_sets = self.eval_grids.shape[0]  # Should be 96
        samples_per_pattern = num_sets // 3  # Should be 32
        
        # Pattern mapping: sets 0-31 = pattern 1 (O), sets 32-63 = pattern 2 (T), sets 64-95 = pattern 3 (L)
        pattern_sequence = np.concatenate([
            np.ones(samples_per_pattern, dtype=int),      # Pattern 1 (O-tetromino)
            np.ones(samples_per_pattern, dtype=int) * 2,  # Pattern 2 (T-tetromino) 
            np.ones(samples_per_pattern, dtype=int) * 3   # Pattern 3 (L-tetromino)
        ])
        
        logging.info(
            f"T-SNE pattern mapping: {samples_per_pattern} samples per pattern, total patterns: {np.unique(pattern_sequence)}"
        )

        # Task IDs: each of the num_sets tasks contributes one point per source
        task_id_sequence = np.arange(num_sets, dtype=int)
        
        # Add individual encoder latents (unique source_id per encoder)
        for enc_idx, enc_params in enumerate(current_enc_params_list):
            try:
                mu_i, logvar_i = self.encoders[enc_idx].apply(
                    {"params": enc_params}, 
                    pairs, 
                    shapes, 
                    True, 
                    mutable=False
                )
                lat = mu_i.mean(axis=-2)  # Mean over pairs
                lat_np = np.array(lat).reshape(-1, lat.shape[-1])
                
                # Log the actual latent dimension from this encoder
                actual_latent_dim = lat_np.shape[-1]
                logging.info(f"Main eval - Encoder {enc_idx} - Input pairs shape: {pairs.shape}, shapes shape: {shapes.shape}")
                logging.info(f"Main eval - Encoder {enc_idx} - Encoder params keys: {list(enc_params.keys())}")
                logging.info(f"Main eval - Encoder {enc_idx} - mu_i shape: {mu_i.shape}, logvar_i shape: {logvar_i.shape}")
                
                if actual_latent_dim != 32:
                    logging.warning(f"Main eval - Encoder {enc_idx} has unexpected latent dim: {actual_latent_dim}, expected 32")
                
                # Ensure consistent latent dimension for T-SNE
                if actual_latent_dim != 32:
                    # Pad or truncate to match expected dimension
                    if actual_latent_dim < 32:
                        # Pad with zeros
                        padding = np.zeros((lat_np.shape[0], 32 - actual_latent_dim))
                        lat_np = np.concatenate([lat_np, padding], axis=1)
                    else:
                        # Truncate
                        lat_np = lat_np[:, :32]
                
                logging.info(f"Main eval - Encoder {enc_idx} - final latent shape: {lat_np.shape}")
                all_latents.append(lat_np)
                source_ids.extend([enc_idx] * lat_np.shape[0])  # enc_idx for each encoder (0, 1, 2)
                pattern_ids_list.append(pattern_sequence)  # Same pattern sequence for each encoder
                task_ids_list.append(task_id_sequence)  # Same task IDs for each encoder output
                
            except Exception as e:
                logging.error(f"Main eval - Encoder {enc_idx} failed: {e}")
                continue
        
        # Add the generation context (source_id = num_encoders)
        if "context" in info:
            generation_context = info["context"]
            logging.info(f"Main eval - Found context in info, shape: {generation_context.shape}")
            if generation_context is not None:
                # Reshape like train.py does
                context_np = np.array(generation_context).reshape(-1, generation_context.shape[-1])
                
                # Log the context latent dimension
                context_latent_dim = context_np.shape[-1]
                logging.info(f"Main eval - Context latent dim: {context_latent_dim}")
                
                if context_latent_dim != 32:
                    logging.warning(f"Main eval - Context has unexpected latent dim: {context_latent_dim}, expected 32")
                    
                    # Ensure consistent latent dimension for T-SNE
                    if context_latent_dim < 32:
                        # Pad with zeros
                        padding = np.zeros((context_np.shape[0], 32 - context_latent_dim))
                        context_np = np.concatenate([context_np, padding], axis=1)
                    else:
                        # Truncate
                        context_np = context_np[:, :32]
                    
                    logging.info(f"Main eval - Context final latent shape: {context_np.shape}")
                
                all_latents.append(context_np)
                source_ids.extend([len(enc_params_list)] * context_np.shape[0])  # num_encoders for generation context
                pattern_ids_list.append(pattern_sequence)  # Same pattern sequence for context
                task_ids_list.append(task_id_sequence)  # Task IDs for context points
                logging.info(f"Main eval - Added context to T-SNE: {len(context_np)} points")
        else:
            logging.warning(f"Main eval - No 'context' key found in info. Available keys: {list(info.keys())}")
        
        if all_latents:
            latents_concat = np.concatenate(all_latents, axis=0)
            source_ids_np = np.array(source_ids)
            pattern_ids_concat = np.concatenate(pattern_ids_list, axis=0)
            task_ids_np = np.concatenate(task_ids_list, axis=0)
            
            # Log T-SNE structure: each pattern should have multiple sets with 4 points each (3 encoders + 1 context)
            total_points = latents_concat.shape[0]
            total_patterns = 3  # O, T, L tetrominos
            points_per_pattern = total_points // total_patterns
            logging.info(f"T-SNE structure: {total_points} total points, {total_patterns} patterns, {points_per_pattern} points per pattern")
            logging.info(f"Expected: {len(enc_params_list)} encoders + 1 context = {len(enc_params_list) + 1} points per set")
            logging.info(f"Generating 3 T-SNE visualizations: main (encoders+context), context-only, encoders-only (pattern 1)")
            
            # Downsample points for t-SNE to be memory efficient
            max_points = int(cfg.eval.get("tsne_max_points", 500))
            if latents_concat.shape[0] > max_points:
                # Simple random sampling while preserving pattern distribution
                # Since we have 3 patterns, ensure we keep some from each
                points_per_pattern = total_points // total_patterns
                max_points_per_pattern = max_points // total_patterns
                
                if max_points_per_pattern > 0:
                    # Sample from each pattern
                    point_indices = []
                    for pattern_idx in range(total_patterns):
                        # Find all points for this pattern
                        pattern_mask = pattern_ids_concat == (pattern_idx + 1)
                        pattern_point_indices = np.where(pattern_mask)[0]
                        
                        # Sample from this pattern
                        if len(pattern_point_indices) > max_points_per_pattern:
                            sampled_indices = np.random.RandomState(42).choice(
                                pattern_point_indices, 
                                size=max_points_per_pattern, 
                                replace=False
                            )
                        else:
                            sampled_indices = pattern_point_indices
                        
                        point_indices.extend(sampled_indices)
                    
                    # Apply sampling
                    latents_concat = latents_concat[point_indices]
                    source_ids_np = source_ids_np[point_indices]
                    pattern_ids_concat = pattern_ids_concat[point_indices]
                    task_ids_np = task_ids_np[point_indices]
                    
                    logging.info(f"T-SNE downsampled: {len(point_indices)} points, maintaining pattern distribution")
                else:
                    logging.warning(f"T-SNE max_points too small to sample from all patterns")
            
            # Use visualize_tsne_sources to show different markers for encoders vs context
            fig_tsne = visualize_tsne_sources(
                latents=latents_concat,
                program_ids=pattern_ids_concat,  # Pattern types (1, 2, 3) for colors
                source_ids=source_ids_np,        # 0,1,2 for encoders, 3 for context
                max_points=max_points,
                random_state=42,
                task_ids=task_ids_np,
            )
            
            # 1. ADDITIONAL T-SNE: Show just the context latents (with samples from the 3 patterns)
            context_mask = (source_ids_np == (len(enc_params_list)))
            if np.any(context_mask):
                context_latents = latents_concat[context_mask]
                context_patterns = pattern_ids_concat[context_mask]
                context_task_ids = task_ids_np[context_mask]
                
                # Downsample context points for cleaner visualization
                max_context_points = min(500, len(context_latents))
                if len(context_latents) > max_context_points:
                    # Stratified sampling to maintain pattern distribution
                    context_indices = []
                    for pattern_id in [1, 2, 3]:
                        pattern_mask = context_patterns == pattern_id
                        pattern_indices = np.where(pattern_mask)[0]
                        if len(pattern_indices) > 0:
                            # Sample up to max_context_points // 3 from each pattern
                            max_per_pattern = max_context_points // 3
                            if len(pattern_indices) > max_per_pattern:
                                sampled_indices = np.random.RandomState(42).choice(
                                    pattern_indices, size=max_per_pattern, replace=False
                                )
                            else:
                                sampled_indices = pattern_indices
                            context_indices.extend(sampled_indices)
                    
                    # Apply sampling
                    context_latents = context_latents[context_indices]
                    context_patterns = context_patterns[context_indices]
                    context_task_ids = context_task_ids[context_indices]
                
                # Create T-SNE for context-only latents
                # Use source_id = 0 for all points (will show as same marker type)
                context_source_ids = np.zeros(len(context_latents), dtype=int)
                
                fig_tsne_context = visualize_tsne_sources(
                    latents=context_latents,
                    program_ids=context_patterns,  # Pattern types (1, 2, 3) for colors
                    source_ids=context_source_ids,  # All 0s (same marker type)
                    max_points=max_context_points,
                    random_state=42,
                    task_ids=context_task_ids,
                )
                
                logging.info(f"Generated context-only T-SNE: {len(context_latents)} points")
            else:
                fig_tsne_context = None
                logging.warning("No context points found for context-only T-SNE")
            
            # 2. ADDITIONAL T-SNE: Show just the 3 encoders latents for EACH pattern
            # Generate one T-SNE plot for each pattern (1, 2, 3)
            fig_tsne_encoders_list = []
            
            for target_pattern in [1, 2, 3]:
                pattern_mask = (pattern_ids_concat == target_pattern)
                
                if np.any(pattern_mask):
                    # Get encoder points only (exclude context)
                    encoder_mask = (source_ids_np < len(enc_params_list))
                    combined_mask = pattern_mask & encoder_mask
                    
                    if np.any(combined_mask):
                        encoder_latents = latents_concat[combined_mask]
                        encoder_sources = source_ids_np[combined_mask]
                        encoder_task_ids = task_ids_np[combined_mask]
                        
                        # Downsample encoder points for cleaner visualization
                        max_encoder_points = min(300, len(encoder_latents))
                        if len(encoder_latents) > max_encoder_points:
                            # Stratified sampling to maintain encoder distribution
                            encoder_indices = []
                            for enc_id in range(len(enc_params_list)):
                                enc_mask = encoder_sources == enc_id
                                enc_indices = np.where(enc_mask)[0]
                                if len(enc_indices) > 0:
                                    # Sample up to max_encoder_points // num_encoders from each encoder
                                    max_per_encoder = max_encoder_points // len(enc_params_list)
                                    if len(enc_indices) > max_per_encoder:
                                        sampled_indices = np.random.RandomState(42).choice(
                                            enc_indices, size=max_per_encoder, replace=False
                                        )
                                    else:
                                        sampled_indices = enc_indices
                                    encoder_indices.extend(sampled_indices)
                            
                            # Apply sampling
                            encoder_latents = encoder_latents[encoder_indices]
                            encoder_sources = encoder_sources[encoder_indices]
                            encoder_task_ids = encoder_task_ids[encoder_indices]
                        
                        # Create T-SNE for encoder-only latents (specific pattern)
                        # Use pattern_id = target_pattern for all points (will show as same color)
                        encoder_patterns = np.full(len(encoder_latents), target_pattern, dtype=int)
                        
                        # Create custom title for this pattern-specific T-SNE
                        pattern_names = {1: "O-tetromino", 2: "T-tetromino", 3: "L-tetromino"}
                        custom_title = f"t-SNE Visualisation of Latent Embeddings: Pattern {target_pattern}"
                        
                        # Create a custom T-SNE visualization for pattern-specific plots with source color coding
                        fig_tsne_encoders_single = self._create_pattern_specific_tsne(
                            latents=encoder_latents,
                            source_ids=encoder_sources,    # 0,1,2 for different encoders
                            task_ids=encoder_task_ids,
                            title=custom_title,
                            max_points=max_encoder_points,
                            random_state=42
                        )
                        
                        fig_tsne_encoders_list.append(fig_tsne_encoders_single)
                        logging.info(f"Generated encoder-only T-SNE (pattern {target_pattern}): {len(encoder_latents)} points")
                    else:
                        fig_tsne_encoders_list.append(None)
                        logging.warning(f"No encoder points found for pattern {target_pattern}")
                else:
                    fig_tsne_encoders_list.append(None)
                    logging.warning(f"No points found for pattern {target_pattern}")
            
            # For backward compatibility, keep the first pattern T-SNE as the main one
            fig_tsne_encoders = fig_tsne_encoders_list[0] if fig_tsne_encoders_list else None
            
            # COMPUTE CLUSTERING METRICS AND UPLOAD TO WANDB
            try:
                # Compute metrics for different k values to check sensitivity
                k_values = [3, 5, 10]
                clustering_metrics = {}
                
                # OPTION 1: Context-only clustering (like train.py) - for direct comparison
                context_mask = (source_ids_np == (len(enc_params_list)))
                if np.any(context_mask):
                    ctx_emb = latents_concat[context_mask]
                    ctx_prog = pattern_ids_concat[context_mask]
                    logging.info(f"Context-only clustering: {ctx_emb.shape[0]} points, patterns: {np.unique(ctx_prog)}")
                    
                    for k in k_values:
                        # Modularity Q on context-only (comparable to train.py)
                        modularity_q = compute_modularity_q(ctx_emb, ctx_prog, k=k)
                        clustering_metrics[f"clustering/context/modularity_q_k{k}"] = modularity_q
                        
                        # Adjusted Rand Index on context-only (comparable to train.py)
                        ari_score = compute_adjusted_rand_index(ctx_emb, ctx_prog, k=k)
                        clustering_metrics[f"clustering/context/ari_k{k}"] = ari_score
                else:
                    logging.warning("No context points found for context-only clustering; skipping")
                
                # OPTION 2: Full latent space clustering (current implementation) - for source analysis
                for k in k_values:
                    # Modularity Q on all embeddings (sources: encoders vs context)
                    modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"clustering/source/modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index on all embeddings (sources: encoders vs context)
                    ari_score = compute_adjusted_rand_index(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"clustering/source/ari_k{k}"] = ari_score
                
                # Log clustering metrics to WandB
                wandb.log(clustering_metrics, step=step if 'step' in locals() else None)
                logging.info(f"Clustering metrics computed: {clustering_metrics}")
                
            except Exception as e:
                logging.warning(f"Clustering metrics computation failed: {e}")
        else:
            fig_tsne = None

        # Log all T-SNE plots to wandb
        wandb_log_data = {
            f"test/{test_name}/pixel_accuracy": wandb.Image(fig_heatmap),
            f"test/{test_name}/generation": wandb.Image(fig_gen),
            f"test/{test_name}/latents": wandb.Image(fig_tsne),
            f"test/{test_name}/latents_context_only": wandb.Image(fig_tsne_context) if fig_tsne_context is not None else None,
            **metrics,
        }
        
        # Log all pattern-specific T-SNE plots
        pattern_names = {1: "O-tetromino", 2: "T-tetromino", 3: "L-tetromino"}
        for pattern_idx, fig_tsne_encoders_single in enumerate(fig_tsne_encoders_list, 1):
            if fig_tsne_encoders_single is not None:
                wandb_log_data[f"test/{test_name}/latents_encoders_pattern{pattern_idx}"] = wandb.Image(fig_tsne_encoders_single)
                logging.info(f"Logged T-SNE for pattern {pattern_idx} ({pattern_names[pattern_idx]})")
            else:
                logging.warning(f"No T-SNE plot available for pattern {pattern_idx}")
        
        # For backward compatibility, also log the first pattern T-SNE as the main one
        if fig_tsne_encoders is not None:
            wandb_log_data[f"test/{test_name}/latents_encoders_pattern1"] = wandb.Image(fig_tsne_encoders)
        
        wandb.log(wandb_log_data)

        # NEW: Confidence panel per pattern (one task per pattern)
        try:
            # Select one example index per pattern
            panel_indices = []
            for pid in (1, 2, 3):
                pid_idxs = np.where(pattern_sequence == pid)[0]
                if pid_idxs.size > 0:
                    panel_indices.append(int(pid_idxs[0]))
            # For each selected example, compute encoder means/vars and PoE aggregation on encoder latents
            for pid, idx in zip((1, 2, 3), panel_indices):
                # Build per-encoder latents for this single task
                enc_mus = []
                enc_logvars = []
                for enc_idx, enc_params in enumerate(current_enc_params_list):
                    mu_i, logvar_i = self.encoders[enc_idx].apply(
                        {"params": enc_params}, 
                        pairs[idx:idx+1], 
                        shapes[idx:idx+1], 
                        True, 
                        mutable=False
                    )
                    # Use QUERY PAIR ONLY (pair index 0) -> shape [1, D]
                    mu_i_np = np.array(mu_i)
                    logvar_i_np = np.array(logvar_i)
                    enc_mus.append(mu_i_np[:, 0, :])
                    enc_logvars.append(logvar_i_np[:, 0, :])

                # Compute PoE (precision-weighted) from encoder posteriors for the combined curve
                alphas_np = np.asarray(alphas)
                precisions = [np.exp(-lv) for lv in enc_logvars]  # [1, D]
                poe_precision = np.zeros_like(precisions[0])
                for a, p in zip(alphas_np, precisions):
                    poe_precision = poe_precision + a * p
                poe_var = 1.0 / (poe_precision + 1e-8)
                num = np.zeros_like(enc_mus[0])
                for a, p, m in zip(alphas_np, precisions, enc_mus):
                    num = num + a * p * m
                poe_mu = num / (poe_precision + 1e-8)
                poe_logvar = np.log(poe_var + 1e-8)

                panel_title = f"Pattern {pid} - Confidence"
                enc_labels = [f"Encoder {i}" for i in range(len(enc_mus))]
                fig_panel = visualize_struct_confidence_panel(
                    sample_grids=pairs_np[idx],
                    sample_shapes=shapes_np[idx],
                    encoder_mus=[em.squeeze(0) for em in enc_mus],
                    encoder_logvars=[ev.squeeze(0) for ev in enc_logvars],
                    poe_mu=poe_mu.squeeze(0),
                    poe_logvar=poe_logvar.squeeze(0),
                    title=panel_title,
                    encoder_labels=enc_labels,
                    combined_label="PoE",
                )
                wandb.log({f"test/{test_name}/confidence_panel/pattern_{pid}": wandb.Image(fig_panel)})
                plt.close(fig_panel)
        except Exception as e:
            logging.warning(f"Confidence panel generation failed: {e}")

        # Free figures to save memory
        plt.close(fig_heatmap)
        plt.close(fig_gen)
        plt.close(fig_tsne)
        if fig_tsne_context is not None:
            plt.close(fig_tsne_context)
        if fig_tsne_encoders is not None:
            plt.close(fig_tsne_encoders)

        # Release large intermediates
        del all_latents, latents_concat, source_ids_np, pattern_ids_concat
        return metrics

    def _create_pattern_specific_tsne(
        self,
        latents: np.ndarray,
        source_ids: np.ndarray,
        task_ids: np.ndarray,
        title: str,
        max_points: int = 300,
        random_state: int = 42
    ) -> Optional[plt.Figure]:
        """
        Create a custom T-SNE visualization for pattern-specific plots with source color coding.
        
        This method creates T-SNE plots that match EXACTLY the style of visualize_tsne_sources:
        - Same color palette, size, shapes, legend title, title style, axes style
        - All points have the same pattern (same color)
        - Different sources (encoders) have different colors and markers
        - EXACTLY matches test/structured_mean/latents_context_only styling
        
        Args:
            latents: [N, D] array of latent embeddings
            source_ids: [N] array of source IDs (0, 1, 2 for encoders)
            task_ids: [N] array of task IDs
            title: Title for the T-SNE plot
            max_points: Maximum number of points to show
            random_state: Random state for T-SNE
            
        Returns:
            matplotlib Figure with the T-SNE visualization
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            logging.warning("sklearn or matplotlib not available for T-SNE visualization")
            return None
        
        # Downsample if needed
        if len(latents) > max_points:
            indices = np.random.RandomState(random_state).choice(
                len(latents), size=max_points, replace=False
            )
            latents = latents[indices]
            source_ids = source_ids[indices]
            task_ids = task_ids[indices]
        
        # Perform T-SNE - EXACTLY like visualize_tsne_sources
        tsne = TSNE(n_components=2, perplexity=2, max_iter=1000, random_state=random_state)
        latents_2d = tsne.fit_transform(latents)
        
        # Create figure - EXACTLY like visualize_tsne_sources
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Use the EXACT SAME color scheme and markers as visualize_tsne_sources
        # For pattern-specific T-SNE: all points have the same pattern, so use encoder colors
        source_colors = {
            0: '#FBB998',  # Encoder 0 - ORANGE (same as visualize_tsne_sources)
            1: '#DB74DB',  # Encoder 1 - PINK (same as visualize_tsne_sources)
            2: '#5361E5'   # Encoder 2 - blue (same as visualize_tsne_sources)

        }
        source_markers = {
            0: 'o',    # Encoder 0 - Circle (same as visualize_tsne_sources)
            1: 's',    # Encoder 1 - Square (same as visualize_tsne_sources)
            2: '^'     # Encoder 2 - Triangle (same as visualize_tsne_sources)
        }
        source_labels = {
            0: "Encoder 0",
            1: "Encoder 1", 
            2: "Encoder 2"
        }
        
        # Plot points for each source - EXACTLY like visualize_tsne_sources
        unique_sources = sorted(list(np.unique(source_ids)))
        
        for source_id in unique_sources:
            mask = source_ids == source_id
            if np.any(mask):
                color = source_colors.get(source_id, '#AAAAAA')
                marker = source_markers.get(source_id, 'o')
                
                # Plot points for this source - EXACTLY like visualize_tsne_sources
                ax.scatter(
                    latents_2d[mask, 0], 
                    latents_2d[mask, 1],
                    c=[color], 
                    marker=marker,
                    alpha=0.7,
                    s=100,
                    edgecolors='none'
                )
        
        # Set title and labels - EXACTLY like visualize_tsne_sources
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        
        # Build legend - EXACTLY like visualize_tsne_sources
        shape_handles = []
        for src in unique_sources:
            marker = source_markers.get(src, 'o')
            label = source_labels.get(src, f"Source {src}")
            shape_handles.append(
                Line2D([0], [0], marker=marker, linestyle='None', color='black',
                       markerfacecolor='white', markeredgecolor='black', markersize=10, label=label)
            )
        
        # Add legend - EXACTLY like visualize_tsne_sources
        ax.legend(handles=shape_handles, bbox_to_anchor=(1.05, 1), loc="upper left", 
                  borderaxespad=0.0, title="Sources (shape)")
        
        # Set tight layout - EXACTLY like visualize_tsne_sources
        plt.tight_layout()
        
        return fig
        
    def test_dataset_submission(
        self,
        state: TrainState,
        enc_params_list: list[dict] = None,
        test_name: str = None,
        dataset_grids: chex.Array = None,
        dataset_shapes: chex.Array = None,
        program_ids: Optional[chex.Array] = None,
        batch_size: int = None,
        num_tasks_to_show: int = 5,
        inference_mode: str = "mean",
        inference_kwargs: dict = None,
    ) -> tuple[dict[str, float], Optional[plt.Figure], plt.Figure, Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure], list[Optional[plt.Figure]]]:
        """
        Test dataset submission method for structured training (similar to train.py).
        Generates outputs using leave-one-out approach and computes metrics.
        
        Returns:
            - A dictionary containing the metrics.
            - A figure containing the visualization of the generated grids.
            - A figure containing the visualization of the pixel accuracy heatmap.
            - A figure containing the visualization of the latents (T-SNE).
            - A figure containing the visualization of the latents samples (None for structured training).
            - A figure containing the visualization of the search progress (None if not applicable).
            - A figure containing the visualization of the context-only T-SNE.
            - A list of figures containing the visualization of the encoder-only T-SNE for each pattern.
        """
        # Use current encoder weights from state.params, not the original artifact weights
        current_enc_params_list = state.params["encoders"]
        
        # Extract parameters from dataset_dict if called from main training loop
        if test_name is None and hasattr(self, 'test_datasets'):
            # This is a call from the main training loop, extract from dataset_dict
            dataset_dict = enc_params_list  # enc_params_list is actually dataset_dict here
            test_name = dataset_dict["test_name"]
            dataset_grids = dataset_dict["dataset_grids"]
            dataset_shapes = dataset_dict["dataset_shapes"]
            program_ids = dataset_dict.get("program_ids")
            batch_size = dataset_dict["batch_size"]
            num_tasks_to_show = dataset_dict.get("num_tasks_to_show", 5)
            inference_mode = dataset_dict.get("inference_mode", "mean")
            inference_kwargs = dataset_dict.get("inference_kwargs", {})
            enc_params_list = None  # Will use current_enc_params_list
        
        if inference_kwargs is None:
            inference_kwargs = {}
            
        # Define alphas for PoE (same as main evaluation)
        alphas = jnp.asarray(self.cfg.structured.alphas, dtype=jnp.float32)
            
        # Create leave_one_out data
        raw_leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
        raw_leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)

        # make_leave_one_out currently returns data with an extra dimension
        # (L, N, N-1, ...). Slice away the redundant axis to match the
        # expected shapes used by the model just like we do during evaluation.
        if raw_leave_one_out_grids.shape[1] == dataset_grids.shape[1]:
            leave_one_out_grids = raw_leave_one_out_grids[:, 0, ...]
            leave_one_out_shapes = raw_leave_one_out_shapes[:, 0, ...]
        else:
            leave_one_out_grids = raw_leave_one_out_grids
            leave_one_out_shapes = raw_leave_one_out_shapes
        
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
                    {"params": state.params["decoder"]},
                    method=self.model.generate_output,
                    pairs=batch_leave_one_out_grids,
                    grid_shapes=batch_leave_one_out_shapes,
                    input=batch_grids[:, 0, ..., 0],
                    input_grid_shape=batch_shapes[:, 0, ..., 0],
                    key=key,
                    dropout_eval=True,
                    mode=inference_mode,
                    return_two_best=False,
                    poe_alphas=alphas,  # Use the same alphas as main evaluation
                    encoder_params_list=state.params["encoders"],
                    decoder_params=state.params["decoder"],
                    repulsion_kl_coeff=self.cfg.training.get("repulsion_kl"),  # repulsion_kl_coeff
                )
                
                # Normalize context to 2D (num_points, latent_dim) to avoid concat shape mismatches
                # And drop any extra keys to mimic train.py behavior (only 'context')
                normalized_batch_info = {}
                if "context" in batch_info and batch_info["context"] is not None:
                    import numpy as _np
                    try:
                        ctx_np = _np.asarray(batch_info["context"])  # force numpy
                        ctx_np = ctx_np.reshape(-1, int(ctx_np.shape[-1]))  # unconditional 2D reshape
                        normalized_batch_info["context"] = ctx_np
                        logging.info(f"Test batch {i} - context normalized to {ctx_np.shape}")
                    except Exception as e:
                        logging.error(f"Test batch {i} - context normalization failed: {e}; dropping context for this batch")
                # Replace batch_info with only the normalized context (drop other keys)
                batch_info = normalized_batch_info
                
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
        
        # Merge info: only 'context' (match train.py behavior)
        info = {}
        import numpy as _np
        contexts = []
        for b_idx, inf in enumerate(all_info):
            ctx = inf.get("context")
            if ctx is None:
                logging.warning(f"Test - batch {b_idx} has None context; skipping")
                continue
            try:
                ctx_np = _np.asarray(ctx).reshape(-1, int(_np.asarray(ctx).shape[-1]))
                logging.info(f"Test - batch {b_idx} context ready for concat: shape={ctx_np.shape}")
                contexts.append(ctx_np)
            except Exception as e:
                logging.error(f"Test - batch {b_idx} context reshape failed: {e}; skipping this batch context")
                continue
        if contexts:
            try:
                merged = _np.concatenate(contexts, axis=0)
                logging.info(f"Test merged context shape: {merged.shape}")
                info["context"] = jnp.asarray(merged)
            except Exception as e:
                logging.error(f"Test - Failed to concatenate contexts: {e}; dropping context from info")
        else:
            logging.info("Test - No context tensors to merge")
        
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
                pattern_ids_list = []

                # Use provided program IDs to respect per-task patterns
                pattern_ids_np = np.array(program_ids)
                logging.info(
                    f"Test dataset pattern types: {np.unique(pattern_ids_np)}"
                )

                task_id_sequence = np.arange(dataset_grids.shape[0], dtype=int)
                task_ids_list = []

                # Add encoder outputs (unique source_id per encoder)
                for enc_idx, enc_params in enumerate(current_enc_params_list):
                    try:
                        mu_i, logvar_i = self.encoders[enc_idx].apply(
                            {"params": enc_params}, 
                            dataset_grids, 
                            dataset_shapes, 
                            True, 
                            mutable=False
                        )
                        lat = mu_i.mean(axis=-2)  # Mean over pairs
                        lat_np = np.array(lat).reshape(-1, lat.shape[-1])
                        
                        # Log the actual latent dimension from this encoder
                        actual_latent_dim = lat_np.shape[-1]
                        logging.info(f"Test eval - Encoder {enc_idx} - mu_i shape: {mu_i.shape}, logvar_i shape: {logvar_i.shape}")
                        
                        if actual_latent_dim != 32:
                            logging.warning(f"Test eval - Encoder {enc_idx} has unexpected latent dim: {actual_latent_dim}, expected 32")
                        
                        # Ensure consistent latent dimension for T-SNE
                        if actual_latent_dim != 32:
                            # Pad or truncate to match expected dimension
                            if actual_latent_dim < 32:
                                # Pad with zeros
                                padding = np.zeros((lat_np.shape[0], 32 - actual_latent_dim))
                                lat_np = np.concatenate([lat_np, padding], axis=1)
                            else:
                                # Truncate
                                lat_np = lat_np[:, :32]
                        
                        logging.info(f"Test eval - Encoder {enc_idx} - final latent shape: {lat_np.shape}")
                        all_latents.append(lat_np)
                        source_ids.extend([enc_idx] * lat_np.shape[0])  # enc_idx for each encoder (0, 1, 2)
                        pattern_ids_list.append(pattern_ids_np)
                        task_ids_list.append(task_id_sequence)
                        
                    except Exception as e:
                        logging.error(f"Test eval - Encoder {enc_idx} failed: {e}")
                        continue
                
                # Add generation context (source_id = num_encoders)
                context_np = np.array(context).reshape(-1, context.shape[-1])
                
                # Log the context latent dimension
                context_latent_dim = context_np.shape[-1]
                logging.info(f"Test eval - Context latent dim: {context_latent_dim}")
                
                if context_latent_dim != 32:
                    logging.warning(f"Test eval - Context has unexpected latent dim: {context_latent_dim}, expected 32")
                    
                    # Ensure consistent latent dimension for T-SNE
                    if context_latent_dim < 32:
                        # Pad with zeros
                        padding = np.zeros((context_np.shape[0], 32 - context_latent_dim))
                        context_np = np.concatenate([context_np, padding], axis=1)
                    else:
                        # Truncate
                        context_np = context_np[:, :32]
                    
                    logging.info(f"Test eval - Context final latent shape: {context_np.shape}")
                
                all_latents.append(context_np)
                source_ids.extend([len(enc_params_list)] * context_np.shape[0])  # num_encoders for context
                pattern_ids_list.append(pattern_ids_np)
                task_ids_list.append(task_id_sequence)
                
                if all_latents:
                    latents_concat = np.concatenate(all_latents, axis=0)
                    source_ids_np = np.array(source_ids)
                    pattern_ids_concat = np.concatenate(pattern_ids_list, axis=0)
                    task_ids_np = np.concatenate(task_ids_list, axis=0)
                    
                    # Log T-SNE structure for test datasets
                    total_points = latents_concat.shape[0]
                    unique_patterns = np.unique(pattern_ids_concat)
                    pattern_counts = {int(p): int((pattern_ids_concat == p).sum()) for p in unique_patterns}
                    logging.info(
                        f"Test T-SNE structure: {total_points} total points, {len(unique_patterns)} patterns, counts per pattern: {pattern_counts}"
                    )
                    logging.info(f"Expected: {len(enc_params_list)} encoders + 1 context = {len(enc_params_list) + 1} points per set")
                    logging.info(f"Test: Generating 3 T-SNE visualizations: main (encoders+context), context-only, encoders-only (single pattern)")
                    
                    # Use visualize_tsne_sources for different markers
                    fig_latents = visualize_tsne_sources(
                        latents=latents_concat,
                        program_ids=pattern_ids_concat,
                        source_ids=source_ids_np,
                        max_points=500,
                        random_state=42,
                        task_ids=task_ids_np,
                    )
                    
                    # 1. ADDITIONAL T-SNE: Show just the context latents (with samples from the 3 patterns)
                    context_mask = (source_ids_np == (len(enc_params_list)))
                    if np.any(context_mask):
                        context_latents = latents_concat[context_mask]
                        context_patterns = pattern_ids_concat[context_mask]
                        context_task_ids = task_ids_np[context_mask]
                        
                        # Downsample context points for cleaner visualization
                        max_context_points = min(300, len(context_latents))
                        if len(context_latents) > max_context_points:
                            # Stratified sampling to maintain pattern distribution
                            context_indices = []
                            for pattern_id in np.unique(context_patterns):
                                pattern_mask = context_patterns == pattern_id
                                pattern_indices = np.where(pattern_mask)[0]
                                if len(pattern_indices) > 0:
                                    # Sample up to max_context_points // num_patterns from each pattern
                                    max_per_pattern = max_context_points // len(np.unique(context_patterns))
                                    if len(pattern_indices) > max_per_pattern:
                                        sampled_indices = np.random.RandomState(42).choice(
                                            pattern_indices, size=max_per_pattern, replace=False
                                        )
                                    else:
                                        sampled_indices = pattern_indices
                                    context_indices.extend(sampled_indices)
                            
                            # Apply sampling
                            context_latents = context_latents[context_indices]
                            context_patterns = context_patterns[context_indices]
                            context_task_ids = context_task_ids[context_indices]
                        
                        # Create T-SNE for context-only latents
                        # Use source_id = 0 for all points (will show as same marker type)
                        context_source_ids = np.zeros(len(context_latents), dtype=int)
                        
                        fig_tsne_context = visualize_tsne_sources(
                            latents=context_latents,
                            program_ids=context_patterns,  # Pattern types for colors
                            source_ids=context_source_ids,  # All 0s (same marker type)
                            max_points=max_context_points,
                            random_state=42,
                            task_ids=context_task_ids,
                        )
                        
                        logging.info(f"Test: Generated context-only T-SNE: {len(context_latents)} points")
                    else:
                        fig_tsne_context = None
                        logging.warning("Test: No context points found for context-only T-SNE")
                    
                    # 2. ADDITIONAL T-SNE: Show just the 3 encoders latents for EACH pattern
                    # Generate one T-SNE plot for each pattern
                    fig_tsne_encoders_list = []
                    available_patterns = np.unique(pattern_ids_concat)
                    
                    for target_pattern in available_patterns:
                        pattern_mask = (pattern_ids_concat == target_pattern)
                        
                        if np.any(pattern_mask):
                            # Get encoder points only (exclude context)
                            encoder_mask = (source_ids_np < len(enc_params_list))
                            combined_mask = pattern_mask & encoder_mask
                            
                            if np.any(combined_mask):
                                encoder_latents = latents_concat[combined_mask]
                                encoder_sources = source_ids_np[combined_mask]
                                encoder_task_ids = task_ids_np[combined_mask]
                                
                                # Downsample encoder points for cleaner visualization
                                max_encoder_points = min(200, len(encoder_latents))
                                if len(encoder_latents) > max_encoder_points:
                                    # Stratified sampling to maintain encoder distribution
                                    encoder_indices = []
                                    for enc_id in range(len(enc_params_list)):
                                        enc_mask = encoder_sources == enc_id
                                        enc_indices = np.where(enc_mask)[0]
                                        if len(enc_indices) > 0:
                                            # Sample up to max_encoder_points // num_encoders from each encoder
                                            max_per_encoder = max_encoder_points // len(enc_params_list)
                                            if len(enc_indices) > max_per_encoder:
                                                sampled_indices = np.random.RandomState(42).choice(
                                                    enc_indices, size=max_per_encoder, replace=False
                                                )
                                            else:
                                                sampled_indices = enc_indices
                                            encoder_indices.extend(sampled_indices)
                                    
                                    # Apply sampling
                                    encoder_latents = encoder_latents[encoder_indices]
                                    encoder_sources = encoder_sources[encoder_indices]
                                    encoder_task_ids = encoder_task_ids[encoder_indices]
                                
                                # Create T-SNE for encoder-only latents (specific pattern)
                                # Use pattern_id = target_pattern for all points (will show as same color)
                                encoder_patterns = np.full(len(encoder_latents), target_pattern, dtype=int)
                                
                                # Create custom title for this pattern-specific T-SNE
                                custom_title = f"t-SNE Visualisation of Latent Embeddings: Pattern {target_pattern}"
                                
                                # Create a custom T-SNE visualization for pattern-specific plots with source color coding
                                fig_tsne_encoders_single = self._create_pattern_specific_tsne(
                                    latents=encoder_latents,
                                    source_ids=encoder_sources,    # 0,1,2 for different encoders
                                    task_ids=encoder_task_ids,
                                    title=custom_title,
                                    max_points=max_encoder_points,
                                    random_state=42
                                )
                                
                                fig_tsne_encoders_list.append(fig_tsne_encoders_single)
                                logging.info(f"Test: Generated encoder-only T-SNE (pattern {target_pattern}): {len(encoder_latents)} points")
                            else:
                                fig_tsne_encoders_list.append(None)
                                logging.warning(f"Test: No encoder points found for pattern {target_pattern}")
                        else:
                            fig_tsne_encoders_list.append(None)
                            logging.warning(f"Test: No points found for pattern {target_pattern}")
                    
                    # For backward compatibility, keep the first pattern T-SNE as the main one
                    fig_tsne_encoders = fig_tsne_encoders_list[0] if fig_tsne_encoders_list else None
                    
                    # COMPUTE CLUSTERING METRICS FOR TEST DATASETS
                    try:
                        # Compute metrics for different k values
                        k_values = [3, 5, 10]
                        test_clustering_metrics = {}
                        
                        # OPTION 1: Context-only clustering (like train.py) - for direct comparison
                        context_mask = (source_ids_np == (len(enc_params_list)))
                        if np.any(context_mask):
                            ctx_emb = latents_concat[context_mask]
                            ctx_prog = pattern_ids_concat[context_mask]
                            logging.info(f"Test context-only clustering: {ctx_emb.shape[0]} points, patterns: {np.unique(ctx_prog)}")
                            
                            for k in k_values:
                                # Modularity Q on context-only (comparable to train.py)
                                modularity_q = compute_modularity_q(ctx_emb, ctx_prog, k=k)
                                test_clustering_metrics[f"clustering/{test_name}/context/modularity_q_k{k}"] = modularity_q
                                
                                # Adjusted Rand Index on context-only (comparable to train.py)
                                ari_score = compute_adjusted_rand_index(ctx_emb, ctx_prog, k=k)
                                test_clustering_metrics[f"clustering/{test_name}/context/ari_k{k}"] = ari_score
                        else:
                            logging.warning(f"Test: No context points found for context-only clustering; skipping")
                        
                        # OPTION 2: Full latent space clustering (current implementation) - for source analysis
                        for k in k_values:
                            # Modularity Q on all embeddings (sources: encoders vs context)
                            modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                            test_clustering_metrics[f"clustering/{test_name}/source/modularity_q_k{k}"] = modularity_q
                            
                            # Adjusted Rand Index on all embeddings (sources: encoders vs context)
                            ari_score = compute_adjusted_rand_index(latents_concat, source_ids_np, k=k)
                            test_clustering_metrics[f"clustering/{test_name}/ari_k{k}"] = ari_score
                        
                        # Add clustering metrics to the main metrics dict
                        metrics.update(test_clustering_metrics)
                        logging.info(f"Test clustering metrics computed: {test_clustering_metrics}")
                        
                    except Exception as e:
                        logging.warning(f"Test clustering metrics computation failed: {e}")
        
        return metrics, fig_gen, fig_heatmap, fig_latents, None, fig_search_progress, fig_tsne_context, fig_tsne_encoders_list


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


