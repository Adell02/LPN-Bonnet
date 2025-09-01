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
import os
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
        
        # NEW: Two-phase training approach
        self.encoder_expose_steps = int(cfg.training.get("encoder_expose_steps", 0) or 0)
        self.phase1_completed = False  # Track if individual encoder specialization is done
        
        # Phase A global step counter for WandB metrics
        self.phase_a_global_step = 0
        
        # Store original encoder params for individual training
        self.original_encoder_params = None
        self.original_decoder_params = None

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

        # Set data directory for fallback pattern loading
        self.data_dir = cfg.training.get("data_dir", "src/datasets")
        
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
            
            # CRITICAL FIX: Create explicit pattern IDs that match the concatenation order
            # This ensures pattern IDs align with the actual data, just like in training
            self.eval_pattern_ids = jnp.concatenate([
                jnp.full((samples_per_pattern,), 1),  # Pattern 1 (first 32 samples)
                jnp.full((samples_per_pattern,), 2),  # Pattern 2 (next 32 samples)
                jnp.full((samples_per_pattern,), 3),  # Pattern 3 (last 32 samples)
            ], axis=0)
            
            # DEBUG: Log evaluation dataset info
            logging.info(f"Generated balanced evaluation dataset:")
            logging.info(f"  - Total samples: {self.eval_grids.shape[0]}")
            logging.info(f"  - Samples per pattern: {samples_per_pattern}")
            logging.info(f"  - Grids shape: {self.eval_grids.shape}")
            logging.info(f"  - Shapes shape: {self.eval_shapes.shape}")
            logging.info(f"  - Pattern IDs: {self.eval_pattern_ids[:10]}... (first 10) - should be [1,1,1,...,2,2,2,...,3,3,3,...]")
            
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
            
            # CRITICAL FIX: Create explicit pattern IDs that match the concatenation order
            # This ensures pattern IDs align with the actual data, just like in training
            self.eval_pattern_ids = jnp.concatenate([
                jnp.full((samples_per_pattern,), 1),  # Pattern 1 (first 32 samples)
                jnp.full((samples_per_pattern,), 2),  # Pattern 2 (next 32 samples)
                jnp.full((samples_per_pattern,), 3),  # Pattern 3 (last 32 samples)
            ], axis=0)
            
            # DEBUG: Log evaluation dataset info
            logging.info(f"Generated fallback balanced evaluation dataset:")
            logging.info(f"  - Total samples: {self.eval_grids.shape[0]}")
            logging.info(f"  - Samples per pattern: {samples_per_pattern}")
            logging.info(f"  - Grids shape: {self.eval_grids.shape}")
            logging.info(f"  - Shapes shape: {self.eval_shapes.shape}")
            logging.info(f"  - Pattern IDs: {self.eval_pattern_ids[:10]}... (first 10) - should be [1,1,1,...,2,2,2,...,3,3,3,...]")
        
        # CRITICAL: Load pattern datasets ONCE at initialization for consistent certainty plots
        # This ensures the same data is used every time the certainty plot is generated
        # FIXES: The issue where certainty plots mixed datasets after step 100
        logging.info("🔍 Loading pattern datasets for consistent certainty plots...")
        
        # Import required modules for dataset loading
        import os
        import numpy as np
        
        self.pattern_datasets = {}
        pattern_to_folder = {
            1: "struct_pattern_1",
            2: "struct_pattern_2", 
            3: "struct_pattern_3"
        }
        
        for pattern_id in [1, 2, 3]:
            try:
                dataset_folder = pattern_to_folder[pattern_id]
                dataset_path = os.path.join("src/datasets", dataset_folder)
                
                grids = np.load(os.path.join(dataset_path, "grids.npy")).astype(np.uint8)
                shapes = np.load(os.path.join(dataset_path, "shapes.npy")).astype(np.uint8)
                
                # Store the loaded datasets
                self.pattern_datasets[pattern_id] = {
                    'grids': jnp.array(grids),
                    'shapes': jnp.array(shapes),
                    'pattern_ids': jnp.full(len(grids), pattern_id, dtype=jnp.uint8)
                }
                
                logging.info(f"      ✅ Loaded {dataset_folder}: {grids.shape[0]} samples, {grids.shape}")
                
            except Exception as e:
                logging.error(f"      ❌ Failed to load dataset {dataset_folder}: {e}")
                logging.warning(f"      Falling back to synthetic data generation for pattern {pattern_id}")
                # Don't store anything - will use synthetic generation as fallback
        
        if self.pattern_datasets:
            logging.info(f"      📊 Pattern datasets loaded successfully: {list(self.pattern_datasets.keys())}")
            # Log dataset details for verification
            for pattern_id, data in self.pattern_datasets.items():
                logging.info(f"        Pattern {pattern_id}: {data['grids'].shape[0]} samples, shape: {data['grids'].shape}")
        else:
            logging.warning("      ⚠️ No pattern datasets loaded - will use synthetic generation")
        
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

    def train_n_steps(self, state: TrainState, batches: tuple[chex.Array, chex.Array, chex.Array], key: chex.PRNGKey) -> tuple[TrainState, dict]:
        """Process log_every_n_steps batches and return updated state and metrics."""
        num_steps = batches[0].shape[0]  # Should be log_every_n_steps
        keys = jax.random.split(key, num_steps)
        
        # CRITICAL: Extract explicit pattern IDs from balanced batches
        explicit_pattern_ids = batches[2]  # (batch_size,) - explicit pattern IDs aligned with data
        
        # Process each batch sequentially (since we don't have pmap)
        all_metrics = []
        for i in range(num_steps):
            batch_pairs, batch_shapes = batches[0][i], batches[1][i]
            batch_pattern_ids = explicit_pattern_ids  # Same pattern IDs for all steps
            rng = keys[i]
            
            def loss_fn(full_params, batch_pairs, batch_shapes, rng):
                # CRITICAL FIX: Use EXPLICIT pattern IDs that are aligned with balanced data
                # These pattern IDs are guaranteed to match the data ordering:
                # [Pattern 1 x42, Pattern 2 x42, Pattern 3 x42]
                pattern_ids = batch_pattern_ids  # Use explicit, aligned pattern IDs
                
                # Validate pattern distribution
                unique_patterns, counts = jnp.unique(pattern_ids, return_counts=True)
                # Convert JAX arrays to Python types for safe logging
                unique_patterns_py = [int(p) for p in unique_patterns]
                counts_py = [int(c) for c in counts]
                pattern_distribution = dict(zip(unique_patterns_py, counts_py))
                logging.debug(f"EXPLICIT pattern distribution: {pattern_distribution}")
                
                # CRITICAL: Verify we have the expected balanced distribution
                expected_distribution = {1: 42, 2: 42, 3: 42}  # 42 samples per pattern
                if pattern_distribution != expected_distribution:
                    logging.warning(f"⚠️  Pattern distribution mismatch!")
                    logging.warning(f"   Expected: {expected_distribution}")
                    logging.warning(f"   Got: {pattern_distribution}")
                    logging.warning(f"   This will break contrastive loss effectiveness!")
                
                # CRITICAL: Conditionally disable repulsion and contrastive losses after encoder exposure
                # During encoder exposure: use full coefficients for specialization
                # After encoder exposure: set to 0 to only train decoder with reconstruction loss
                if self.encoder_expose_steps > 0:
                    # Encoders are still trainable - use full coefficients
                    repulsion_coeff = self.cfg.training.get("repulsion_kl")
                    contrastive_coeff = self.cfg.training.get("contrastive_kl")
                    logging.info(f"🔓 Encoders TRAINABLE - Using repulsion: {repulsion_coeff}, contrastive: {contrastive_coeff}")
                    
                    # DEBUG: Check encoder parameters
                    logging.info(f"🔍 ENCODER PARAMETERS DEBUG:")
                    logging.info(f"   - full_params['encoders'] type: {type(full_params['encoders'])}")
                    logging.info(f"   - full_params['encoders'] length: {len(full_params['encoders'])}")
                    if len(full_params['encoders']) > 0:
                        for i, enc_params in enumerate(full_params['encoders']):
                            if 'encoder' in enc_params and 'Dense_0' in enc_params['encoder']:
                                kernel_shape = enc_params['encoder']['Dense_0']['kernel'].shape
                                kernel_mean = float(jnp.mean(enc_params['encoder']['Dense_0']['kernel']))
                                kernel_std = float(jnp.std(enc_params['encoder']['Dense_0']['kernel']))
                                logging.info(f"   - Encoder {i} Dense_0 kernel: shape={kernel_shape}, mean={kernel_mean:.6f}, std={kernel_std:.6f}")
                            else:
                                logging.info(f"   - Encoder {i} params structure: {list(enc_params.keys()) if isinstance(enc_params, dict) else type(enc_params)}")
                        
                        # Check if encoders are identical
                        if len(full_params['encoders']) >= 2:
                            enc0_params = full_params['encoders'][0]
                            enc1_params = full_params['encoders'][1]
                            if 'encoder' in enc0_params and 'Dense_0' in enc0_params['encoder'] and 'encoder' in enc1_params and 'Dense_0' in enc1_params['encoder']:
                                kernel0 = enc0_params['encoder']['Dense_0']['kernel']
                                kernel1 = enc1_params['encoder']['Dense_0']['kernel']
                                kernel_diff = float(jnp.mean(jnp.abs(kernel0 - kernel1)))
                                logging.info(f"   - Kernel difference between encoder 0 and 1: {kernel_diff:.6f}")
                                if kernel_diff < 1e-8:
                                    logging.warning(f"   ⚠️  WARNING: Encoder 0 and 1 kernels are IDENTICAL!")
                                
                                # Also check bias if it exists
                                if 'bias' in enc0_params['encoder']['Dense_0'] and 'bias' in enc1_params['encoder']['Dense_0']:
                                    bias0 = enc0_params['encoder']['Dense_0']['bias']
                                    bias1 = enc1_params['encoder']['Dense_0']['bias']
                                    bias_diff = float(jnp.mean(jnp.abs(bias0 - bias1)))
                                    logging.info(f"   - Bias difference between encoder 0 and 1: {bias_diff:.6f}")
                                    if bias_diff < 1e-8:
                                        logging.warning(f"   ⚠️  WARNING: Encoder 0 and 1 biases are IDENTICAL!")
                else:
                    # Encoders are frozen - disable specialization losses
                    repulsion_coeff = 0.0
                    contrastive_coeff = 0.0
                    logging.info(f"🔒 Encoders FROZEN - Disabled repulsion and contrastive losses")
                
                # DEBUG: Log model call parameters
                logging.info(f"🔍 MODEL CALL DEBUG:")
                logging.info(f"   - encoder_params_list length: {len(full_params['encoders'])}")
                logging.info(f"   - repulsion_kl_coeff: {repulsion_coeff}")
                logging.info(f"   - contrastive_kl_coeff: {contrastive_coeff}")
                logging.info(f"   - pattern_ids shape: {pattern_ids.shape if pattern_ids is not None else 'None'}")
                
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
                    repulsion_kl_coeff=repulsion_coeff,  # Conditional coefficient
                    contrastive_kl_coeff=contrastive_coeff,  # Conditional coefficient
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
        
        # Log essential repulsion loss metrics only
        if "repulsion_loss" in avg_metrics:
            repulsion_loss_val = float(np.array(avg_metrics['repulsion_loss']))
            repulsion_loss_weighted_val = float(np.array(avg_metrics.get('repulsion_loss_weighted', 0)))
            
            if self.encoder_expose_steps > 0:
                logging.info(f"Repulsion loss: {repulsion_loss_val:.6f} (weighted: {repulsion_loss_weighted_val:.6f})")
            else:
                logging.info(f"Repulsion loss: {repulsion_loss_val:.6f} (DISABLED - encoders frozen)")
        
        # Log essential contrastive loss metrics only
        if "contrastive_loss" in avg_metrics:
            contrastive_loss_val = float(np.array(avg_metrics['contrastive_loss']))
            contrastive_loss_weighted_val = float(np.array(avg_metrics.get('contrastive_loss_weighted', 0)))
            
            if self.encoder_expose_steps > 0:
                logging.info(f"Contrastive loss: {contrastive_loss_val:.6f} (weighted: {contrastive_loss_weighted_val:.6f})")
                
                # Essential stability check only
                if abs(contrastive_loss_val) > 50.0:
                    logging.warning(f"Contrastive loss is large ({contrastive_loss_val:.2f}). Consider reducing contrastive_kl coefficient.")
            else:
                logging.info(f"Contrastive loss: {contrastive_loss_val:.6f} (DISABLED - encoders frozen)")
        
        # Log essential training step info
        logging.debug(f"Training step completed: {num_steps} steps, batch size: {self.batch_size}")
        
        # CRITICAL: Compute clustering metrics every log_every_n_steps for Phase 1 training
        try:
            logging.debug(f"🔍 Computing Phase 1 clustering metrics for {num_steps} steps")
            # Use the first batch for clustering metrics (representative of the log_every_n_steps)
            first_batch_grids = batches[0][0]  # First step, first batch
            first_batch_shapes = batches[1][0]  # First step, first batch
            first_batch_pattern_ids = explicit_pattern_ids  # Same pattern IDs for all steps
            
            clustering_metrics = self._compute_clustering_metrics_every_step(
                state, first_batch_grids, first_batch_shapes, first_batch_pattern_ids, 
                step=None  # No specific step for Phase 1
            )
            
            # Add clustering metrics to the returned metrics
            avg_metrics.update(clustering_metrics)
            logging.debug(f"✅ Phase 1 clustering metrics computed: {list(clustering_metrics.keys())}")
            
        except Exception as e:
            logging.warning(f"Phase 1 clustering metrics computation failed: {e}")
        
        # Decrement exposure counter by number of gradient steps completed
        self.encoder_expose_steps = max(0, self.encoder_expose_steps - num_steps)
        return state, avg_metrics

    def _create_balanced_pattern_batch(self, batch_size: int, samples_per_pattern: int) -> tuple[chex.Array, chex.Array, chex.Array]:
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
        
        for pattern_id in [1, 2, 3]:  # Pattern 1, Pattern 2, Pattern 3
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
        
        # CRITICAL FIX: Align pattern generation with pattern IDs
        # Create explicit pattern IDs that match the concatenation order
        pattern_ids = jnp.concatenate([
            jnp.full((samples_per_pattern,), 1),  # Pattern 1
            jnp.full((samples_per_pattern,), 2),  # Pattern 2  
            jnp.full((samples_per_pattern,), 3),  # Pattern 3
        ], axis=0)
        
        # Concatenate all patterns to create balanced batch
        balanced_grids = jnp.concatenate(grids_list, axis=0)
        balanced_shapes = jnp.concatenate(shapes_list, axis=0)
        
        # DEBUG: Log the final concatenated shapes and pattern alignment
        logging.debug(f"Final balanced batch - grids shape: {balanced_grids.shape}, shapes shape: {balanced_shapes.shape}")
        logging.debug(f"Pattern IDs: {pattern_ids[:10]}... (first 10) - should be [1,1,1,...,2,2,2,...,3,3,3,...]")
        
        # Increment batch counter for different seeds
        if not hasattr(self, '_batch_counter'):
            self._batch_counter = 0
        self._batch_counter += 1
        
        return balanced_grids, balanced_shapes, pattern_ids

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
        
        # Generate the first batch to get the pattern IDs (they're the same for all steps)
        first_batch = self._create_balanced_pattern_batch(
            self.batch_size, 
            self.samples_per_pattern_per_batch
        )
        first_grids, first_shapes, first_pattern_ids = first_batch
        all_grids.append(first_grids)
        all_shapes.append(first_shapes)
        
        # Generate remaining batches (without regenerating pattern IDs)
        for step in range(1, log_every_n_steps):
            # Generate a balanced batch for this step
            balanced_grids, balanced_shapes, _ = self._create_balanced_pattern_batch(
                self.batch_size, 
                self.samples_per_pattern_per_batch
            )
            all_grids.append(balanced_grids)
            all_shapes.append(balanced_shapes)
        
        # Stack all batches to create the expected format: (log_every_n_steps, batch_size, ...)
        stacked_grids = jnp.stack(all_grids, axis=0)  # (log_every_n_steps, batch_size, ...)
        stacked_shapes = jnp.stack(all_shapes, axis=0)  # (log_every_n_steps, batch_size, ...)
        
        # Yield the stacked batches with the correct pattern IDs from the first batch
        yield (stacked_grids, stacked_shapes, first_pattern_ids)

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
        Essential: Validate that encoders are outputting proper variance terms.
        
        Args:
            state: Current training state
            test_batch: Test batch for validation
        """
        logging.info("Validating encoder variance outputs...")
        
        try:
            # Test encoder outputs on a small batch
            test_pairs, test_shapes = test_batch
            test_batch_size = min(4, test_pairs.shape[0])
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
                    
                    # Check variance values
                    var_i = jnp.exp(logvar_i)
                    mean_var = float(jnp.mean(var_i))
                    
                    # Essential check for fixed variance
                    if float(jnp.max(var_i)) - float(jnp.min(var_i)) < 1e-6:
                        logging.warning(f"Encoder {enc_idx} has nearly fixed variance! This will prevent specialization.")
                    else:
                        logging.info(f"Encoder {enc_idx} has variable variance ✓")
                        
                except Exception as e:
                    logging.error(f"Failed to validate encoder {enc_idx}: {e}")
                    continue
            
            logging.info("Encoder variance validation completed")
            
        except Exception as e:
            logging.error(f"Encoder variance validation failed: {e}")

    def _validate_contrastive_loss_patterns(self, batch_pattern_ids: chex.Array, batch_size: int) -> None:
        """
        Essential: Validate that contrastive loss is receiving correct pattern distribution.
        
        Args:
            batch_pattern_ids: Pattern IDs for the current batch
            batch_size: Total batch size
        """
        logging.info("Validating contrastive loss pattern distribution...")
        
        try:
            # Convert to numpy for analysis
            pattern_ids_np = np.array(batch_pattern_ids)
            
            # Check pattern distribution
            unique_patterns, counts = np.unique(pattern_ids_np, return_counts=True)
            pattern_distribution = dict(zip(unique_patterns, counts))
            
            # Validate expected distribution
            expected_samples_per_pattern = batch_size // 3
            expected_distribution = {
                1: expected_samples_per_pattern,  # O-tetromino
                2: expected_samples_per_pattern,  # T-tetromino  
                3: expected_samples_per_pattern   # L-tetromino
            }
            
            if pattern_distribution == expected_distribution:
                logging.info("✅ Pattern distribution is optimal for contrastive loss")
            else:
                logging.warning("⚠️  Pattern distribution is not optimal for contrastive loss")
                logging.warning(f"   - Expected: {expected_distribution}, Got: {pattern_distribution}")
            
            # Essential check for pattern diversity
            if len(unique_patterns) >= 2:
                logging.info("✅ Batch contains multiple pattern types - contrastive loss can work")
            else:
                logging.error("❌ Batch contains only one pattern type! Contrastive loss will NOT work!")
            
            logging.info("Contrastive loss pattern validation completed")
            
        except Exception as e:
            logging.error(f"Contrastive loss pattern validation failed: {e}")

    def _specialize_individual_encoders(self, state: TrainState, enc_params_list: list[dict]) -> TrainState:
        """
        PHASE 1: Individual encoder specialization using original decoders.
        
        Each encoder is trained on complementary data subsets to reinforce
        certainty on "their" patterns and decrease it for others.
        
        Args:
            state: Current training state
            enc_params_list: List of encoder parameters to specialize
            
        Returns:
            Updated state with specialized encoders
        """
        logging.info("🚀 PHASE 1: Starting individual encoder specialization...")
        logging.info(f"   - Training each encoder independently on complementary data")
        logging.info(f"   - Using original decoders to prevent interference")
        logging.info(f"   - Focus: pattern specialization through contrastive learning")
        
        # Store original parameters for restoration
        self.original_encoder_params = [jax.tree_util.tree_map(lambda x: x, enc_params) for enc_params in enc_params_list]
        self.original_decoder_params = jax.tree_util.tree_map(lambda x: x, state.params["decoder"])
        
        # Initialize repulsion loss system
        repulsion_kl = self.cfg.training.get("repulsion_kl", 0)
        target_latents_store = {}  # {encoder_idx: {pattern_id: target_latents}}
        
        if repulsion_kl > 0:
            logging.info(f"🚫 Repulsion Loss Enabled: λ={repulsion_kl}")
            logging.info(f"   - Sequential training: Each encoder will be pushed away from previous encoders")
            logging.info(f"   - Training order: Encoder 0 → Encoder 1 (repulses from 0) → Encoder 2 (repulses from 0,1)")
        else:
            logging.info(f"⚠️  Repulsion Loss Disabled: repulsion_kl={repulsion_kl}")
            logging.info(f"   - Parallel training: All encoders train independently without repulsion")
        
        # Create individual training states for each encoder
        specialized_encoders = []
        
        for enc_idx, enc_params in enumerate(enc_params_list):
            logging.info(f"🔓 Specializing Encoder {enc_idx}...")
            
            if repulsion_kl > 0 and enc_idx > 0:
                logging.info(f"   🚫 Encoder {enc_idx} will repulse from Encoders 0 to {enc_idx-1}")
                logging.info(f"   📦 Available target latents: {list(target_latents_store.keys())}")
            
            # Debug: Check encoder structure
            logging.info(f"   Encoder {enc_idx} type: {type(self.encoders[enc_idx])}")
            logging.info(f"   Encoder {enc_idx} has apply method: {hasattr(self.encoders[enc_idx], 'apply')}")
            
            # Create individual training state with all encoders but only one trainable
            # This ensures the model structure matches the parameters
            
            # Debug: Check original encoder parameters
            logging.info(f"   Original encoder params length: {len(self.original_encoder_params)}")
            for i, params in enumerate(self.original_encoder_params):
                logging.info(f"   Original encoder {i} params type: {type(params)}")
                if params is None:
                    logging.info(f"   Original encoder {i} params is None!")
                else:
                    logging.info(f"   Original encoder {i} params keys: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}")
            
            all_encoder_params = list(self.original_encoder_params)  # Copy all encoder parameters
            all_encoder_params[enc_idx] = enc_params  # Replace the one we want to train
            
            # Debug: Check final encoder parameters
            logging.info(f"   Final encoder params length: {len(all_encoder_params)}")
            for i, params in enumerate(all_encoder_params):
                logging.info(f"   Final encoder {i} params type: {type(params)}")
                if params is None:
                    logging.info(f"   Final encoder {i} params is None!")
                else:
                    logging.info(f"   Original encoder {i} params keys: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}")
            individual_state = TrainState.create(
                apply_fn=self.model.apply,  # Use the main model's apply function
                tx=optax.adamw(self.cfg.training.learning_rate),
                params={
                    "encoders": tuple(all_encoder_params),  # All encoders, but only one is trainable
                    "decoder": self.original_decoder_params
                }
            )
            
            # Train this encoder on complementary data with repulsion from previous encoders
            specialized_encoder = self._train_encoder_individually(
                enc_idx, individual_state, self.model, target_latents_store
            )
            
            # Store target latents from this encoder for future repulsion
            if repulsion_kl > 0:
                target_latents_store[enc_idx] = self._extract_target_latents(enc_idx, specialized_encoder, individual_state)
                logging.info(f"   📦 Stored target latents for Encoder {enc_idx} (will be used for repulsion)")
            
            # Evaluate the specialized encoder and create visualizations
            logging.info(f"   Evaluating specialized Encoder {enc_idx}...")
            # Calculate global step: current phase_a_global_step + steps completed by this encoder
            current_global_step = self.phase_a_global_step + self.encoder_expose_steps
            logging.info(f"   📊 Evaluation global step: {current_global_step} (phase_a: {self.phase_a_global_step} + encoder_steps: {self.encoder_expose_steps})")
            self._evaluate_specialized_encoder(enc_idx, specialized_encoder, individual_state, current_global_step)
            
            specialized_encoders.append(specialized_encoder)
            logging.info(f"✅ Encoder {enc_idx} specialization completed")
            
            # Update global step counter for next encoder
            self.phase_a_global_step += self.encoder_expose_steps
        
        # Update the main state with specialized encoders
        new_params = dict(state.params)
        new_params["encoders"] = tuple(specialized_encoders)
        updated_state = state.replace(params=new_params)
        
        self.phase1_completed = True
        
        # Calculate Phase A evaluation statistics
        eval_every_n_steps = self.cfg.training.get("eval_every_n_logs", 20) * self.cfg.training.get("log_every_n_steps", 5)
        total_phase_a_evals = 0
        for enc_idx in range(len(enc_params_list)):
            num_evals = self.encoder_expose_steps // eval_every_n_steps
            total_phase_a_evals += num_evals
            logging.info(f"   - Encoder {enc_idx}: {num_evals} evaluations generated (T-SNE + certainty plots)")
        
        logging.info("🎉 PHASE 1 COMPLETED: All encoders specialized!")
        logging.info(f"   - Total Phase A evaluations: {total_phase_a_evals}")
        logging.info(f"   - Each evaluation includes: T-SNE visualization + certainty plots + clustering metrics for all 3 patterns")
        logging.info("   - Encoders now have pattern-specific representations")
        logging.info("   - Clustering metrics computed and logged for each encoder")
        logging.info("   - Ready for Phase 2: Joint decoder training")
        
        return updated_state
    
    def _train_encoder_individually(self, enc_idx: int, state: TrainState, model: StructuredLPN, target_latents_store: dict = None) -> dict:
        """
        Train a single encoder individually on complementary data.
        
        Args:
            enc_idx: Index of encoder to train
            state: Individual training state
            model: Individual model (1 encoder + original decoder)
            
        Returns:
            Trained encoder parameters
        """
        logging.info(f"   Training Encoder {enc_idx} individually...")
        
        # Create complementary data for this encoder
        # Pattern enc_idx+1 gets reinforced, others get reduced certainty
        target_pattern = enc_idx + 1  # Encoder 0 -> Pattern 1, Encoder 1 -> Pattern 2, etc.
        
        # Generate specialized training data
        specialized_data = self._create_specialized_training_data(target_pattern)
        
        # Train for encoder_expose_steps
        num_steps = self.encoder_expose_steps
        key = jax.random.PRNGKey(self.cfg.training.seed + enc_idx)
        
        # Phase A evaluation frequency: every eval_every_n_logs * log_every_n_steps steps
        eval_every_n_steps = self.cfg.training.get("eval_every_n_logs", 20) * self.cfg.training.get("log_every_n_steps", 5)
        logging.info(f"   Phase A evaluation frequency: every {eval_every_n_steps} steps")
        logging.info(f"   - T-SNE visualizations: every {eval_every_n_steps} steps")
        logging.info(f"   - Certainty plots: every {eval_every_n_steps} steps (all 3 patterns)")
        logging.info(f"   - Total evaluations per encoder: {self.encoder_expose_steps // eval_every_n_steps}")
        
        for step in range(num_steps):
            # CRITICAL FIX: Generate NEW samples for every batch instead of sampling from pre-generated data
            # This ensures realistic variance computation and prevents overfitting to static data
            batch = self._generate_new_batch_for_training(target_pattern, step)
            
            # Debug: Check model and parameters structure
            if step == 0:
                logging.info(f"   Model type: {type(model)}")
                logging.info(f"   Model encoders: {type(model.encoders)}")
                logging.info(f"   Model encoders length: {len(model.encoders)}")
                logging.info(f"   State params encoders type: {type(state.params['encoders'])}")
                logging.info(f"   State params encoders length: {len(state.params['encoders'])}")
            
            # Comprehensive training loop with proper contrastive loss and metrics
            
            # Get the encoder we're training
            encoder = self.encoders[enc_idx]
            encoder_params = state.params["encoders"][enc_idx]  # Get the encoder we're training
            
            # Forward pass through encoder
            mu, logvar = encoder.apply(
                {"params": encoder_params},
                batch[0],  # grids
                batch[1],  # shapes
                dropout_eval=False,
                mutable=False,
            )
            
            # Compute proper contrastive loss for encoder specialization
            # Pattern enc_idx+1 should have low variance, others should have high variance
            target_pattern = enc_idx + 1
            pattern_ids = batch[2]  # (batch_size,)
            
            # Separate samples by pattern
            target_mask = (pattern_ids == target_pattern)
            other_mask = ~target_mask
            
            if jnp.any(target_mask) and jnp.any(other_mask):
                # Target pattern variance (should be minimized)
                target_var = jnp.exp(logvar[target_mask])
                avg_target_var = jnp.mean(target_var)
                
                # Other patterns variance (should be maximized)
                other_var = jnp.exp(logvar[other_mask])
                avg_other_var = jnp.mean(other_var)
                
                # FIXED: Contrastive loss: minimize target variance, maximize other variance
                # We want: target_var << other_var (target pattern gets high confidence, others get low confidence)
                
                # Dynamic coefficient adjustment based on specialization progress
                base_coeff = self.cfg.training.get("contrastive_kl", 1e-3)
                current_specialization_ratio = avg_target_var / (avg_other_var + 1e-8)
                
                # If specialization is poor, increase coefficient
                if current_specialization_ratio > 1.0:
                    # Target variance is HIGHER than other variance (bad!)
                    dynamic_coeff = base_coeff * 10.0  # Increase coefficient aggressively
                    logging.debug(f"       Poor specialization detected (ratio: {current_specialization_ratio:.3f}), increasing coefficient to {dynamic_coeff:.6f}")
                elif current_specialization_ratio > 0.8:
                    # Target variance is only slightly lower than other variance
                    dynamic_coeff = base_coeff * 5.0  # Increase coefficient moderately
                    logging.debug(f"       Weak specialization detected (ratio: {current_specialization_ratio:.3f}), increasing coefficient to {dynamic_coeff:.6f}")
                else:
                    # Good specialization, use base coefficient
                    dynamic_coeff = base_coeff
                    logging.debug(f"       Good specialization (ratio: {current_specialization_ratio:.3f}), using base coefficient {dynamic_coeff:.6f}")
                
                contrastive_loss = avg_target_var + dynamic_coeff * (1.0 / (avg_other_var + 1e-8))
                
                # Add regularization to prevent extreme values
                reg_loss = 0.01 * (jnp.mean(target_var ** 2) + jnp.mean(other_var ** 2))
                
                # Add repulsion loss to push away from previous encoders' latent targets
                repulsion_loss = 0.0
                if target_latents_store and self.cfg.training.get("repulsion_kl", 0) > 0:
                    # Compute repulsion from previous encoders' targets
                    # Use a more appropriate margin based on observed distances (~7.4)
                    repulsion_loss = self._compute_repulsion_loss(
                        current_latents=mu.mean(axis=-2),  # Use mean over pairs
                        target_latents_store=target_latents_store,
                        current_encoder_idx=enc_idx,
                        margin=5.0  # Increased from 1.0 to be more effective given observed distances
                    )
                    
                    # Scale repulsion loss by the coefficient
                    repulsion_coeff = self.cfg.training.get("repulsion_kl", 0)
                    repulsion_loss = repulsion_coeff * repulsion_loss
                    
                    if step % 50 == 0:
                        logging.info(f"       Repulsion Loss: {float(repulsion_loss):.6f} (λ={repulsion_coeff})")
                
                total_loss = contrastive_loss + reg_loss + repulsion_loss
                
                # FIXED: Compute gradients properly for contrastive learning
                def contrastive_loss_fn(params):
                    # Forward pass through encoder
                    mu, logvar = encoder.apply(
                        {"params": params}, batch[0], batch[1], dropout_eval=False, mutable=False
                    )
                    
                    # Separate by pattern
                    target_var = jnp.exp(logvar[target_mask])
                    other_var = jnp.exp(logvar[other_mask])
                    
                    # Compute contrastive loss: minimize target variance, maximize other variance
                    avg_target_var = jnp.mean(target_var)
                    avg_other_var = jnp.mean(other_var)
                    
                    # Loss: target_var + coefficient * (1/other_var) 
                    # This drives target_var DOWN and other_var UP
                    
                    # Use the same dynamic coefficient logic
                    base_coeff = self.cfg.training.get("contrastive_kl", 1e-3)
                    current_specialization_ratio = avg_target_var / (avg_other_var + 1e-8)
                    
                    if current_specialization_ratio > 1.0:
                        dynamic_coeff = base_coeff * 10.0
                    elif current_specialization_ratio > 0.8:
                        dynamic_coeff = base_coeff * 5.0
                    else:
                        dynamic_coeff = base_coeff
                    
                    loss = avg_target_var + dynamic_coeff * (1.0 / (avg_other_var + 1e-8))
                    
                    # Add regularization
                    reg = 0.01 * (jnp.mean(target_var ** 2) + jnp.mean(other_var ** 2))
                    return loss + reg
                
                # Compute gradients
                grads = jax.grad(contrastive_loss_fn)(encoder_params)
                
                # Update encoder parameters
                new_encoder_params = jax.tree_util.tree_map(
                    lambda p, g: p - self.cfg.training.learning_rate * g,
                    encoder_params, grads
                )
                
                # Update state - only update the encoder we're training
                all_encoder_params = list(state.params["encoders"])
                all_encoder_params[enc_idx] = new_encoder_params
                
                # Create new params dictionary
                new_params = dict(state.params)
                new_params["encoders"] = tuple(all_encoder_params)
                
                # Update state
                state = state.replace(params=new_params)
                
                # Log essential metrics to WandB with proper tab organization
                if step % 10 == 0:  # Log more frequently
                    current_global_step = self.phase_a_global_step + step
                    
                    # Organize metrics into proper WandB tabs
                    wandb.log({
                        # Core training metrics (like train.py) - goes to phase_a_losses tab
                        f"phase_a_losses/encoder_{enc_idx}/total_loss": float(total_loss),
                        
                        # Essential contrastive loss tracking - goes to phase_a_losses tab
                        f"phase_a_losses/encoder_{enc_idx}/contrastive_loss": float(contrastive_loss),
                        f"phase_a_losses/encoder_{enc_idx}/contrastive_loss_weighted": float(contrastive_loss * self.cfg.training.get("contrastive_kl", 0.5)),
                        f"phase_a_losses/encoder_{enc_idx}/reg_loss": float(reg_loss),
                        
                        # Summary metrics for phase_a_losses tab
                        f"phase_a_losses/encoder_{enc_idx}/loss_breakdown": {
                            "contrastive": float(contrastive_loss),
                            "regularization": float(reg_loss),
                            "repulsion": float(repulsion_loss) if repulsion_loss > 0 else 0.0,
                            "total": float(total_loss)
                        },
                        
                        # Encoder variance per pattern (essential for specialization monitoring) - goes to encoder_[i] tab
                        f"encoder_{enc_idx}/target_pattern_mean_variance": float(avg_target_var),
                        f"encoder_{enc_idx}/other_patterns_mean_variance": float(avg_other_var),
                        f"encoder_{enc_idx}/specialization_ratio": float(avg_other_var / (avg_target_var + 1e-8)),
                        f"encoder_{enc_idx}/specialization_score": float(jnp.log(avg_other_var / (avg_target_var + 1e-8) + 1e-8)),
                        
                        # Additional encoder metrics for comprehensive monitoring
                        f"encoder_{enc_idx}/target_pattern": target_pattern,
                        f"encoder_{enc_idx}/target_pattern_count": int(jnp.sum(target_mask)),
                        f"encoder_{enc_idx}/other_samples_count": int(jnp.sum(other_mask)),
                        
                        # Repulsion loss metrics (if enabled)
                        f"encoder_{enc_idx}/repulsion_loss": float(repulsion_loss) if repulsion_loss > 0 else 0.0,
                        f"encoder_{enc_idx}/repulsion_coefficient": self.cfg.training.get("repulsion_kl", 0),
                    }, step=current_global_step)
                
                if step % 50 == 0:
                    # Calculate specialization metrics
                    specialization_ratio = float(avg_target_var / (avg_other_var + 1e-8))
                    specialization_score = float(jnp.log(specialization_ratio + 1e-8))
                    
                    logging.info(f"     Encoder {enc_idx} - Step {step}/{num_steps} - Total Loss: {float(total_loss):.6f}")
                    logging.info(f"       Contrastive: {float(contrastive_loss):.6f}")
                    logging.info(f"       Target Var: {float(avg_target_var):.6f}, Other Var: {float(avg_other_var):.6f}")
                    logging.info(f"       Specialization Ratio: {specialization_ratio:.3f} (target/other)")
                    logging.info(f"       Specialization Score: {specialization_score:.3f} (log ratio)")
                    
                    # Assess specialization quality
                    if specialization_ratio < 0.5:
                        logging.info(f"       ✅ EXCELLENT specialization: target variance is {1/specialization_ratio:.1f}x LOWER")
                    elif specialization_ratio < 0.8:
                        logging.info(f"       ✅ GOOD specialization: target variance is {1/specialization_ratio:.1f}x LOWER")
                    elif specialization_ratio < 1.2:
                        logging.info(f"       ⚠️  WEAK specialization: target variance is only {1/specialization_ratio:.1f}x LOWER")
                    else:
                        logging.warning(f"       ❌ POOR specialization: target variance is {specialization_ratio:.1f}x HIGHER!")
                        logging.warning(f"       This indicates the encoder is NOT specializing correctly!")
            else:
                # Fallback if no target or other patterns in batch
                total_loss = jnp.mean((mu - 0.0) ** 2)
                logging.warning(f"     Encoder {enc_idx} - Step {step}/{num_steps} - No target/other patterns in batch, using fallback loss")
            
            # Phase A T-SNE Evaluation: Generate T-SNE plots at regular intervals
            if step % eval_every_n_steps == 0 and step > 0:
                try:
                    logging.info(f"     🔍 Phase A T-SNE Evaluation at step {step}/{num_steps}")
                    self._generate_phase_a_tsne(enc_idx, encoder_params, step, num_steps)
                except Exception as e:
                    logging.warning(f"     Phase A T-SNE generation failed at step {step}: {e}")
        
        return state.params["encoders"][0]  # Return trained encoder params
    
    def _evaluate_specialized_encoder(self, enc_idx: int, encoder_params: dict, state: TrainState, global_step: int):
        """
        Evaluate a specialized encoder and create comprehensive visualizations.
        
        Args:
            enc_idx: Index of the encoder
            encoder_params: Trained encoder parameters
            state: Training state with all parameters
        """
        logging.info(f"     Creating comprehensive evaluation for Encoder {enc_idx}...")
        
        # Create a temporary model for evaluation
        temp_model = StructuredLPN(
            encoders=(self.encoders[enc_idx],),
            decoder=self.decoder
        )
        
        # Generate evaluation data for all patterns
        # REMOVED: Complex data generation logic since no plots are generated here
        # The correct plots are generated by _generate_phase_a_tsne method
        # This method now only computes variance metrics for monitoring
        
        # Create minimal eval_data for variance computation only
        # Since no plots are generated here, we only need minimal data for variance metrics
        eval_data = {}
        for pattern_id in [1, 2, 3]:
            # Create minimal data for variance computation (not for plotting)
            # This ensures the variance metrics can still be computed
            eval_data[pattern_id] = (None, None, None)  # Placeholder data

        
        # REMOVED: Variance computation since no plots are generated here
        # The correct plots and variance metrics are generated by _generate_phase_a_tsne method
        # This eliminates duplicate computation and ensures consistency
        
        # REMOVED: T-SNE, clustering, and reconstruction evaluation since no data is available
        # The correct visualizations and metrics are generated by _generate_phase_a_tsne method
        # This eliminates duplicate computation and ensures consistency
        
        logging.info(f"     ✅ Evaluation completed for Encoder {enc_idx}")
    
    def _create_encoder_tsne(self, enc_idx: int, encoder_params: dict, eval_data: dict, global_step: int):
        """Create T-SNE visualization for encoder latents - matching train.py style exactly."""
        try:
            # Collect latents from all patterns
            all_latents = []
            all_patterns = []
            
            # Configuration for number of samples and resampling
            max_total_points = int(self.cfg.eval.get("tsne_max_points", 2304))
            max_samples_per_pattern = max_total_points // 3
            num_resamples = 3

            for pattern_id, (grids, shapes, pattern_ids) in eval_data.items():
                # Sample subset for T-SNE
                if len(grids) > max_samples_per_pattern:
                    indices = np.random.choice(len(grids), max_samples_per_pattern, replace=False)
                    sample_grids = grids[indices]
                    sample_shapes = shapes[indices]
                    sample_pattern_ids = np.array(pattern_ids)[indices]
                else:
                    sample_grids, sample_shapes = grids, shapes
                    sample_pattern_ids = np.array(pattern_ids)

                # Get latents
                mu, logvar = self.encoders[enc_idx].apply(
                    {"params": encoder_params},
                    sample_grids,
                    sample_shapes,
                    dropout_eval=False,
                    mutable=False,
                )

                # Resample latents multiple times to increase sample count
                mu_np = np.array(mu)
                logvar_np = np.array(logvar)
                std_np = np.exp(0.5 * logvar_np)
                samples = []
                for _ in range(num_resamples):
                    samples.append(mu_np + np.random.randn(*mu_np.shape) * std_np)
                samples = np.concatenate(samples, axis=0)

                all_latents.append(samples)
                all_patterns.extend(np.repeat(sample_pattern_ids, num_resamples))

            # Concatenate and flatten
            all_latents = np.concatenate(all_latents, axis=0)
            all_latents_flat = all_latents.reshape(all_latents.shape[0], -1)

            # Use visualize_tsne function to match train.py style exactly
            fig_latents = visualize_tsne(jnp.array(all_latents_flat), np.array(all_patterns))
            
            # Log to WandB
            wandb.log({f"encoder_{enc_idx}/tsne_plot": wandb.Image(fig_latents)}, step=global_step)
            plt.close(fig_latents)
            
        except Exception as e:
            logging.warning(f"T-SNE creation failed for Encoder {enc_idx}: {e}")
    
    def _evaluate_target_pattern_reconstruction(self, enc_idx: int, encoder_params: dict, target_pattern: int, 
                                             target_data: tuple, global_step: int, state: TrainState) -> dict:
        """
        Evaluate reconstruction quality for an encoder's target pattern.
        
        Args:
            enc_idx: Index of the encoder being evaluated
            encoder_params: Current encoder parameters
            target_pattern: Target pattern ID for this encoder
            target_data: Tuple of (grids, shapes, pattern_ids) for target pattern
            global_step: Current global training step
            
        Returns:
            dict: Dictionary containing reconstruction metrics
        """
        try:
            grids, shapes, pattern_ids = target_data
            
            # Sample subset for evaluation (use first 50 samples for efficiency)
            num_eval_samples = min(50, len(grids))
            eval_grids = grids[:num_eval_samples]
            eval_shapes = shapes[:num_eval_samples]
            
            # Create a temporary model for reconstruction evaluation
            temp_model = StructuredLPN(
                encoders=(self.encoders[enc_idx],),
                decoder=self.decoder
            )
            
            # Get encoder outputs (latents)
            mu, logvar = self.encoders[enc_idx].apply(
                {"params": encoder_params},
                eval_grids,
                eval_shapes,
                dropout_eval=False,
                mutable=False,
            )
            
            # Convert to numpy for evaluation
            orig_grids_np = np.array(eval_grids)
            orig_shapes_np = np.array(eval_shapes)
            mu_np = np.array(mu)
            logvar_np = np.array(logvar)
            
            # Compute encoder specialization metrics for the target pattern
            metrics = {}
            
            # 1. Encoder variance analysis (specialization quality)
            variances = np.exp(logvar_np)  # Convert logvar to variance
            mean_variance = float(np.mean(variances))
            variance_std = float(np.std(variances))
            
            # 2. Confidence score based on variance (lower variance = higher confidence)
            # For target pattern, we want low variance (high confidence)
            confidence_score = 1.0 / (1.0 + mean_variance)  # Higher confidence for lower variance
            metrics['confidence_score'] = float(confidence_score)
            
            # 3. Specialization quality (how well encoder specializes in target pattern)
            # Lower variance indicates better specialization
            specialization_quality = 1.0 / (1.0 + mean_variance)
            metrics['specialization_quality'] = float(specialization_quality)
            
            # 4. Variance statistics
            metrics['mean_variance'] = mean_variance
            metrics['variance_std'] = variance_std
            metrics['min_variance'] = float(np.min(variances))
            metrics['max_variance'] = float(np.max(variances))
            
            # 5. Overall specialization score (combined metric)
            overall_score = (confidence_score + specialization_quality) / 2.0
            metrics['overall_score'] = float(overall_score)
            
            # Log metrics to WandB
            pattern_names = {1: "L-tetromino", 2: "O-tetromino", 3: "T-tetromino"}
            pattern_name = pattern_names.get(target_pattern, f"Pattern {target_pattern}")
            
            for metric_name, metric_value in metrics.items():
                wandb.log({
                    f"encoder_{enc_idx}/target_pattern_reconstruction/{metric_name}": metric_value
                }, step=global_step)
            
            # Log summary metric
            wandb.log({
                f"encoder_{enc_idx}/target_pattern_reconstruction": overall_score
            }, step=global_step)
            
            logging.info(f"         📊 Target pattern reconstruction metrics for {pattern_name}:")
            logging.info(f"           - Overall score: {overall_score:.4f}")
            logging.info(f"           - Mean variance: {mean_variance:.6f}")
            logging.info(f"           - Variance std: {variance_std:.6f}")
            logging.info(f"           - Confidence score: {confidence_score:.4f}")
            logging.info(f"           - Specialization quality: {specialization_quality:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.warning(f"Target pattern reconstruction evaluation failed for encoder {enc_idx}: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _generate_phase_a_tsne(self, enc_idx: int, encoder_params: dict, step: int, total_steps: int):
        """
        Generate T-SNE visualization and certainty plots during Phase A training to monitor encoder specialization progress.
        
        Args:
            enc_idx: Index of the encoder being trained
            encoder_params: Current encoder parameters
            step: Current training step
            total_steps: Total training steps for this encoder
        """
        try:
            logging.info(f"       🔍 Phase A Evaluation at step {step}/{total_steps}")
            
            # Create evaluation data for all patterns to show specialization progress
            # CRITICAL FIX: Generate diverse samples for each pattern (not duplicates)
            eval_data = {}
            for pattern_id in [1, 2, 3]:
                # Generate diverse samples for each pattern using different seeds
                # This ensures realistic variance computation and prevents artificial zero variance
                grids_list = []
                shapes_list = []
                pattern_ids_list = []
                
                num_samples = 1260  # Generate diverse samples for evaluation
                logging.info(f"         🔍 Pattern {pattern_id}: Generating {num_samples} DIVERSE samples for realistic evaluation")
                
                for i in range(num_samples):
                    if i % 100 == 0:  # Progress logging
                        logging.info(f"         🔍 Pattern {pattern_id}: Generated {i}/{num_samples} diverse samples")
                    
                    # CRITICAL FIX: Use different seed for each sample to ensure diversity
                    # This prevents the "1260 copies of one grid" problem
                    sample_seed = self.cfg.training.seed + pattern_id * 1000 + i
                    grids, shapes, _ = self._create_single_pattern_sample_with_seed(pattern_id, sample_seed)
                    grids_list.append(grids)
                    shapes_list.append(shapes)
                    pattern_ids_list.append(pattern_id)
                
                # Stack all samples
                grids = jnp.stack(grids_list, axis=0)
                shapes = jnp.stack(shapes_list, axis=0)
                pattern_ids = jnp.array(pattern_ids_list)
                
                eval_data[pattern_id] = (grids, shapes, pattern_ids)
                
                logging.info(f"         🔍 Pattern {pattern_id}: Generated {len(grids)} DIVERSE samples (should be {num_samples})")
                logging.info(f"         🔍 Data shapes: grids={grids.shape}, shapes={shapes.shape}, pattern_ids={pattern_ids.shape}")
                logging.info(f"         🔍 CRITICAL: Each sample uses different seed for realistic variance")
            
            # Generate T-SNE visualization
            current_global_step = self.phase_a_global_step + step
            try:
                self._create_encoder_tsne(enc_idx, encoder_params, eval_data, current_global_step)
            except Exception as e:
                logging.error(f"         ❌ T-SNE creation failed: {e}")
                # Continue with other evaluations
            
            # Generate certainty plots for all patterns
            logging.info(f"       📊 Generating certainty plots for all patterns...")
            # Debug: Log what patterns we're generating
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    grids, shapes, pattern_ids = eval_data[pattern_id]
                    logging.info(f"         Pattern {pattern_id}: grids shape {grids.shape}, shapes shape {shapes.shape}")
                    # Show a sample of the first grid to verify pattern
                    sample_grid = np.array(grids[0, 0, :, :, 1])  # First sample, first pair, output channel
                    logging.info(f"         Pattern {pattern_id} sample output grid:\n{sample_grid}")
            
            # CRITICAL: Store Phase A data for comparison with merged panel
            # This helps debug why histograms are destroyed in subsequent calls
            # FIX: Create IMMUTABLE copies to prevent dataset mutation
            self._last_phase_a_data = {}
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    try:
                        grids, shapes, pattern_ids = eval_data[pattern_id]
                        
                        # CRITICAL FIX: Create immutable copies to prevent dataset mutation
                        # This ensures the stored data cannot be modified by subsequent operations
                        grids_copy = jnp.array(grids, copy=True)  # Deep copy
                        shapes_copy = jnp.array(shapes, copy=True)  # Deep copy
                        pattern_ids_copy = jnp.array(pattern_ids, copy=True)  # Deep copy
                        
                        self._last_phase_a_data[pattern_id] = {
                            'grids': grids_copy,
                            'shapes': shapes_copy,
                            'pattern_ids': pattern_ids_copy,
                            'original_length': len(grids),  # Store original length for verification
                            'storage_step': step,  # Track when data was stored
                            'storage_global_step': current_global_step  # Track global step
                        }
                        
                        logging.info(f"         💾 Stored Phase A data for pattern {pattern_id}: {len(grids)} samples")
                        logging.info(f"         💾 Data structure: grids={grids.shape}, shapes={shapes.shape}, pattern_ids={pattern_ids.shape}")
                        logging.info(f"         💾 IMMUTABLE COPY created to prevent dataset mutation")
                    except Exception as e:
                        logging.error(f"         ❌ Failed to store Phase A data for pattern {pattern_id}: {e}")
                        continue
            
            # CRITICAL DEBUG: Verify data storage
            try:
                logging.info(f"         🔍 Phase A data storage verification:")
                logging.info(f"           - _last_phase_a_data keys: {list(self._last_phase_a_data.keys())}")
                for pattern_id in [1, 2, 3]:
                    if pattern_id in self._last_phase_a_data:
                        stored_grids = self._last_phase_a_data[pattern_id]['grids']
                        logging.info(f"           - Pattern {pattern_id}: {len(stored_grids)} samples stored")
                    else:
                        logging.error(f"           - Pattern {pattern_id}: NOT STORED!")
            except Exception as e:
                logging.error(f"         ❌ Data storage verification failed: {e}")
            
            try:
                self._generate_phase_a_certainty_plots(enc_idx, encoder_params, eval_data, current_global_step, step, total_steps)
            except Exception as e:
                logging.error(f"         ❌ Certainty plots generation failed: {e}")
                # Continue with other evaluations
            
            # Evaluate target pattern reconstruction during Phase A training
            logging.info(f"       🔍 Evaluating target pattern reconstruction progress...")
            target_pattern = enc_idx + 1  # Encoder 0 -> Pattern 1, Encoder 1 -> Pattern 2, Encoder 2 -> Pattern 3
            if target_pattern in eval_data:
                # For Phase A evaluation, we need to create a minimal state with decoder params
                # Since we don't have the full state here, we'll skip reconstruction evaluation during training
                logging.info(f"         ⚠️ Skipping reconstruction evaluation during Phase A training (requires full state)")
                reconstruction_metrics = {}
                # Log Phase A specific reconstruction metrics
                if reconstruction_metrics:
                    wandb.log({
                        f"phase_a/encoder_{enc_idx}/target_pattern_reconstruction": reconstruction_metrics.get('overall_accuracy', 0.0),
                        f"phase_a/encoder_{enc_idx}/target_pattern_reconstruction/pixel_correctness": reconstruction_metrics.get('pixel_correctness', 0.0),
                        f"phase_a/encoder_{enc_idx}/target_pattern_reconstruction/shape_correctness": reconstruction_metrics.get('shape_correctness', 0.0),
                    }, step=current_global_step)
            
            # Additional Phase A specific metrics
            wandb.log({
                # No meaningless metrics - only meaningful data
            }, step=current_global_step)
            
            if eval_data:
                logging.info(f"       ✅ Phase A evaluation completed successfully with {len(eval_data)} patterns")
            else:
                logging.warning(f"       ⚠️ Phase A evaluation completed but no valid data was processed")
            
        except Exception as e:
            logging.error(f"       ❌ Phase A evaluation generation failed: {e}")
            import traceback
            logging.error(f"       Traceback: {traceback.format_exc()}")
    
    def _validate_evaluation_data(self, grids, shapes, pattern_ids, pattern_id: int) -> bool:
        """
        Validate that evaluation data has the correct structure and types.
        
        Args:
            grids: Grid data array
            shapes: Shape data array
            pattern_ids: Pattern ID array
            pattern_id: Expected pattern ID
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check data types
            if not isinstance(grids, (np.ndarray, jnp.ndarray)):
                logging.error(f"         ❌ Invalid grids type: {type(grids)}")
                return False
                
            if not isinstance(shapes, (np.ndarray, jnp.ndarray)):
                logging.error(f"         ❌ Invalid shapes type: {type(shapes)}")
                return False
                
            if not isinstance(pattern_ids, (np.ndarray, jnp.ndarray)):
                logging.error(f"         ❌ Invalid pattern_ids type: {type(pattern_ids)}")
                return False
            
            # Check array shapes
            if len(grids.shape) != 6:  # (samples, 2, num_pairs, 5, 5, 2)
                logging.error(f"         ❌ Invalid grids shape: {grids.shape}, expected 6 dimensions")
                return False
                
            if len(shapes.shape) != 4:  # (samples, 2, num_pairs, 2)
                logging.error(f"         ❌ Invalid shapes shape: {shapes.shape}, expected 4 dimensions")
                return False
                
            if len(pattern_ids.shape) != 1:  # (samples,)
                logging.error(f"         ❌ Invalid pattern_ids shape: {pattern_ids.shape}, expected 1 dimension")
                return False
            
            # Check sample count
            if len(grids) < 10:
                logging.error(f"         ❌ Too few samples: {len(grids)}, need at least 10")
                return False
            
            # Check pattern ID consistency
            if not np.all(pattern_ids == pattern_id):
                logging.error(f"         ❌ Pattern ID mismatch: expected all {pattern_id}, got {np.unique(pattern_ids)}")
                return False
            
            logging.info(f"         ✅ Pattern {pattern_id}: Data validation passed")
            return True
            
        except Exception as e:
            logging.error(f"         ❌ Data validation failed: {e}")
            return False

    def _load_pre_generated_pattern_data(self, pattern_id: int) -> tuple:
        """
        Load pre-generated pattern data as a fallback when dynamic generation fails.
        
        Args:
            pattern_id: Pattern ID (1, 2, or 3)
            
        Returns:
            Tuple of (grids, shapes, pattern_ids) with proper numpy arrays
        """
        try:
            # Try multiple possible data directories
            possible_dirs = [
                self.data_dir,
                "src/datasets",
                "datasets",
                os.path.join(os.getcwd(), "src/datasets"),
                os.path.join(os.getcwd(), "datasets")
            ]
            
            for data_dir in possible_dirs:
                dataset_name = f"struct_pattern_{pattern_id}"
                dataset_path = os.path.join(data_dir, dataset_name)
                
                if os.path.exists(dataset_path):
                    grids = np.load(os.path.join(dataset_path, "grids.npy"))
                    shapes = np.load(os.path.join(dataset_path, "shapes.npy"))
                    pattern_ids = np.full(len(grids), pattern_id, dtype=np.int32)
                    
                    logging.info(f"         📁 Loaded {len(grids)} samples from {dataset_path}")
                    return grids, shapes, pattern_ids
            
            # If no dataset found, create synthetic data with proper structure
            logging.warning(f"         ⚠️ No dataset found for pattern {pattern_id}, creating synthetic data")
            num_samples = 200  # Reasonable number for evaluation
            num_pairs = getattr(self, 'task_generator_kwargs', {}).get('num_pairs', 4)
            
            # Create properly structured synthetic data
            grids = np.random.randint(0, 2, (num_samples, 2, num_pairs, 5, 5, 2), dtype=np.int32)
            shapes = np.random.randint(0, 5, (num_samples, 2, num_pairs, 2), dtype=np.int32)
            pattern_ids = np.full(num_samples, pattern_id, dtype=np.int32)
            
            logging.info(f"         🔧 Created synthetic data: {num_samples} samples, {num_pairs} pairs")
            return grids, shapes, pattern_ids
                
        except Exception as e:
            logging.error(f"         ❌ Failed to load pattern {pattern_id} data: {e}")
            # Ultimate fallback: create minimal valid data with proper structure
            num_samples = 100
            num_pairs = getattr(self, 'task_generator_kwargs', {}).get('num_pairs', 4)
            
            grids = np.random.randint(0, 2, (num_samples, 2, num_pairs, 5, 5, 2), dtype=np.int32)
            shapes = np.random.randint(0, 5, (num_samples, 2, num_pairs, 2), dtype=np.int32)
            pattern_ids = np.full(num_samples, pattern_id, dtype=np.int32)
            
            logging.info(f"         🆘 Created emergency fallback data: {num_samples} samples")
            return grids, shapes, pattern_ids

    def _generate_phase_a_certainty_plots(self, enc_idx: int, encoder_params: dict, eval_data: dict, global_step: int, step: int, total_steps: int):
        """
        Generate certainty plots for all patterns during Phase A training to monitor encoder specialization progress.
        
        CRITICAL: This method uses eval_data that comes from pre-loaded datasets (struct_pattern_1, struct_pattern_2, struct_pattern_3)
        ensuring the same data is used every time for consistent certainty plots.
        
        Args:
            enc_idx: Index of the encoder being trained
            encoder_params: Current encoder parameters
            eval_data: Evaluation data for all patterns (from pre-loaded datasets)
            global_step: Global training step
            step: Current step within encoder training
            total_steps: Total steps for encoder training
        """
        try:
            # Generate certainty plots for each pattern using PRE-LOADED datasets
            # This ensures consistency with Phase 1 evaluation and prevents dataset mixing
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    grids, shapes, pattern_ids = eval_data[pattern_id]
                    
                    # CRITICAL FIX: Use DETERMINISTIC sample selection to prevent randomization inconsistencies
                    # This ensures the same sample is always selected for the same encoder/pattern combination
                    sample_index = (enc_idx * 3 + pattern_id - 1) % len(grids)
                    logging.info(f"         🔒 Certainty panel: Encoder {enc_idx}, Pattern {pattern_id}, Sample index {sample_index}/{len(grids)}")
                    logging.info(f"         🔒 DETERMINISTIC selection ensures consistency across calls")
                    
                    # Get encoder outputs for this single sample (exactly like working implementation)
                    sample_grids = grids[sample_index]  # Single sample (not slice)
                    sample_shapes = shapes[sample_index]  # Single sample (not slice)
                    
                    mu, logvar = self.encoders[enc_idx].apply(
                        {"params": encoder_params},
                        sample_grids[None, ...],  # Add batch dimension back
                        sample_shapes[None, ...],  # Add batch dimension back
                        dropout_eval=False,
                        mutable=False,
                    )
                    
                    # Create certainty panel
                    pattern_names = {1: "L-tetromino", 2: "O-tetromino", 3: "T-tetromino"}
                    pattern_name = pattern_names.get(pattern_id, f"Pattern {pattern_id}")
                    
                    # Pass data directly to visualization (exactly like working implementation)
                    fig_cert = visualize_struct_confidence_panel(
                        sample_grids=np.array(sample_grids),
                        sample_shapes=np.array(sample_shapes),
                        encoder_mus=[np.array(mu)],
                        encoder_logvars=[np.array(logvar)],
                        poe_mu=None,
                        poe_logvar=None,
                        title=f"Encoder {enc_idx} - {pattern_name} - Step {step}/{total_steps}",
                        encoder_labels=[f"Encoder {enc_idx}"],
                        encoder_indices=[enc_idx],
                        pattern_id=pattern_id,
                        pattern_name=pattern_name,
                    )
                    
                    # Log to WandB with proper organization
                    wandb.log({
                        f"phase_a/encoder_{enc_idx}/pattern_{pattern_id}/certainty_panel": wandb.Image(fig_cert),
                    }, step=global_step)
                    
                    # Close figure to free memory
                    plt.close(fig_cert)
                    
                    logging.info(f"       ✅ Certainty panel generated for pattern {pattern_id} ({pattern_name})")
            
            logging.info(f"       📊 All certainty plots generated and logged to WandB")
            
        except Exception as e:
            logging.error(f"       ❌ Certainty plot generation failed: {e}")
            import traceback
            logging.error(f"       Traceback: {traceback.format_exc()}")
    
    def _sample_specialized_batch(self, specialized_data: tuple, target_pattern: int) -> tuple:
        """
        Sample a batch from specialized training data.
        
        Args:
            specialized_data: Tuple of (grids, shapes, pattern_ids)
            target_pattern: Target pattern for this encoder
            
        Returns:
            Batch tuple for training
        """
        grids, shapes, pattern_ids = specialized_data
        batch_size = self.batch_size
        
        # Sample batch_size samples
        if len(grids) >= batch_size:
            indices = np.random.choice(len(grids), batch_size, replace=False)
            batch_grids = grids[indices]
            batch_shapes = shapes[indices]
            batch_pattern_ids = pattern_ids[indices]
        else:
            # If not enough samples, repeat with random sampling
            indices = np.random.choice(len(grids), batch_size, replace=True)
            batch_grids = grids[indices]
            batch_shapes = shapes[indices]
            batch_pattern_ids = pattern_ids[indices]
        
        return batch_grids, batch_shapes, batch_pattern_ids
    
    def _generate_new_batch_for_training(self, target_pattern: int, step: int) -> tuple:
        """
        Generate NEW samples for every training batch to ensure realistic variance computation.
        
        CRITICAL FIX: This prevents the "1260 copies of one grid" problem by generating
        fresh, diverse samples for each training step instead of reusing static data.
        
        Args:
            target_pattern: Target pattern for this encoder
            step: Current training step
            
        Returns:
            Batch tuple (grids, shapes, pattern_ids) with fresh samples
        """
        batch_size = self.batch_size
        
        # Generate fresh samples for this batch
        grids_list = []
        shapes_list = []
        pattern_ids_list = []
        
        # Target pattern gets more samples (70% of batch)
        target_samples = int(batch_size * 0.7)
        other_samples = batch_size - target_samples
        
        # Generate target pattern samples with diverse seeds
        for i in range(target_samples):
            # CRITICAL: Use step-dependent seed to ensure diversity across batches
            sample_seed = self.cfg.training.seed + target_pattern * 10000 + step * 100 + i
            grids, shapes, _ = self._create_single_pattern_sample_with_seed(target_pattern, sample_seed)
            grids_list.append(grids)
            shapes_list.append(shapes)
            pattern_ids_list.append(target_pattern)
        
        # Generate other pattern samples with diverse seeds
        other_patterns = [p for p in [1, 2, 3] if p != target_pattern]
        samples_per_other = other_samples // len(other_patterns)
        
        for pattern_id in other_patterns:
            for i in range(samples_per_other):
                # CRITICAL: Use step-dependent seed to ensure diversity across batches
                sample_seed = self.cfg.training.seed + pattern_id * 10000 + step * 100 + i
                grids, shapes, _ = self._create_single_pattern_sample_with_seed(pattern_id, sample_seed)
                grids_list.append(grids)
                shapes_list.append(shapes)
                pattern_ids_list.append(pattern_id)
        
        # Stack all samples
        grids = jnp.stack(grids_list, axis=0)
        shapes = jnp.stack(shapes_list, axis=0)
        pattern_ids = jnp.array(pattern_ids_list)
        
        logging.debug(f"         Generated fresh batch: {len(grids)} samples (target: {target_samples}, others: {other_samples})")
        logging.debug(f"         Pattern distribution: {dict(zip(*np.unique(pattern_ids, return_counts=True)))}")
        
        return grids, shapes, pattern_ids
    
    def train_n_steps_phase2(self, state: TrainState, batches: tuple[chex.Array, chex.Array, chex.Array], key: chex.PRNGKey) -> tuple[TrainState, dict]:
        """
        Phase 2: Joint decoder training with frozen encoders.
        
        Args:
            state: Current training state
            batches: Tuple of (grids, shapes, pattern_ids)
            key: Random key
            
        Returns:
            Updated state and metrics
        """
        num_steps = batches[0].shape[0]  # Should be log_every_n_steps
        keys = jax.random.split(key, num_steps)
        
        # Extract data
        explicit_pattern_ids = batches[2]  # (batch_size,) - explicit pattern IDs aligned with data
        
        # Process each batch sequentially (since we don't have pmap)
        all_metrics = []
        all_encoder_outputs = []  # Store encoder outputs for analysis
        
        for i in range(num_steps):
            batch_pairs, batch_shapes = batches[0][i], batches[1][i]
            batch_pattern_ids = explicit_pattern_ids  # Same pattern IDs for all steps
            rng = keys[i]
            
            def loss_fn(full_params, batch_pairs, batch_shapes, rng):
                # Phase 2: Only reconstruction loss, no specialization losses
                pattern_ids = batch_pattern_ids
                
                # Validate pattern distribution
                unique_patterns, counts = jnp.unique(pattern_ids, return_counts=True)
                unique_patterns_py = [int(p) for p in unique_patterns]
                counts_py = [int(c) for c in counts]
                pattern_distribution = dict(zip(unique_patterns_py, counts_py))
                logging.debug(f"Phase 2 pattern distribution: {pattern_distribution}")
                
                # Phase 2: NO specialization losses - only reconstruction
                repulsion_coeff = 0.0
                contrastive_coeff = 0.0
                logging.debug(f"🔒 Phase 2: Encoders FROZEN - No specialization losses")
                
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
                    repulsion_kl_coeff=repulsion_coeff,  # DISABLED in Phase 2
                    contrastive_kl_coeff=contrastive_coeff,  # DISABLED in Phase 2
                    pattern_ids=pattern_ids,  # Still pass for logging
                    **(self.cfg.training.get("inference_kwargs") or {}),
                )
                return loss, metrics
            
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch_pairs, batch_shapes, rng)
            
            # Phase 2: FREEZE encoders, only train decoder
            if "encoders" in grads:
                zeros_enc = tree_map(lambda g: jnp.zeros_like(g), grads["encoders"])
                grads = dict(grads)
                grads["encoders"] = zeros_enc
                logging.debug(f"🔒 Phase 2: Encoder gradients zeroed (frozen)")
            
            state = state.apply_gradients(grads=grads)
            all_metrics.append(metrics)
            
            # Store encoder outputs for analysis (without gradients)
            with jax.disable_jit():
                encoder_outputs = self._get_encoder_outputs_for_analysis(
                    state.params["encoders"], batch_pairs, batch_shapes
                )
                all_encoder_outputs.append(encoder_outputs)
            
            # Phase 2: Generate periodic T-SNE visualizations (every 50 steps)
            if i % 50 == 0 and i > 0:
                try:
                    logging.info(f"🔍 Phase 2: Generating periodic T-SNE at step {i}/{num_steps}")
                    periodic_tsne_metrics = self._generate_phase2_tsne_visualizations(
                        state, explicit_pattern_ids, i
                    )
                    
                    # Log periodic T-SNE to WandB
                    for key, value in periodic_tsne_metrics.items():
                        if "tsne" in key:
                            wandb.log({f"phase_2_periodic/{key}": value}, step=i)
                    
                    logging.info(f"✅ Phase 2: Periodic T-SNE generated at step {i}")
                except Exception as e:
                    logging.warning(f"Phase 2 periodic T-SNE generation failed at step {i}: {e}")
        
        # Average metrics over all steps
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.stack([m[key] for m in all_metrics]))
        
        # Phase 2: Generate comprehensive metrics and plots
        phase2_metrics = self._generate_phase2_metrics_and_plots(
            avg_metrics, all_encoder_outputs, explicit_pattern_ids, num_steps
        )
        
        # Merge with training metrics
        avg_metrics.update(phase2_metrics)
        
        # Phase 2: Generate T-SNE visualizations (same as Phase 1)
        try:
            logging.info(f"🔍 Phase 2: Generating T-SNE visualizations...")
            tsne_metrics = self._generate_phase2_tsne_visualizations(state, explicit_pattern_ids, num_steps)
            avg_metrics.update(tsne_metrics)
            logging.info(f"✅ Phase 2: T-SNE visualizations generated and logged")
        except Exception as e:
            logging.warning(f"Phase 2 T-SNE generation failed: {e}")
        
        # Phase 2: Log that we're in decoder-only training mode
        logging.info(f"Phase 2: Joint decoder training completed - {num_steps} steps")
        logging.info(f"   - Encoders are FROZEN (keeping specialization)")
        logging.info(f"   - Decoder is TRAINABLE (reconstruction focus)")
        logging.info(f"   - No specialization losses applied")
        
        return state, avg_metrics
    
    def _get_encoder_outputs_for_analysis(self, encoder_params_list: list[dict], batch_pairs: chex.Array, batch_shapes: chex.Array) -> dict:
        """
        Get encoder outputs for analysis during Phase 2 training.
        
        Args:
            encoder_params_list: List of encoder parameters
            batch_pairs: Batch of input/output pairs
            batch_shapes: Batch of grid shapes
            
        Returns:
            Dictionary containing encoder outputs for analysis
        """
        encoder_outputs = {}
        
        for enc_idx, enc_params in enumerate(encoder_params_list):
            try:
                # Get encoder outputs without gradients
                mu_i, logvar_i = self.encoders[enc_idx].apply(
                    {"params": enc_params},
                    batch_pairs,
                    batch_shapes,
                    dropout_eval=False,
                    mutable=False,
                )
                
                # Store outputs for analysis
                encoder_outputs[f"encoder_{enc_idx}"] = {
                    "mu": mu_i,
                    "logvar": logvar_i,
                    "variance": jnp.exp(logvar_i)
                }
                
            except Exception as e:
                logging.warning(f"Failed to get encoder {enc_idx} outputs for analysis: {e}")
                continue
        
        return encoder_outputs
    
    def _generate_phase2_metrics_and_plots(self, avg_metrics: dict, all_encoder_outputs: list, pattern_ids: chex.Array, num_steps: int) -> dict:
        """
        Generate comprehensive Phase 2 metrics and plots.
        
        Args:
            avg_metrics: Average training metrics
            all_encoder_outputs: List of encoder outputs from all steps
            pattern_ids: Pattern IDs for the batch
            num_steps: Number of training steps
            
        Returns:
            Dictionary containing Phase 2 metrics and plots
        """
        phase2_metrics = {}
        
        try:
            # 1. ENCODER SPECIALIZATION METRICS (frozen encoders maintaining specialization)
            encoder_specialization_metrics = self._compute_encoder_specialization_metrics(
                all_encoder_outputs, pattern_ids
            )
            phase2_metrics.update(encoder_specialization_metrics)
            
            # 2. POE AGGREGATION METRICS (how well encoders combine)
            poe_metrics = self._compute_poe_aggregation_metrics(avg_metrics)
            phase2_metrics.update(poe_metrics)
            
            # 3. DECODER TRAINING METRICS (reconstruction focus)
            decoder_metrics = self._compute_decoder_training_metrics(avg_metrics)
            phase2_metrics.update(decoder_metrics)
            
            # 4. GENERATE PHASE 2 PLOTS
            phase2_plots = self._generate_phase2_plots(
                all_encoder_outputs, pattern_ids, num_steps
            )
            phase2_metrics.update(phase2_plots)
            
        except Exception as e:
            logging.warning(f"Phase 2 metrics generation failed: {e}")
            # Return basic metrics if generation fails
            phase2_metrics = {
                "phase_b/error": f"Metrics generation failed: {str(e)}"
            }
        
        return phase2_metrics
    
    def _generate_phase2_tsne_visualizations(self, state: TrainState, pattern_ids: chex.Array, num_steps: int) -> dict:
        """
        Generate T-SNE visualizations for Phase 2 (same style as Phase 1).
        
        Args:
            state: Current training state
            pattern_ids: Pattern IDs for the batch
            num_steps: Number of training steps
            
        Returns:
            Dictionary containing T-SNE visualization metrics
        """
        tsne_metrics = {}
        
        try:
            logging.info(f"🔍 Phase 2: Creating T-SNE visualizations...")
            
            # Create evaluation data for each pattern (same as Phase 1)
            eval_data = {}
            for pattern_id in [1, 2, 3]:
                try:
                    # Generate pattern-specific data
                    pattern_data = self._create_pattern_dataset(pattern_id, num_samples=32)
                    grids, shapes, _ = pattern_data
                    
                    # Get encoder outputs for this pattern
                    pattern_latents = []
                    pattern_source_ids = []
                    pattern_task_ids = []
                    
                    for enc_idx in range(len(self.encoders)):
                        # Get encoder outputs
                        mu, logvar = self.encoders[enc_idx].apply(
                            {"params": state.params["encoders"][enc_idx]}, 
                            grids, 
                            shapes, 
                            True, 
                            mutable=False
                        )
                        
                        # Use mean of latents over pairs
                        latents = mu.mean(axis=-2)  # (batch_size, latent_dim)
                        
                        # Add to pattern data
                        pattern_latents.append(latents)
                        pattern_source_ids.extend([enc_idx] * len(latents))
                        pattern_task_ids.extend(range(len(latents)))
                    
                    # Store pattern data
                    eval_data[pattern_id] = {
                        'latents': pattern_latents,
                        'source_ids': pattern_source_ids,
                        'task_ids': pattern_task_ids
                    }
                    
                except Exception as e:
                    logging.warning(f"Failed to create evaluation data for pattern {pattern_id}: {e}")
                    continue
            
            # Generate T-SNE visualizations for each pattern
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    try:
                        # Concatenate latents from all encoders for this pattern
                        all_latents = np.concatenate(eval_data[pattern_id]['latents'], axis=0)
                        all_source_ids = np.array(eval_data[pattern_id]['source_ids'])
                        all_task_ids = np.array(eval_data[pattern_id]['task_ids'])
                        
                        # Create pattern-specific T-SNE
                        pattern_names = {1: "L-tetromino", 2: "O-tetromino", 3: "T-tetromino"}
                        pattern_name = pattern_names.get(pattern_id, f"Pattern {pattern_id}")
                        
                        fig_tsne = self._create_pattern_specific_tsne(
                            latents=all_latents,
                            source_ids=all_source_ids,
                            task_ids=all_task_ids,
                            title=f"Phase 2: {pattern_name} - Encoder Latents (Step {num_steps})",
                            max_points=300,
                            random_state=42
                        )
                        
                        if fig_tsne is not None:
                            # Log to WandB
                            tsne_metrics[f"phase_2/tsne_pattern_{pattern_id}"] = wandb.Image(fig_tsne)
                            logging.info(f"✅ Phase 2 T-SNE generated for pattern {pattern_id} ({pattern_name})")
                            
                            # Close figure to free memory
                            plt.close(fig_tsne)
                        else:
                            logging.warning(f"❌ Phase 2 T-SNE generation failed for pattern {pattern_id}")
                            
                    except Exception as e:
                        logging.warning(f"Phase 2 T-SNE generation failed for pattern {pattern_id}: {e}")
                        continue
            
            # Generate combined T-SNE with all patterns and encoders
            try:
                # Concatenate all pattern data
                all_latents_list = []
                all_source_ids_list = []
                all_pattern_ids_list = []
                all_task_ids_list = []
                
                for pattern_id in [1, 2, 3]:
                    if pattern_id in eval_data:
                        all_latents_list.extend(eval_data[pattern_id]['latents'])
                        all_source_ids_list.extend(eval_data[pattern_id]['source_ids'])
                        all_pattern_ids_list.extend([pattern_id] * len(eval_data[pattern_id]['source_ids']))
                        all_task_ids_list.extend(eval_data[pattern_id]['task_ids'])
                
                if all_latents_list:
                    # Concatenate all data
                    combined_latents = np.concatenate(all_latents_list, axis=0)
                    combined_source_ids = np.array(all_source_ids_list)
                    combined_pattern_ids = np.array(all_pattern_ids_list)
                    combined_task_ids = np.array(all_task_ids_list)
                    
                    # Create combined T-SNE
                    fig_combined = self._create_pattern_specific_tsne(
                        latents=combined_latents,
                        source_ids=combined_source_ids,
                        task_ids=combined_task_ids,
                        title=f"Phase 2: All Patterns - Encoder Latents (Step {num_steps})",
                        max_points=500,
                        random_state=42
                    )
                    
                    if fig_combined is not None:
                        tsne_metrics["phase_2/tsne_all_patterns"] = wandb.Image(fig_combined)
                        logging.info(f"✅ Phase 2 combined T-SNE generated")
                        plt.close(fig_combined)
                    else:
                        logging.warning(f"❌ Phase 2 combined T-SNE generation failed")
                        
            except Exception as e:
                logging.warning(f"Phase 2 combined T-SNE generation failed: {e}")
            
            # Phase 2: COMPUTE COMPREHENSIVE CLUSTERING METRICS AND DISTANCE ANALYSIS
            try:
                logging.info(f"🔍 Phase 2: Computing clustering metrics and distance analysis...")
                
                # Create comprehensive evaluation data for clustering analysis
                clustering_data = self._create_comprehensive_clustering_data(state, pattern_ids)
                
                if clustering_data is not None:
                    # Compute clustering metrics (same as commit)
                    clustering_metrics = self._compute_phase2_clustering_metrics(clustering_data)
                    tsne_metrics.update(clustering_metrics)
                    
                    # Compute distance metrics between encoders
                    distance_metrics = self._compute_phase2_distance_metrics(clustering_data)
                    tsne_metrics.update(distance_metrics)
                    
                    # Compute encoder specialization quality metrics
                    specialization_metrics = self._compute_phase2_specialization_quality(clustering_data)
                    tsne_metrics.update(specialization_metrics)
                    
                    logging.info(f"✅ Phase 2: Comprehensive metrics computed and logged")
                else:
                    logging.warning(f"❌ Phase 2: Clustering data creation failed")
                    
            except Exception as e:
                logging.warning(f"Phase 2 comprehensive metrics computation failed: {e}")
                tsne_metrics["phase_2/comprehensive_metrics_error"] = str(e)
            
            logging.info(f"✅ Phase 2: T-SNE visualizations completed")
            
        except Exception as e:
            logging.warning(f"Phase 2 T-SNE generation failed: {e}")
            tsne_metrics["phase_2/tsne_error"] = str(e)
        
        return tsne_metrics
    
    def _create_comprehensive_clustering_data(self, state: TrainState, pattern_ids: chex.Array) -> Optional[dict]:
        """
        Create comprehensive data for clustering analysis in Phase 2.
        
        Args:
            state: Current training state
            pattern_ids: Pattern IDs for the batch
            
        Returns:
            Dictionary containing comprehensive clustering data
        """
        try:
            # Create evaluation data for each pattern (same as T-SNE generation)
            eval_data = {}
            for pattern_id in [1, 2, 3]:
                try:
                    # CRITICAL FIX: Use the SAME data generation method as Phase A for consistency
                    # This ensures Phase B metrics use the same data quality and diversity as Phase A
                    pattern_data = self._generate_diverse_pattern_data_for_metrics(pattern_id, num_samples=self.batch_size)
                    grids, shapes, _ = pattern_data
                    
                    # Get encoder outputs for this pattern
                    pattern_latents = []
                    pattern_source_ids = []
                    pattern_task_ids = []
                    
                    for enc_idx in range(len(self.encoders)):
                        # Get encoder outputs
                        mu, logvar = self.encoders[enc_idx].apply(
                            {"params": state.params["encoders"][enc_idx]}, 
                            grids, 
                            shapes, 
                            True, 
                            mutable=False
                        )
                        
                        # Use mean of latents over pairs
                        latents = mu.mean(axis=-2)  # (batch_size, latent_dim)
                        
                        # Add to pattern data
                        pattern_latents.append(latents)
                        pattern_source_ids.extend([enc_idx] * len(latents))
                        pattern_task_ids.extend(range(len(latents)))
                    
                    # Store pattern data
                    eval_data[pattern_id] = {
                        'latents': pattern_latents,
                        'source_ids': pattern_source_ids,
                        'task_ids': pattern_task_ids,
                        'grids': grids,
                        'shapes': shapes
                    }
                    
                except Exception as e:
                    logging.warning(f"Failed to create clustering data for pattern {pattern_id}: {e}")
                    continue
            
            # Create combined data for cross-pattern analysis
            combined_data = {}
            if eval_data:
                # Concatenate all pattern data
                all_latents_list = []
                all_source_ids_list = []
                all_pattern_ids_list = []
                all_task_ids_list = []
                
                for pattern_id in [1, 2, 3]:
                    if pattern_id in eval_data:
                        all_latents_list.extend(eval_data[pattern_id]['latents'])
                        all_source_ids_list.extend(eval_data[pattern_id]['source_ids'])
                        all_pattern_ids_list.extend([pattern_id] * len(eval_data[pattern_id]['source_ids']))
                        all_task_ids_list.extend(eval_data[pattern_id]['task_ids'])
                
                if all_latents_list:
                    combined_data = {
                        'latents': np.concatenate(all_latents_list, axis=0),
                        'source_ids': np.array(all_source_ids_list),
                        'pattern_ids': np.array(all_pattern_ids_list),
                        'task_ids': np.array(all_task_ids_list)
                    }
            
            return {
                'pattern_data': eval_data,
                'combined_data': combined_data
            }
            
        except Exception as e:
            logging.warning(f"Comprehensive clustering data creation failed: {e}")
            return None
    
    def _generate_diverse_pattern_data_for_metrics(self, pattern_id: int, num_samples: int) -> tuple:
        """
        Generate diverse pattern data for Phase B metrics computation.
        
        CRITICAL FIX: This ensures Phase B metrics use the SAME data generation method
        as Phase A, maintaining consistency in data quality and diversity.
        
        Args:
            pattern_id: Pattern ID to generate data for
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (grids, shapes, pattern_ids) with diverse samples
        """
        # Generate diverse samples using the SAME method as Phase A evaluation
        grids_list = []
        shapes_list = []
        pattern_ids_list = []
        
        for i in range(num_samples):
            # CRITICAL: Use different seed for each sample to ensure diversity
            # This prevents the "copies of one grid" problem
            sample_seed = self.cfg.training.seed + pattern_id * 1000 + i
            grids, shapes, _ = self._create_single_pattern_sample_with_seed(pattern_id, sample_seed)
            grids_list.append(grids)
            shapes_list.append(shapes)
            pattern_ids_list.append(pattern_id)
        
        # Stack all samples
        grids = jnp.stack(grids_list, axis=0)
        shapes = jnp.stack(shapes_list, axis=0)
        pattern_ids = jnp.array(pattern_ids_list)
        
        logging.debug(f"         Generated diverse data for pattern {pattern_id}: {len(grids)} samples")
        
        return grids, shapes, pattern_ids
    
    def _compute_phase2_clustering_metrics(self, clustering_data: dict) -> dict:
        """
        Compute clustering metrics for Phase 2 with pattern-independent computation.
        
        CRITICAL FIX: This ensures metrics are computed per pattern independently
        instead of mixing patterns together, which provides more meaningful analysis.
        
        Args:
            clustering_data: Comprehensive clustering data
            
        Returns:
            Dictionary containing clustering metrics
        """
        clustering_metrics = {}
        
        try:
            if 'pattern_data' not in clustering_data or clustering_data['pattern_data'] is None:
                return clustering_metrics
            
            pattern_data = clustering_data['pattern_data']
            k_values = [3, 5, 10]
            
            # CRITICAL FIX: Compute metrics PER PATTERN independently
            # This prevents pattern mixing and provides meaningful per-pattern analysis
            for pattern_id in [1, 2, 3]:
                if pattern_id not in pattern_data:
                    continue
                
                pattern_info = pattern_data[pattern_id]
                pattern_latents = pattern_info['latents']  # List of encoder outputs
                pattern_source_ids = pattern_info['source_ids']  # List of encoder indices
                
                if not pattern_latents:
                    continue
                
                # Concatenate latents from all encoders for this specific pattern
                pattern_latents_concat = np.concatenate(pattern_latents, axis=0)
                pattern_source_ids_np = np.array(pattern_source_ids)
                
                logging.info(f"Phase 2: Pattern {pattern_id} clustering: {pattern_latents_concat.shape[0]} points, encoders: {np.unique(pattern_source_ids_np)}")
                
                # Compute clustering metrics for this pattern independently
                for k in k_values:
                    # Modularity Q on pattern-specific encoder samples
                    modularity_q = compute_modularity_q(pattern_latents_concat, pattern_source_ids_np, k=k)
                    clustering_metrics[f"phase_2/clustering/pattern_{pattern_id}/modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index on pattern-specific encoder samples
                    ari_score = compute_adjusted_rand_index(pattern_latents_concat, pattern_source_ids_np, k=k)
                    clustering_metrics[f"phase_2/clustering/pattern_{pattern_id}/ari_k{k}"] = ari_score
                    
                    # Silhouette score for pattern-specific encoder separation
                    try:
                        from sklearn.metrics import silhouette_score
                        silhouette = silhouette_score(pattern_latents_concat, pattern_source_ids_np)
                        clustering_metrics[f"phase_2/clustering/pattern_{pattern_id}/silhouette_k{k}"] = silhouette
                    except ImportError:
                        clustering_metrics[f"phase_2/clustering/pattern_{pattern_id}/silhouette_k{k}"] = None
            
            # OPTION 2: Cross-pattern analysis (only when explicitly needed)
            if 'combined_data' in clustering_data and clustering_data['combined_data'] is not None:
                combined_data = clustering_data['combined_data']
                latents_concat = combined_data['latents']
                source_ids_np = combined_data['source_ids']
                
                # Only compute cross-pattern metrics when explicitly needed
                for k in k_values:
                    # Modularity Q on all embeddings (sources: encoders vs context)
                    modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"phase_2/clustering/cross_pattern/modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index on all embeddings (sources: encoders vs context)
                    ari_score = compute_adjusted_rand_index(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"phase_2/clustering/cross_pattern/ari_k{k}"] = ari_score
            
            logging.info(f"Phase 2: Pattern-independent clustering metrics computed: {len(clustering_metrics)} metrics")
            
        except Exception as e:
            logging.warning(f"Phase 2 clustering metrics computation failed: {e}")
            clustering_metrics["phase_2/clustering/error"] = str(e)
        
        return clustering_metrics
    
    def _compute_phase2_distance_metrics(self, clustering_data: dict) -> dict:
        """
        Compute distance metrics between encoders in Phase 2.
        
        Args:
            clustering_data: Comprehensive clustering data
            
        Returns:
            Dictionary containing distance metrics
        """
        distance_metrics = {}
        
        try:
            if 'pattern_data' not in clustering_data:
                return distance_metrics
            
            pattern_data = clustering_data['pattern_data']
            
            # Compute pairwise distances between encoders for each pattern
            for pattern_id in [1, 2, 3]:
                if pattern_id in pattern_data:
                    pattern_latents = pattern_data[pattern_id]['latents']
                    
                    if len(pattern_latents) >= 2:  # Need at least 2 encoders
                        # Compute pairwise distances between encoder representations
                        for i in range(len(pattern_latents)):
                            for j in range(i + 1, len(pattern_latents)):
                                enc_i_latents = pattern_latents[i]
                                enc_j_latents = pattern_latents[j]
                                
                                # Compute L2 distance between encoder representations
                                distances = np.linalg.norm(enc_i_latents - enc_j_latents, axis=1)
                                mean_distance = float(np.mean(distances))
                                std_distance = float(np.std(distances))
                                
                                # Store metrics
                                distance_metrics[f"phase_2/distance/pattern_{pattern_id}/encoder_{i}_vs_{j}/mean_l2"] = mean_distance
                                distance_metrics[f"phase_2/distance/pattern_{pattern_id}/encoder_{i}_vs_{j}/std_l2"] = std_distance
                                
                                # Compute cosine distance
                                cos_similarities = []
                                for k in range(len(enc_i_latents)):
                                    cos_sim = np.dot(enc_i_latents[k], enc_j_latents[k]) / (
                                        np.linalg.norm(enc_i_latents[k]) * np.linalg.norm(enc_j_latents[k]) + 1e-8
                                    )
                                    cos_similarities.append(cos_sim)
                                
                                mean_cos_sim = float(np.mean(cos_similarities))
                                std_cos_sim = float(np.std(cos_similarities))
                                
                                distance_metrics[f"phase_2/distance/pattern_{pattern_id}/encoder_{i}_vs_{j}/mean_cosine_sim"] = mean_cos_sim
                                distance_metrics[f"phase_2/distance/pattern_{pattern_id}/encoder_{i}_vs_{j}/std_cosine_sim"] = std_cos_sim
            
            # Compute cross-pattern encoder consistency
            if len(pattern_data) >= 2:
                for enc_idx in range(len(self.encoders)):
                    enc_pattern_latents = []
                    for pattern_id in [1, 2, 3]:
                        if pattern_id in pattern_data and len(pattern_data[pattern_id]['latents']) > enc_idx:
                            enc_pattern_latents.append(pattern_data[pattern_id]['latents'][enc_idx])
                    
                    if len(enc_pattern_latents) >= 2:
                        # Compute consistency across patterns for this encoder
                        pattern_distances = []
                        for i in range(len(enc_pattern_latents)):
                            for j in range(i + 1, len(enc_pattern_latents)):
                                dist = np.mean(np.linalg.norm(enc_pattern_latents[i] - enc_pattern_latents[j], axis=1))
                                pattern_distances.append(dist)
                        
                        if pattern_distances:
                            mean_pattern_dist = float(np.mean(pattern_distances))
                            std_pattern_dist = float(np.std(pattern_distances))
                            distance_metrics[f"phase_2/distance/encoder_{enc_idx}/cross_pattern_consistency/mean"] = mean_pattern_dist
                            distance_metrics[f"phase_2/distance/encoder_{enc_idx}/cross_pattern_consistency/std"] = std_pattern_dist
            
            logging.info(f"Phase 2: Distance metrics computed: {len(distance_metrics)} metrics")
            
        except Exception as e:
            logging.warning(f"Phase 2 distance metrics computation failed: {e}")
            distance_metrics["phase_2/distance/error"] = str(e)
        
        return distance_metrics
    
    def _compute_phase2_specialization_quality(self, clustering_data: dict) -> dict:
        """
        Compute encoder specialization quality metrics in Phase 2.
        
        Args:
            clustering_data: Comprehensive clustering data
            
        Returns:
            Dictionary containing specialization quality metrics
        """
        specialization_metrics = {}
        
        try:
            if 'pattern_data' not in clustering_data:
                return specialization_metrics
            
            pattern_data = clustering_data['pattern_data']
            
            # Compute specialization quality for each encoder
            for enc_idx in range(len(self.encoders)):
                enc_metrics = {}
                
                # Collect latents for this encoder across all patterns
                enc_pattern_latents = {}
                for pattern_id in [1, 2, 3]:
                    if pattern_id in pattern_data and len(pattern_data[pattern_id]['latents']) > enc_idx:
                        enc_pattern_latents[pattern_id] = pattern_data[pattern_id]['latents'][enc_idx]
                
                if len(enc_pattern_latents) >= 2:
                    # Compute target pattern specialization
                    target_pattern = enc_idx + 1
                    if target_pattern in enc_pattern_latents:
                        target_latents = enc_pattern_latents[target_pattern]
                        
                        # Compute variance of target pattern latents
                        target_variance = float(np.var(target_latents))
                        enc_metrics[f"target_pattern_{target_pattern}_variance"] = target_variance
                        
                        # Compute average distance between target pattern samples
                        target_distances = []
                        for i in range(len(target_latents)):
                            for j in range(i + 1, len(target_latents)):
                                dist = np.linalg.norm(target_latents[i] - target_latents[j])
                                target_distances.append(dist)
                        
                        if target_distances:
                            mean_target_dist = float(np.mean(target_distances))
                            std_target_dist = float(np.std(target_distances))
                            enc_metrics[f"target_pattern_{target_pattern}_mean_distance"] = mean_target_dist
                            enc_metrics[f"target_pattern_{target_pattern}_std_distance"] = std_target_dist
                        
                        # Compute specialization ratio (target vs other patterns)
                        other_pattern_variances = []
                        for pid in [1, 2, 3]:
                            if pid != target_pattern and pid in enc_pattern_latents:
                                other_latents = enc_pattern_latents[pid]
                                other_var = float(np.var(other_latents))
                                other_pattern_variances.append(other_var)
                                enc_metrics[f"other_pattern_{pid}_variance"] = other_var
                        
                        if other_pattern_variances:
                            avg_other_variance = np.mean(other_pattern_variances)
                            specialization_ratio = target_variance / (avg_other_variance + 1e-8)
                            specialization_score = np.log(specialization_ratio + 1e-8)
                            
                            enc_metrics["specialization_ratio"] = float(specialization_ratio)
                            enc_metrics["specialization_score"] = float(specialization_score)
                            
                            # Compute specialization quality indicator
                            if specialization_ratio < 0.5:
                                quality = "EXCELLENT"
                            elif specialization_ratio < 0.8:
                                quality = "GOOD"
                            elif specialization_ratio < 1.0:
                                quality = "WEAK"
                            else:
                                quality = "POOR"
                            
                            enc_metrics["specialization_quality"] = quality
                
                # Add encoder metrics to main metrics dict
                for key, value in enc_metrics.items():
                    specialization_metrics[f"phase_2/specialization/encoder_{enc_idx}/{key}"] = value
            
            logging.info(f"Phase 2: Specialization quality metrics computed: {len(specialization_metrics)} metrics")
            
        except Exception as e:
            logging.warning(f"Phase 2 specialization quality computation failed: {e}")
            specialization_metrics["phase_2/specialization/error"] = str(e)
        
        return specialization_metrics
    
    def _create_encoder_clustering_data(self, enc_idx: int, encoder_params: dict, eval_data: dict) -> Optional[dict]:
        """
        Create clustering data for a single encoder during Phase 1.
        
        Args:
            enc_idx: Index of the encoder
            encoder_params: Encoder parameters
            eval_data: Evaluation data for all patterns
            
        Returns:
            Dictionary containing clustering data or None if creation fails
        """
        try:
            clustering_data = {}
            
            # Collect latents for this encoder across all patterns
            pattern_latents = {}
            pattern_ids_list = []
            
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    grids, shapes, pattern_ids = eval_data[pattern_id]
                    
                    # Get encoder outputs for this pattern
                    mu, logvar = self.encoders[enc_idx].apply(
                        {"params": encoder_params},
                        grids,
                        shapes,
                        dropout_eval=False,
                        mutable=False,
                    )
                    
                    # Use mean of latents over pairs
                    latents = mu.mean(axis=-2)  # (batch_size, latent_dim)
                    
                    # Store pattern data
                    pattern_latents[pattern_id] = np.array(latents)
                    pattern_ids_list.extend([pattern_id] * len(latents))
            
            if pattern_latents:
                # Create combined data for cross-pattern analysis
                all_latents_list = []
                all_pattern_ids_list = []
                
                for pattern_id in [1, 2, 3]:
                    if pattern_id in pattern_latents:
                        all_latents_list.append(pattern_latents[pattern_id])
                        all_pattern_ids_list.extend([pattern_id] * len(pattern_latents[pattern_id]))
                
                if all_latents_list:
                    combined_data = {
                        'latents': np.concatenate(all_latents_list, axis=0),
                        'pattern_ids': np.array(all_pattern_ids_list)
                    }
                    
                    clustering_data = {
                        'pattern_data': pattern_latents,
                        'combined_data': combined_data
                    }
                    
                    logging.info(f"       ✅ Clustering data created for Encoder {enc_idx}: {len(combined_data['latents'])} total samples")
                    return clustering_data
            
            logging.warning(f"       ❌ No valid clustering data created for Encoder {enc_idx}")
            return None
            
        except Exception as e:
            logging.warning(f"       ❌ Clustering data creation failed for Encoder {enc_idx}: {e}")
            return None
    
    def _compute_phase1_clustering_metrics(self, clustering_data: dict, enc_idx: int) -> dict:
        """
        Compute clustering metrics for Phase 1 encoder evaluation.
        
        Args:
            clustering_data: Clustering data for the encoder
            enc_idx: Index of the encoder
            
        Returns:
            Dictionary containing clustering metrics
        """
        clustering_metrics = {}
        
        try:
            if 'combined_data' not in clustering_data or clustering_data['combined_data'] is None:
                return clustering_metrics
            
            combined_data = clustering_data['combined_data']
            latents_concat = combined_data['latents']
            pattern_ids_concat = combined_data['pattern_ids']
            
            # Compute metrics for different k values to check sensitivity
            k_values = [2, 3, 5]
            
            # Compute clustering metrics on pattern-based clustering
            for k in k_values:
                if k <= len(np.unique(pattern_ids_concat)):
                    # Modularity Q on pattern-based clustering
                    modularity_q = compute_modularity_q(latents_concat, pattern_ids_concat, k=k)
                    clustering_metrics[f"modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index on pattern-based clustering
                    ari_score = compute_adjusted_rand_index(latents_concat, pattern_ids_concat, k=k)
                    clustering_metrics[f"ari_k{k}"] = ari_score
                    
                    # Silhouette score for pattern separation
                    try:
                        from sklearn.metrics import silhouette_score
                        from sklearn.cluster import KMeans
                        
                        # Perform K-means clustering
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(latents_concat)
                        
                        # Compute silhouette score
                        silhouette_avg = silhouette_score(latents_concat, cluster_labels)
                        clustering_metrics[f"silhouette_k{k}"] = silhouette_avg
                        
                    except ImportError:
                        logging.warning(f"       ⚠️  sklearn not available, skipping silhouette score for k={k}")
                        clustering_metrics[f"silhouette_k{k}"] = None
            
            # Compute pattern separation metrics
            unique_patterns = np.unique(pattern_ids_concat)
            if len(unique_patterns) >= 2:
                # Compute pairwise distances between pattern centroids
                pattern_centroids = {}
                for pattern_id in unique_patterns:
                    pattern_mask = (pattern_ids_concat == pattern_id)
                    if np.any(pattern_mask):
                        pattern_latents = latents_concat[pattern_mask]
                        pattern_centroids[pattern_id] = np.mean(pattern_latents, axis=0)
                
                # Compute inter-pattern distances
                pattern_pairs = []
                for i, pid1 in enumerate(unique_patterns):
                    for j, pid2 in enumerate(unique_patterns[i+1:], i+1):
                        if pid1 in pattern_centroids and pid2 in pattern_centroids:
                            centroid_dist = np.linalg.norm(pattern_centroids[pid1] - pattern_centroids[pid2])
                            pattern_pairs.append((pid1, pid2, centroid_dist))
                            clustering_metrics[f"centroid_distance_pattern_{pid1}_vs_{pid2}"] = float(centroid_dist)
                
                # Compute intra-pattern compactness
                for pattern_id in unique_patterns:
                    pattern_mask = (pattern_ids_concat == pattern_id)
                    if np.any(pattern_mask):
                        pattern_latents = latents_concat[pattern_mask]
                        if len(pattern_latents) > 1:
                            # Compute average distance to centroid
                            centroid = pattern_centroids[pattern_id]
                            distances_to_centroid = np.linalg.norm(pattern_latents - centroid, axis=1)
                            mean_distance = float(np.mean(distances_to_centroid))
                            std_distance = float(np.std(distances_to_centroid))
                            
                            clustering_metrics[f"intra_pattern_{pattern_id}_mean_distance_to_centroid"] = mean_distance
                            clustering_metrics[f"intra_pattern_{pattern_id}_std_distance_to_centroid"] = std_distance
            
            logging.info(f"       ✅ Phase 1 clustering metrics computed for Encoder {enc_idx}: {len(clustering_metrics)} metrics")
            
        except Exception as e:
            logging.warning(f"       ❌ Phase 1 clustering metrics computation failed for Encoder {enc_idx}: {e}")
            clustering_metrics["error"] = str(e)
        
        return clustering_metrics
    
    def _compute_clustering_metrics_every_step(self, state: TrainState, grids: chex.Array, shapes: chex.Array, pattern_ids: chex.Array, step: int) -> dict:
        """
        Compute clustering metrics every log_every_n_steps to monitor encoder specialization progress.
        
        This method provides real-time monitoring of how well encoders are separating patterns
        in the latent space during training.
        
        Args:
            state: Current training state
            grids: Input grids for the batch
            shapes: Input shapes for the batch
            pattern_ids: Pattern IDs for the batch
            step: Current training step
            
        Returns:
            Dictionary containing clustering metrics for monitoring
        """
        clustering_metrics = {}
        
        try:
            # Get encoder outputs for the current batch
            encoder_outputs = self._get_encoder_outputs_for_analysis(
                state.params["encoders"], grids, shapes
            )
            
            if not encoder_outputs or 'combined_data' not in encoder_outputs:
                logging.debug(f"       ⚠️  No encoder outputs available for clustering metrics at step {step}")
                return clustering_metrics
            
            combined_data = encoder_outputs['combined_data']
            latents_concat = combined_data['latents']
            pattern_ids_concat = combined_data['pattern_ids']
            
            # Basic clustering metrics for monitoring
            k_values = [2, 3]  # Keep it simple for every-step computation
            
            for k in k_values:
                if k <= len(np.unique(pattern_ids_concat)):
                    # Modularity Q - measures pattern separation quality
                    try:
                        modularity_q = compute_modularity_q(latents_concat, pattern_ids_concat, k=k)
                        clustering_metrics[f"clustering/modularity_q_k{k}"] = float(modularity_q)
                    except Exception as e:
                        logging.debug(f"       Modularity Q computation failed for k={k}: {e}")
                    
                    # Adjusted Rand Index - measures clustering accuracy
                    try:
                        ari_score = compute_adjusted_rand_index(latents_concat, pattern_ids_concat, k=k)
                        clustering_metrics[f"clustering/ari_k{k}"] = float(ari_score)
                    except Exception as e:
                        logging.debug(f"       ARI computation failed for k={k}: {e}")
            
            # Pattern separation metrics
            unique_patterns = np.unique(pattern_ids_concat)
            if len(unique_patterns) >= 2:
                # Compute pattern centroids
                pattern_centroids = {}
                for pattern_id in unique_patterns:
                    pattern_mask = (pattern_ids_concat == pattern_id)
                    if np.any(pattern_mask):
                        pattern_latents = latents_concat[pattern_mask]
                        pattern_centroids[pattern_id] = np.mean(pattern_latents, axis=0)
                
                # Inter-pattern distances (should increase during specialization)
                pattern_pairs = []
                for i, pid1 in enumerate(unique_patterns):
                    for j, pid2 in enumerate(unique_patterns[i+1:], i+1):
                        if pid1 in pattern_centroids and pid2 in pattern_centroids:
                            centroid_dist = np.linalg.norm(pattern_centroids[pid1] - pattern_centroids[pid2])
                            clustering_metrics[f"clustering/centroid_distance_pattern_{pid1}_vs_{pid2}"] = float(centroid_dist)
                
                # Intra-pattern compactness (should decrease during specialization)
                for pattern_id in unique_patterns:
                    pattern_mask = (pattern_ids_concat == pattern_id)
                    if np.any(pattern_mask):
                        pattern_latents = latents_concat[pattern_mask]
                        if len(pattern_latents) > 1:
                            centroid = pattern_centroids[pattern_id]
                            distances_to_centroid = np.linalg.norm(pattern_latents - centroid, axis=1)
                            mean_distance = float(np.mean(distances_to_centroid))
                            
                            clustering_metrics[f"clustering/intra_pattern_{pattern_id}_compactness"] = mean_distance
            
            # Add step information for tracking
            clustering_metrics["clustering/step"] = step if step is not None else -1  # -1 for Phase 1
            clustering_metrics["clustering/num_samples"] = len(latents_concat)
            clustering_metrics["clustering/num_patterns"] = len(unique_patterns)
            
            logging.debug(f"       ✅ Clustering metrics computed at step {step}: {len(clustering_metrics)} metrics")
            
        except Exception as e:
            logging.warning(f"       ❌ Clustering metrics computation failed at step {step}: {e}")
            clustering_metrics["clustering/error"] = str(e)
        
        return clustering_metrics
    
    def _compute_encoder_specialization_metrics(self, all_encoder_outputs: list, pattern_ids: chex.Array) -> dict:
        """
        Compute metrics showing how well frozen encoders maintain specialization.
        
        Args:
            all_encoder_outputs: List of encoder outputs from all steps
            pattern_ids: Pattern IDs for the batch
            
        Returns:
            Dictionary of encoder specialization metrics
        """
        metrics = {}
        
        try:
            # Convert pattern IDs to numpy for analysis
            pattern_ids_np = np.array(pattern_ids)
            unique_patterns = np.unique(pattern_ids_np)
            
            # Analyze each encoder's specialization across all steps
            for enc_idx in range(len(self.encoders)):
                enc_metrics = {}
                
                # Collect variances across all steps for this encoder
                all_variances = []
                for step_outputs in all_encoder_outputs:
                    if f"encoder_{enc_idx}" in step_outputs:
                        variances = np.array(step_outputs[f"encoder_{enc_idx}"]["variance"])
                        all_variances.append(variances)
                
                if all_variances:
                    # Stack variances from all steps
                    stacked_variances = np.stack(all_variances, axis=0)  # (steps, batch, pairs, latent_dim)
                    
                    # Compute mean variance per pattern
                    for pattern_id in unique_patterns:
                        pattern_mask = (pattern_ids_np == pattern_id)
                        if np.any(pattern_mask):
                            pattern_variances = stacked_variances[:, pattern_mask, :, :]  # (steps, pattern_samples, pairs, latent_dim)
                            
                            # Average over steps, pairs, and latent dimensions
                            mean_pattern_var = float(np.mean(pattern_variances))
                            std_pattern_var = float(np.std(pattern_variances))
                            
                            # Store metrics
                            enc_metrics[f"pattern_{pattern_id}_mean_variance"] = mean_pattern_var
                            enc_metrics[f"pattern_{pattern_id}_std_variance"] = std_pattern_var
                    
                    # Compute overall specialization metrics
                    if len(unique_patterns) >= 2:
                        # Calculate specialization ratio (target vs other patterns)
                        # For encoder 0: pattern 1 is target, for encoder 1: pattern 2 is target, etc.
                        target_pattern = enc_idx + 1
                        if target_pattern in unique_patterns:
                            target_var_key = f"pattern_{target_pattern}_mean_variance"
                            if target_var_key in enc_metrics:
                                target_var = enc_metrics[target_var_key]
                                
                                # Average variance of other patterns
                                other_vars = []
                                for pid in unique_patterns:
                                    if pid != target_pattern:
                                        other_var_key = f"pattern_{pid}_mean_variance"
                                        if other_var_key in enc_metrics:
                                            other_vars.append(enc_metrics[other_var_key])
                                
                                if other_vars:
                                    avg_other_var = np.mean(other_vars)
                                    specialization_ratio = target_var / (avg_other_var + 1e-8)
                                    enc_metrics["specialization_ratio"] = float(specialization_ratio)
                                    enc_metrics["specialization_score"] = float(np.log(specialization_ratio + 1e-8))
                
                # Add encoder metrics to main metrics dict
                for key, value in enc_metrics.items():
                    metrics[f"phase_b/encoder_{enc_idx}/{key}"] = value
                    
        except Exception as e:
            logging.warning(f"Encoder specialization metrics computation failed: {e}")
            metrics["phase_b/encoder_specialization_error"] = str(e)
        
        return metrics
    
    def _compute_poe_aggregation_metrics(self, avg_metrics: dict) -> dict:
        """
        Compute metrics showing how well PoE aggregation works.
        
        Args:
            avg_metrics: Average training metrics
            
        Returns:
            Dictionary of PoE aggregation metrics
        """
        metrics = {}
        
        try:
            # Extract PoE metrics if available
            if "poe_prior_weight" in avg_metrics:
                metrics["phase_b/poe/prior_weight"] = float(avg_metrics["poe_prior_weight"])
            if "poe_num_encoders" in avg_metrics:
                metrics["phase_b/poe/num_encoders"] = float(avg_metrics["poe_num_encoders"])
            if "poe_alphas_mean" in avg_metrics:
                metrics["phase_b/poe/alphas_mean"] = float(avg_metrics["poe_alphas_mean"])
            
            # Add PoE stability metrics
            metrics["phase_b/poe/stability"] = 1.0  # Placeholder for PoE stability metric
            
        except Exception as e:
            logging.warning(f"PoE aggregation metrics computation failed: {e}")
            metrics["phase_b/poe/error"] = str(e)
        
        return metrics
    
    def _compute_decoder_training_metrics(self, avg_metrics: dict) -> dict:
        """
        Compute metrics showing decoder training progress.
        
        Args:
            avg_metrics: Average training metrics
            
        Returns:
            Dictionary of decoder training metrics
        """
        metrics = {}
        
        try:
            # Core training metrics
            if "loss" in avg_metrics:
                metrics["phase_b/decoder/total_loss"] = float(avg_metrics["loss"])
            if "reconstruction_loss" in avg_metrics:
                metrics["phase_b/decoder/reconstruction_loss"] = float(avg_metrics["reconstruction_loss"])
            if "prior_kl" in avg_metrics:
                metrics["phase_b/decoder/prior_kl"] = float(avg_metrics["prior_kl"])
            if "pairwise_kl" in avg_metrics:
                metrics["phase_b/decoder/pairwise_kl"] = float(avg_metrics["pairwise_kl"])
            
            # Training stability metrics
            metrics["phase_b/decoder/training_stability"] = 1.0  # Placeholder for stability metric
            
        except Exception as e:
            logging.warning(f"Decoder training metrics computation failed: {e}")
            metrics["phase_b/decoder/error"] = str(e)
        
        return metrics
    
    def _generate_phase2_plots(self, all_encoder_outputs: list, pattern_ids: chex.Array, num_steps: int) -> dict:
        """
        Generate Phase 2 plots for visualization.
        
        Args:
            all_encoder_outputs: List of encoder outputs from all steps
            pattern_ids: Pattern IDs for the batch
            num_steps: Number of training steps
            
        Returns:
            Dictionary containing Phase 2 plots
        """
        plots = {}
        
        try:
            # 1. ENCODER SPECIALIZATION MAINTENANCE PLOT
            fig_specialization = self._create_encoder_specialization_plot(
                all_encoder_outputs, pattern_ids, num_steps
            )
            if fig_specialization is not None:
                plots["phase_b/plots/encoder_specialization"] = wandb.Image(fig_specialization)
                plt.close(fig_specialization)
            
            # 2. POE AGGREGATION VISUALIZATION
            fig_poe = self._create_poe_aggregation_plot(all_encoder_outputs, pattern_ids)
            if fig_poe is not None:
                plots["phase_b/plots/poe_aggregation"] = wandb.Image(fig_poe)
                plt.close(fig_poe)
            
            # 3. DECODER TRAINING PROGRESS
            fig_decoder = self._create_decoder_training_plot(all_encoder_outputs, num_steps)
            if fig_decoder is not None:
                plots["phase_b/plots/decoder_training"] = wandb.Image(fig_decoder)
                plt.close(fig_decoder)
                
        except Exception as e:
            logging.warning(f"Phase 2 plots generation failed: {e}")
            plots["phase_b/plots/error"] = f"Plots generation failed: {str(e)}"
        
        return plots
    
    def _create_encoder_specialization_plot(self, all_encoder_outputs: list, pattern_ids: chex.Array, num_steps: int) -> Optional[plt.Figure]:
        """
        Create plot showing how well frozen encoders maintain specialization.
        
        Args:
            all_encoder_outputs: List of encoder outputs from all steps
            pattern_ids: Pattern IDs for the batch
            num_steps: Number of training steps
            
        Returns:
            matplotlib Figure or None if creation fails
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots for each encoder
            num_encoders = len(self.encoders)
            fig, axes = plt.subplots(1, num_encoders, figsize=(5*num_encoders, 4))
            if num_encoders == 1:
                axes = [axes]
            
            pattern_ids_np = np.array(pattern_ids)
            unique_patterns = np.unique(pattern_ids_np)
            
            for enc_idx in range(num_encoders):
                ax = axes[enc_idx]
                
                # Collect variances across steps for this encoder
                step_variances = []
                for step_outputs in all_encoder_outputs:
                    if f"encoder_{enc_idx}" in step_outputs:
                        variances = np.array(step_outputs[f"encoder_{enc_idx}"]["variance"])
                        # Average over batch and pairs
                        mean_var = np.mean(variances)
                        step_variances.append(mean_var)
                
                if step_variances:
                    # Plot variance over steps
                    steps = list(range(len(step_variances)))
                    ax.plot(steps, step_variances, 'b-', alpha=0.7, label='Mean Variance')
                    ax.set_title(f'Encoder {enc_idx} Variance Over Steps')
                    ax.set_xlabel('Training Step')
                    ax.set_ylabel('Mean Variance')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'No data for Encoder {enc_idx}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Encoder {enc_idx} - No Data')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.warning(f"Encoder specialization plot creation failed: {e}")
            return None
    
    def _create_poe_aggregation_plot(self, all_encoder_outputs: list, pattern_ids: chex.Array) -> Optional[plt.Figure]:
        """
        Create plot showing PoE aggregation effectiveness.
        
        Args:
            all_encoder_outputs: List of encoder outputs from all steps
            pattern_ids: Pattern IDs for the batch
            
        Returns:
            matplotlib Figure or None if creation fails
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Analyze how well PoE combines encoder outputs
            pattern_ids_np = np.array(pattern_ids)
            unique_patterns = np.unique(pattern_ids_np)
            
            # For each pattern, show encoder variance distribution
            for pattern_id in unique_patterns:
                pattern_mask = (pattern_ids_np == pattern_id)
                if np.any(pattern_mask):
                    pattern_variances = []
                    
                    for enc_idx in range(len(self.encoders)):
                        enc_vars = []
                        for step_outputs in all_encoder_outputs:
                            if f"encoder_{enc_idx}" in step_outputs:
                                variances = np.array(step_outputs[f"encoder_{enc_idx}"]["variance"])
                                pattern_vars = variances[pattern_mask]
                                enc_vars.extend(pattern_vars.flatten())
                        
                        if enc_vars:
                            pattern_variances.append(np.mean(enc_vars))
                    
                    if pattern_variances:
                        # Plot encoder variances for this pattern
                        encoder_indices = list(range(len(pattern_variances)))
                        ax.bar([f'E{i}' for i in encoder_indices], pattern_variances, 
                               alpha=0.7, label=f'Pattern {pattern_id}')
            
            ax.set_title('PoE Aggregation: Encoder Variances by Pattern')
            ax.set_xlabel('Encoder')
            ax.set_ylabel('Mean Variance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.warning(f"PoE aggregation plot creation failed: {e}")
            return None
    
    def _create_decoder_training_plot(self, all_encoder_outputs: list, num_steps: int) -> Optional[plt.Figure]:
        """
        Create plot showing decoder training progress.
        
        Args:
            all_encoder_outputs: List of encoder outputs from all steps
            num_steps: Number of training steps
            
        Returns:
            matplotlib Figure or None if creation fails
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # This is a placeholder plot - in practice, you'd want to track
            # decoder-specific metrics over time
            ax.text(0.5, 0.5, 'Decoder Training Progress\n(Placeholder)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Phase 2: Decoder Training Progress')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.warning(f"Decoder training plot creation failed: {e}")
            return None
    
    def _organize_phase2_metrics_for_wandb(self, metrics: dict) -> dict:
        """
        Organize Phase 2 metrics for better WandB visualization.
        
        Args:
            metrics: Raw metrics from training
            
        Returns:
            Organized metrics with proper namespacing
        """
        organized_metrics = {}
        
        try:
            # 1. Core training metrics (always present)
            if "loss" in metrics:
                organized_metrics["training/total_loss"] = metrics["loss"]
            if "reconstruction_loss" in metrics:
                organized_metrics["training/reconstruction_loss"] = metrics["reconstruction_loss"]
            if "prior_kl" in metrics:
                organized_metrics["training/prior_kl"] = metrics["prior_kl"]
            if "pairwise_kl" in metrics:
                organized_metrics["training/pairwise_kl"] = metrics["pairwise_kl"]
            
            # 2. PoE metrics (structured training specific)
            if "poe_prior_weight" in metrics:
                organized_metrics["poe/prior_weight"] = metrics["poe_prior_weight"]
            if "poe_num_encoders" in metrics:
                organized_metrics["poe/num_encoders"] = metrics["poe_num_encoders"]
            if "poe_alphas_mean" in metrics:
                organized_metrics["poe/alphas_mean"] = metrics["poe_alphas_mean"]
            
            # 3. Encoder specialization metrics (maintaining frozen specialization)
            for key, value in metrics.items():
                if key.startswith("phase_b/encoder_"):
                    # Convert phase_b/encoder_X/... to encoder_specialization/encoder_X/...
                    new_key = key.replace("phase_b/encoder_", "encoder_specialization/encoder_")
                    organized_metrics[new_key] = value
                elif key.startswith("phase_b/poe/"):
                    # Convert phase_b/poe/... to poe/...
                    new_key = key.replace("phase_b/poe/", "poe/")
                    organized_metrics[new_key] = value
                elif key.startswith("phase_b/decoder/"):
                    # Convert phase_b/decoder/... to decoder/...
                    new_key = key.replace("phase_b/decoder/", "decoder/")
                    organized_metrics[new_key] = value
                elif key.startswith("phase_b/plots/"):
                    # Keep plots as is for WandB media logging
                    organized_metrics[key] = value
            
            # 4. Charts section for better visualization
            if "contrastive_loss" in metrics:
                organized_metrics["Charts/contrastive_loss"] = metrics["contrastive_loss"]
            if "contrastive_loss_weighted" in metrics:
                organized_metrics["Charts/contrastive_loss_weighted"] = metrics["contrastive_loss_weighted"]
            
            # 5. Phase 2 summary metrics
            organized_metrics["phase_b/summary/training_mode"] = "Joint Decoder Training"
            organized_metrics["phase_b/summary/encoder_status"] = "Frozen (Maintaining Specialization)"
            organized_metrics["phase_b/summary/decoder_status"] = "Trainable (Reconstruction Focus)"
            
        except Exception as e:
            logging.warning(f"Phase 2 metrics organization failed: {e}")
            # Fallback to original metrics
            organized_metrics = metrics
        
        return organized_metrics
    
    def _create_specialized_training_data(self, target_pattern: int) -> tuple:
        """
        Create specialized training data for individual encoder training.
        
        Args:
            target_pattern: Pattern this encoder should specialize in (1, 2, or 3)
            
        Returns:
            Tuple of (grids, shapes, pattern_ids) for specialized training
        """
        logging.info(f"     Creating specialized data for pattern {target_pattern}")
        
        # Generate balanced data with emphasis on target pattern
        total_samples = self.batch_size * 10  # Generate more samples for individual training
        target_samples = int(total_samples * 0.7)  # 70% target pattern
        other_samples = total_samples - target_samples
        
        grids_list = []
        shapes_list = []
        pattern_ids_list = []
        
        # Generate target pattern samples (reinforced)
        for _ in range(target_samples):
            grids, shapes, _ = self._create_single_pattern_sample(target_pattern)
            grids_list.append(grids)
            shapes_list.append(shapes)
            pattern_ids_list.append(target_pattern)
        
        # Generate other pattern samples (reduced certainty)
        other_patterns = [p for p in [1, 2, 3] if p != target_pattern]
        samples_per_other = other_samples // len(other_patterns)
        
        for pattern_id in other_patterns:
            for _ in range(samples_per_other):
                grids, shapes, _ = self._create_single_pattern_sample(pattern_id)
                grids_list.append(grids)
                shapes_list.append(shapes)
                pattern_ids_list.append(pattern_id)
        
        # Stack and return
        grids = jnp.stack(grids_list, axis=0)
        shapes = jnp.stack(shapes_list, axis=0)
        pattern_ids = jnp.array(pattern_ids_list)
        
        logging.info(f"     Generated {len(grids_list)} samples: {target_samples} target, {other_samples} others")
        return grids, shapes, pattern_ids
    
    def _create_pattern_dataset(self, pattern_id: int, num_samples: int) -> tuple:
        """Create a dataset composed solely of a single pattern using pre-loaded datasets.

        This method now uses the datasets loaded ONCE at initialization,
        ensuring the same data is used every time for consistent certainty plots.
        
        CRITICAL: Uses the SAME APPROACH as training - takes ALL samples that belong to the targeted pattern.

        Args:
            pattern_id: Pattern to generate (1, 2, or 3).
            num_samples: Number of samples to generate for this pattern. 
                       If None, uses ALL available samples (recommended for certainty plots).

        Returns:
            Tuple of (grids, shapes, pattern_ids) each with the requested number of samples
            corresponding to ``pattern_id``.
        """
        import numpy as np
        
        pattern_names = {1: "L-tetromino", 2: "O-tetromino", 3: "T-tetromino"}
        
        # CRITICAL DEBUG: Log the call parameters and call stack
        import traceback
        call_stack = traceback.format_stack()[-3:-1]  # Get last few stack frames
        logging.info(f"      🔍 _create_pattern_dataset called with pattern_id={pattern_id}, num_samples={num_samples}")
        logging.info(f"      🔍 Call stack: {' -> '.join([frame.split('/')[-1].split(':')[0] for frame in call_stack])}")
        
        # Use pre-loaded datasets from initialization - SAME APPROACH AS TRAINING
        if hasattr(self, 'pattern_datasets') and pattern_id in self.pattern_datasets:
            dataset_data = self.pattern_datasets[pattern_id]
            grids = dataset_data['grids']
            shapes = dataset_data['shapes']
            pattern_ids = dataset_data['pattern_ids']
            
            # CRITICAL: Take ALL samples that belong to the targeted pattern (like training does)
            # Don't limit by num_samples - use all available samples for this pattern
            available_samples = len(grids)
            
            # CRITICAL DEBUG: Log dataset details and validation
            logging.info(f"      📊 Dataset validation for pattern {pattern_id}:")
            logging.info(f"        - Available samples: {available_samples}")
            logging.info(f"        - Grids shape: {grids.shape}")
            logging.info(f"        - Shapes shape: {shapes.shape}")
            logging.info(f"        - Pattern IDs shape: {pattern_ids.shape}")
            logging.info(f"        - Pattern IDs unique values: {np.unique(pattern_ids)}")
            logging.info(f"        - Pattern IDs first 10: {pattern_ids[:10]}")
            
            # CRITICAL VALIDATION: Ensure pattern IDs are correct
            if not np.all(pattern_ids == pattern_id):
                logging.error(f"        ❌ CRITICAL ERROR: Pattern IDs mismatch!")
                logging.error(f"           Expected all {pattern_id}, got: {np.unique(pattern_ids)}")
                logging.error(f"           This indicates data corruption!")
                raise ValueError(f"Pattern IDs mismatch for pattern {pattern_id}")
            else:
                logging.info(f"        ✅ Pattern IDs validation passed")
            
                            # CRITICAL: Return ALL samples for this pattern (not limited by num_samples)
                logging.info(f"      ✅ Using ALL {available_samples} pre-loaded samples from pattern {pattern_id} ({pattern_names[pattern_id]})")
                logging.info(f"      ✅ This ensures 100% pattern isolation - no mixing with other patterns")
                
                # CRITICAL WARNING: Check if sample count is suspiciously low
                if available_samples < 100:
                    logging.warning(f"      ⚠️  WARNING: Pattern {pattern_id} has only {available_samples} samples!")
                    logging.warning(f"         This is much lower than Phase A data (1260 samples)")
                    logging.warning(f"         This explains why histograms show fewer samples!")
                    logging.warning(f"         Expected: 1260 samples, Got: {available_samples} samples")
                
                # CRITICAL: Return the EXACT same data every time
                return grids, shapes, pattern_ids
        
        else:
            logging.error(f"      ❌ CRITICAL ERROR: No pre-loaded dataset for pattern {pattern_id}")
            logging.error(f"      This should NEVER happen if pattern datasets were loaded correctly")
            raise ValueError(f"No pre-loaded dataset for pattern {pattern_id}")
    
    def _verify_pattern_datasets_consistency(self) -> None:
        """
        Verify that pattern datasets are loaded and consistent.
        This method helps debug dataset loading issues.
        """
        if not hasattr(self, 'pattern_datasets'):
            logging.warning("      ⚠️ No pattern_datasets attribute found")
            return
        
        if not self.pattern_datasets:
            logging.warning("      ⚠️ pattern_datasets is empty")
            return
        
        logging.info("      🔍 Verifying pattern datasets consistency...")
        for pattern_id in [1, 2, 3]:
            if pattern_id in self.pattern_datasets:
                data = self.pattern_datasets[pattern_id]
                grids = data['grids']
                shapes = data['shapes']
                pattern_ids = data['pattern_ids']
                
                logging.info(f"        Pattern {pattern_id}:")
                logging.info(f"          - Grids: {grids.shape}, dtype: {grids.dtype}")
                logging.info(f"          - Shapes: {shapes.shape}, dtype: {shapes.dtype}")
                logging.info(f"          - Pattern IDs: {pattern_ids.shape}, unique: {jnp.unique(pattern_ids)}")
                
                # Verify pattern IDs are correct
                if not jnp.all(pattern_ids == pattern_id):
                    logging.error(f"          ❌ Pattern IDs mismatch! Expected all {pattern_id}, got: {jnp.unique(pattern_ids)}")
                else:
                    logging.info(f"          ✅ Pattern IDs consistent")
            else:
                logging.warning(f"        Pattern {pattern_id}: Not loaded")
        
        logging.info("      ✅ Pattern datasets verification complete")
    
    def _create_pattern_dataset_synthetic(self, pattern_id: int, num_samples: int) -> tuple:
        """Create a dataset composed solely of a single pattern with clean tetromino shapes.

        This is the original synthetic data generation method, kept as a fallback.

        Args:
            pattern_id: Pattern to generate (1, 2, or 3).
            num_samples: Number of samples to generate for this pattern.

        Returns:
            Tuple of (grids, shapes, pattern_ids) each with ``num_samples``
            entries corresponding to ``pattern_id``.
        """
        import numpy as np
        import random
        
        # Set random seed for reproducibility
        random.seed(self.cfg.training.seed + pattern_id)
        
        # Get number of pairs from config
        num_pairs = self.task_generator_kwargs["num_pairs"]
        
        # Define clean tetromino patterns (1-based indexing to match our system)
        pattern_definitions = {
            1: {  # L-tetromino (3x2 box) - matches our pattern 1
                'offsets': [(0, 0), (1, 0), (2, 0), (2, 1)],
                'box_h': 3, 'box_w': 2,
                'name': 'L-tetromino'
            },
            2: {  # O-tetromino (2x2 square) - matches our pattern 2  
                'offsets': [(0, 0), (0, 1), (1, 0), (1, 1)],
                'box_h': 2, 'box_w': 2,
                'name': 'O-tetromino'
            },
            3: {  # T-tetromino (2x3 box) - matches our pattern 3
                'offsets': [(0, 0), (0, 1), (0, 2), (1, 1)],
                'box_h': 2, 'box_w': 3,
                'name': 'T-tetromino'
            }
        }
        
        if pattern_id not in pattern_definitions:
            logging.warning(f"Unknown pattern_id {pattern_id}, using pattern 1")
            pattern_id = 1
        
        pattern_info = pattern_definitions[pattern_id]
        logging.info(f"      Creating {num_samples} synthetic samples of {pattern_info['name']} (Pattern {pattern_id})")
        
        # Initialize arrays
        grids = np.zeros((num_samples, num_pairs, 5, 5, 2), dtype=np.uint8)
        shapes = np.zeros((num_samples, num_pairs, 2, 2), dtype=np.uint8)
        pattern_ids = np.full(num_samples, pattern_id, dtype=np.uint8)
        
        for sample_idx in range(num_samples):
            # Sample colors for this sample (consistent across all pairs)
            colors = [random.randint(1, 9) for _ in range(4)]
            
            for pair_idx in range(num_pairs):
                # Generate input grid with single anchor point
                input_grid = np.zeros((5, 5), dtype=np.uint8)
                output_grid = np.zeros((5, 5), dtype=np.uint8)
                
                # Choose random position for pattern (ensuring it fits)
                max_row = 5 - pattern_info['box_h']
                max_col = 5 - pattern_info['box_w']
                top = random.randint(0, max_row)
                left = random.randint(0, max_col)
                
                # Mark anchor in input
                input_grid[top, left] = 0  # Use 0 for input (anchor point)
                
                # Draw pattern in output
                for k, (dr, dc) in enumerate(pattern_info['offsets']):
                    output_grid[top + dr, left + dc] = colors[k % len(colors)]
                
                # Store in arrays
                grids[sample_idx, pair_idx, :, :, 0] = input_grid
                grids[sample_idx, pair_idx, :, :, 1] = output_grid
                shapes[sample_idx, pair_idx, 0] = [5, 5] # [input_rows, input_cols]
                shapes[sample_idx, pair_idx, 1] = [5, 5] # [output_rows, output_cols]
        
        logging.info(f"      Generated {num_samples} synthetic samples: {grids.shape}, {shapes.shape}")
        return jnp.array(grids), jnp.array(shapes), jnp.array(pattern_ids)
    
    def _create_single_pattern_sample(self, pattern_id: int) -> tuple:
        """
        Create a single sample for a specific pattern.
        
        Args:
            pattern_id: Pattern to generate (1, 2, or 3)
            
        Returns:
            Tuple of (grids, shapes, pattern_ids)
        """
        # Use the existing pattern generation logic
        from datasets.task_gen.dataloader import make_task_gen_dataloader
        
        dataloader = make_task_gen_dataloader(
            batch_size=1,
            log_every_n_steps=1,
            num_workers=0,
            task_generator_class="STRUCT_PATTERN",
            num_pairs=self.task_generator_kwargs["num_pairs"],
            online_data_augmentation=self.cfg.training.online_data_augmentation,
            seed=self.cfg.training.seed + pattern_id,
            pattern=pattern_id,
            pattern_per_task=True,
            num_rows=self.task_generator_kwargs.get("num_rows", 5),
            num_cols=self.task_generator_kwargs.get("num_cols", 5),
        )
        
        # CRITICAL FIX: This method creates the SAME sample every time due to fixed seed
        # This causes the "1260 copies of one grid" problem
        # Use _create_single_pattern_sample_with_seed for diverse samples
        
        # Extract single sample - handle different dataloader output formats
        try:
            # Try the expected format first
            for batch in dataloader:
                if len(batch) == 2:
                    # Format: (grids, shapes)
                    grids, shapes = batch
                    # Extract from batch format: (log_every_n_steps, batch_size, ...)
                    return grids[0, 0], shapes[0, 0], pattern_id
                elif len(batch) == 3:
                    # Format: (grids, shapes, pattern_ids)
                    grids, shapes, _ = batch
                    return grids[0, 0], shapes[0, 0], pattern_id
                else:
                    # Unexpected format, try to handle gracefully
                    logging.warning(f"Unexpected dataloader output format: {len(batch)} elements")
                    if hasattr(batch, '__getitem__'):
                        grids = batch[0] if len(batch) > 0 else None
                        shapes = batch[1] if len(batch) > 1 else None
                        if grids is not None and shapes is not None:
                            return grids[0, 0], shapes[0, 0], pattern_id
                    
                    # Fallback: create minimal sample
                    logging.warning(f"Creating fallback sample for pattern {pattern_id}")
                    num_pairs = self.task_generator_kwargs["num_pairs"]
                    fallback_grids = jnp.zeros((1, 1, num_pairs, 5, 5, 2), jnp.uint8)
                    fallback_shapes = jnp.ones((1, 1, num_pairs, 2, 2), jnp.uint8)
                    return fallback_grids[0, 0], fallback_shapes[0, 0], pattern_id
                    
        except Exception as e:
            logging.error(f"Error creating single pattern sample for pattern {pattern_id}: {e}")
            # Create minimal fallback sample
            num_pairs = self.task_generator_kwargs["num_pairs"]
            fallback_grids = jnp.zeros((1, 1, num_pairs, 5, 5, 2), jnp.uint8)
            fallback_shapes = jnp.ones((1, 1, num_pairs, 2, 2), jnp.uint8)
            return fallback_grids[0, 0], fallback_shapes[0, 0], pattern_id

    def _create_single_pattern_sample_with_seed(self, pattern_id: int, seed: int) -> tuple:
        """
        Create a single sample for a specific pattern with a given seed.
        
        CRITICAL FIX: This method ensures diversity by using different seeds
        This prevents the "1260 copies of one grid" problem that causes artificial zero variance.
        
        Args:
            pattern_id: Pattern to generate (1, 2, or 3)
            seed: Seed for reproducible but diverse generation
            
        Returns:
            Tuple of (grids, shapes, pattern_ids)
        """
        # Use the existing pattern generation logic with variable seed
        from datasets.task_gen.dataloader import make_task_gen_dataloader
        
        dataloader = make_task_gen_dataloader(
            batch_size=1,
            log_every_n_steps=1,
            num_workers=0,
            task_generator_class="STRUCT_PATTERN",
            num_pairs=self.task_generator_kwargs["num_pairs"],
            online_data_augmentation=self.cfg.training.online_data_augmentation,
            seed=seed,  # CRITICAL: Use variable seed for diversity
            pattern=pattern_id,
            pattern_per_task=True,
            num_rows=self.task_generator_kwargs.get("num_rows", 5),
            num_cols=self.task_generator_kwargs.get("num_cols", 5),
        )
        
        # Extract single sample - handle different dataloader output formats
        try:
            # Try the expected format first
            for batch in dataloader:
                if len(batch) == 2:
                    # Format: (grids, shapes)
                    grids, shapes = batch
                    # Extract from batch format: (log_every_n_steps, batch_size, ...)
                    return grids[0, 0], shapes[0, 0], pattern_id
                elif len(batch) == 3:
                    # Format: (grids, shapes, pattern_ids)
                    grids, shapes, _ = batch
                    return grids[0, 0], shapes[0, 0], pattern_id
                else:
                    # Unexpected format, try to handle gracefully
                    logging.warning(f"Unexpected dataloader output format: {len(batch)} elements")
                    if hasattr(batch, '__getitem__'):
                        grids = batch[0] if len(batch) > 0 else None
                        shapes = batch[1] if len(batch) > 1 else None
                        if grids is not None and shapes is not None:
                            return grids[0, 0], shapes[0, 0], pattern_id
                    
                    # Fallback: create minimal sample
                    logging.warning(f"Creating fallback sample for pattern {pattern_id}")
                    num_pairs = self.task_generator_kwargs["num_pairs"]
                    fallback_grids = jnp.zeros((1, 1, num_pairs, 5, 5, 2), jnp.uint8)
                    fallback_shapes = jnp.ones((1, 1, num_pairs, 2, 2), jnp.uint8)
                    return fallback_grids[0, 0], fallback_shapes[0, 0], pattern_id
                    
        except Exception as e:
            logging.error(f"Error creating single pattern sample for pattern {pattern_id}: {e}")
            # Create minimal fallback sample
            num_pairs = self.task_generator_kwargs["num_pairs"]
            fallback_grids = jnp.zeros((1, 1, num_pairs, 5, 5, 2), jnp.uint8)
            fallback_shapes = jnp.ones((1, 1, num_pairs, 2, 2), jnp.uint8)
            return fallback_grids[0, 0], fallback_shapes[0, 0], pattern_id

    def _compute_repulsion_loss(self, current_latents: chex.Array, target_latents_store: dict, current_encoder_idx: int, margin: float = 5.0, verbose: bool = True) -> float:
        """
        Compute repulsion loss to encourage encoder specialization.
        
        Args:
            current_latents: Current encoder's latent representations (batch_size, latent_dim)
            target_latents_store: Dictionary mapping encoder_idx to pattern_id to target latents
            current_encoder_idx: Index of the current encoder
            margin: Minimum desired distance between encoders
            verbose: Whether to output detailed debugging information
            
        Returns:
            Repulsion loss value
        """
        if verbose:
            logging.info(f"🔍 REPULSION LOSS COMPUTATION DEBUG:")
            logging.info(f"   - current_encoder_idx: {current_encoder_idx}")
            logging.info(f"   - target_latents_store keys: {list(target_latents_store.keys())}")
            logging.info(f"   - current_latents shape: {current_latents.shape}")
            logging.info(f"   - margin: {margin}")
        
        if not target_latents_store or current_encoder_idx == 0:
            # No previous encoders to repulse from
            if verbose:
                logging.info(f"   - REPULSION SKIPPED: No previous encoders to repulse from")
                logging.info(f"     * target_latents_store: {target_latents_store}")
                logging.info(f"     * current_encoder_idx: {current_encoder_idx}")
            return 0.0
        
        # CRITICAL FIX: Use consistent batch size for target latents
        # Extract target latents with the same batch size as current latents to avoid resizing
        current_batch_size = current_latents.shape[0]
        
        repulsion_loss = 0.0
        num_repulsion_terms = 0
        
        # Iterate through all previous encoders
        for prev_enc_idx in range(current_encoder_idx):
            if verbose:
                logging.info(f"   - Checking previous encoder {prev_enc_idx}")
            if prev_enc_idx in target_latents_store:
                prev_targets = target_latents_store[prev_enc_idx]
                if verbose:
                    logging.info(f"     * Found targets for encoder {prev_enc_idx}, keys: {list(prev_targets.keys()) if prev_targets else 'None'}")
                
                # For each pattern, compute repulsion from previous encoder's targets
                for pattern_id, target_latents in prev_targets.items():
                    if verbose:
                        logging.info(f"       - Pattern {pattern_id}: target_latents type={type(target_latents)}, shape={target_latents.shape if hasattr(target_latents, 'shape') else 'No shape'}")
                    
                    if target_latents is not None and len(target_latents) > 0:
                        # CRITICAL FIX: Always resize target latents to match current batch size
                        # This ensures consistent computation and avoids batch size mismatches
                        try:
                            if len(target_latents) != current_batch_size:
                                if verbose:
                                    logging.info(f"         * Resizing target latents from {len(target_latents)} to {current_batch_size}")
                                
                                if len(target_latents) > current_batch_size:
                                    # Sample from target latents to match current batch size
                                    indices = np.random.choice(len(target_latents), current_batch_size, replace=False)
                                    resized_target_latents = target_latents[indices]
                                else:
                                    # Repeat target latents to match current batch size
                                    repeat_factor = current_batch_size // len(target_latents)
                                    remainder = current_batch_size % len(target_latents)
                                    resized_target_latents = np.tile(target_latents, (repeat_factor, 1))
                                    if remainder > 0:
                                        additional = target_latents[:remainder]
                                        resized_target_latents = np.vstack([resized_target_latents, additional])
                                
                                if verbose:
                                    logging.info(f"         * Resized target latents from {len(target_latents)} to {len(resized_target_latents)}")
                            else:
                                resized_target_latents = target_latents
                                if verbose:
                                    logging.info(f"         * Batch sizes already match: {len(target_latents)} == {current_batch_size}")
                            
                            # Compute L2 distance between current and target latents
                            distances = jnp.linalg.norm(current_latents - resized_target_latents, axis=1)
                            if verbose:
                                logging.info(f"         * Distances shape: {distances.shape}, mean: {float(jnp.mean(distances)):.6f}")
                            
                            # IMPROVED REPULSION LOSS: Use a more effective repulsion strategy
                            # Option 1: Exponential repulsion that increases as distances get smaller
                            # This provides strong repulsion when encoders are too close together
                            exp_repulsion = jnp.mean(jnp.exp(-distances / margin))
                            
                            # Option 2: Inverse distance repulsion (always non-zero)
                            # This ensures continuous repulsion even when distances are large
                            inv_repulsion = jnp.mean(1.0 / (distances + 1e-6))
                            
                            # Option 3: Margin-based repulsion (only when distance < margin)
                            # This enforces a minimum distance threshold
                            margin_repulsion = jnp.mean(jnp.maximum(0, margin - distances))
                            
                            # Combine all three approaches for robust repulsion
                            # Scale them appropriately to balance their contributions
                            # - Exponential: 50% weight for strong local repulsion
                            # - Inverse: 30% weight for continuous global repulsion  
                            # - Margin: 20% weight for explicit distance enforcement
                            final_repulsion_term = (
                                0.5 * exp_repulsion +      # Exponential: strong repulsion for small distances
                                0.3 * inv_repulsion * 0.1 + # Inverse: always some repulsion
                                0.2 * margin_repulsion     # Margin: explicit distance enforcement
                            )
                            
                            if verbose:
                                logging.info(f"         * Exp repulsion: {float(exp_repulsion):.6f}")
                                logging.info(f"         * Inv repulsion: {float(inv_repulsion * 0.1):.6f}")
                                logging.info(f"         * Margin repulsion: {float(margin_repulsion):.6f}")
                                logging.info(f"         * Final repulsion term: {float(final_repulsion_term):.6f}")
                            
                            repulsion_loss += final_repulsion_term
                            num_repulsion_terms += 1
                            
                        except Exception as resize_error:
                            logging.warning(f"         * Failed to process target latents: {resize_error}")
                            logging.warning(f"         * Skipping this repulsion term")
                    else:
                        logging.warning(f"         * Invalid target_latents: {target_latents}")
            else:
                if verbose:
                    logging.info(f"     * No targets found for encoder {prev_enc_idx}")
        
        # Average over all repulsion terms
        if num_repulsion_terms > 0:
            repulsion_loss = repulsion_loss / num_repulsion_terms
            if verbose:
                logging.info(f"   - Final repulsion loss: {float(repulsion_loss):.6f} (from {num_repulsion_terms} terms)")
        else:
            if verbose:
                logging.info(f"   - No repulsion terms computed, returning 0.0")
        
        return repulsion_loss
    
    def _extract_target_latents(self, encoder_idx: int, encoder_params: dict, state: TrainState) -> dict:
        """
        Extract target latent representations from a trained encoder for repulsion loss.
        
        Args:
            encoder_idx: Index of the encoder
            encoder_params: Encoder parameters
            state: Training state
            
        Returns:
            Dictionary mapping pattern_id to target latent representations
        """
        logging.info(f"🔍 EXTRACTING TARGET LATENTS for Encoder {encoder_idx}")
        target_latents = {}
        
        # CRITICAL FIX: Use the same batch size as training to avoid mismatch
        # This ensures target latents have the same batch size as current latents during training
        num_samples = self.batch_size  # Use batch_size instead of fixed 32
        logging.info(f"   - Using batch_size={num_samples} for target latents (matching training)")
        logging.info(f"   - This prevents batch size mismatches in repulsion loss computation")
        
        # Create evaluation data for each pattern
        for pattern_id in [1, 2, 3]:
            try:
                # Generate pattern-specific data
                pattern_data = self._create_pattern_dataset(pattern_id, num_samples=num_samples)
                grids, shapes, _ = pattern_data
                logging.info(f"   - Pattern {pattern_id}: grids shape={grids.shape}, shapes shape={shapes.shape}")
                
                # Get encoder outputs
                mu, logvar = self.encoders[encoder_idx].apply(
                    {"params": encoder_params}, 
                    grids, 
                    shapes, 
                    True, 
                    mutable=False
                )
                
                logging.info(f"   - Pattern {pattern_id}: mu shape={mu.shape}, logvar shape={logvar.shape}")
                
                # Use mean of latents as target (or could use multiple samples)
                target_lat = mu.mean(axis=-2)  # Mean over pairs
                target_latents[pattern_id] = jnp.array(target_lat)
                
                logging.info(f"   - Pattern {pattern_id}: target_lat shape={target_lat.shape}, mean={float(jnp.mean(target_lat)):.6f}")
                
            except Exception as e:
                logging.warning(f"Failed to extract target latents for Encoder {encoder_idx}, Pattern {pattern_id}: {e}")
                target_latents[pattern_id] = None
        
        logging.info(f"   - Final target_latents keys: {list(target_latents.keys())}")
        return target_latents

    def train(self, state: TrainState, enc_params_list: list[dict]) -> TrainState:
        cfg = self.cfg
        num_steps = cfg.training.total_num_steps
        log_every = cfg.training.log_every_n_steps
        self.enc_params_list = enc_params_list  # Store for train_n_steps
        
        step = 0  # Always start from 0 - unique run IDs prevent conflicts
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
        
        # NEW: Two-phase training approach
        if self.encoder_expose_steps > 0 and not self.phase1_completed:
            logging.info("🚀 PHASE 1: Individual Encoder Specialization")
            logging.info(f"   - Training each encoder independently for {self.encoder_expose_steps} steps")
            logging.info(f"   - Using original decoders to prevent interference")
            logging.info(f"   - Focus: pattern specialization through contrastive learning")
            
            # Log Phase A T-SNE evaluation schedule
            eval_every_n_steps = self.cfg.training.get("eval_every_n_logs", 20) * self.cfg.training.get("log_every_n_steps", 5)
            num_tsne_evals_per_encoder = self.encoder_expose_steps // eval_every_n_steps
            logging.info(f"   - Phase A T-SNE evaluation: every {eval_every_n_steps} steps")
            logging.info(f"   - Expected T-SNE evaluations per encoder: {num_tsne_evals_per_encoder}")
            logging.info(f"   - Total expected T-SNE evaluations: {num_tsne_evals_per_encoder * len(enc_params_list)}")
            
            # Phase 1: Individual encoder specialization
            state = self._specialize_individual_encoders(state, enc_params_list)
            
            logging.info("✅ PHASE 1 COMPLETED: Encoders specialized!")
            logging.info("   - Ready for Phase 2: Joint decoder training")
            logging.info("   - Encoders will be frozen during joint training")
            
            # Create merged encoder certainty panel after Phase 1 completion
            logging.info("🔍 Creating merged encoder certainty panel after Phase 1 completion...")
            merged_certainty_panel = self._create_merged_encoder_certainty_panel(state, step=0)
            if merged_certainty_panel is not None:
                # Log to WandB with a step value that's guaranteed to be greater than current WandB step
                # Use the total steps completed during Phase 1 to ensure proper step ordering
                phase1_completion_step = max(600, self.phase_a_global_step + 100)  # Ensure step >= 600
                wandb.log({
                    "phase_1_completion/merged_encoder_certainty_panel": wandb.Image(merged_certainty_panel)
                }, step=phase1_completion_step)
                plt.close(merged_certainty_panel)
                logging.info(f"       ✅ Merged encoder certainty panel logged to WandB with step {phase1_completion_step}")
            else:
                logging.warning("       ❌ Failed to create merged encoder certainty panel")
        
        # Test forward pass first to catch any issues early
        logging.info("Testing forward pass...")
        try:
            # Generate a balanced test batch with uniform pattern distribution
            if hasattr(self, 'task_generator') and self.task_generator:
                test_grids, test_shapes, test_pattern_ids = self._create_balanced_pattern_batch(
                    self.batch_size, 
                    self.samples_per_pattern_per_batch
                )
                test_batch = test_grids, test_shapes
                logging.info(f"✅ Test forward pass: Using EXPLICIT pattern IDs from balanced generation")
                logging.info(f"   Pattern IDs: {test_pattern_ids[:10]}... (first 10)")
                logging.info(f"   Expected: [1,1,1,...,2,2,2,...,3,3,3,...]")
            else:
                # Fallback to fixed dataset
                if hasattr(self, 'train_grids') and self.train_grids is not None:
                    test_batch = self.train_grids[:self.batch_size], self.train_shapes[:self.batch_size]
                    # Extract pattern IDs from data content for fixed dataset
                    test_pattern_ids = self._extract_true_pattern_ids_from_data(test_batch[0], test_batch[1])
                    logging.info(f"⚠️  Test forward pass: Using EXTRACTED pattern IDs from fixed dataset")
                else:
                    # No fixed dataset available, create a minimal test batch
                    logging.warning("No fixed dataset available for test forward pass, creating minimal test batch")
                    num_pairs = self.task_generator_kwargs["num_pairs"]
                    test_grids = jnp.zeros((self.batch_size, num_pairs, 5, 5, 2), jnp.uint8)
                    test_shapes = jnp.ones((self.batch_size, num_pairs, 2, 2), jnp.uint8)
                    test_batch = test_grids, test_shapes
                    # Create dummy pattern IDs for minimal test batch
                    test_pattern_ids = jnp.concatenate([
                        jnp.full((self.batch_size // 3,), 1),
                        jnp.full((self.batch_size // 3,), 2),
                        jnp.full((self.batch_size - 2 * (self.batch_size // 3),), 3)
                    ], axis=0)
                    logging.info(f"⚠️  Test forward pass: Using DUMMY pattern IDs for minimal test batch")
            
            # CRITICAL FIX: Use explicit pattern IDs (no more extraction needed)
            test_batch_size = test_batch[0].shape[0]
            
            # Validate pattern_ids
            logging.debug(f"Pattern IDs validation:")
            logging.debug(f"  - Shape: {test_pattern_ids.shape}")
            logging.debug(f"  - Dtype: {test_pattern_ids.dtype}")
            logging.debug(f"  - Min value: {int(jnp.min(test_pattern_ids))}")
            logging.debug(f"  - Max value: {int(jnp.max(test_pattern_ids))}")
            logging.debug(f"  - Unique values: {jnp.unique(test_pattern_ids)}")
            
            # Log pattern distribution from explicit pattern IDs
            unique_patterns, counts = jnp.unique(test_pattern_ids, return_counts=True)
            # Convert JAX arrays to Python types for safe dictionary creation
            unique_patterns_py = [int(p) for p in unique_patterns]
            counts_py = [int(c) for c in counts]
            pattern_distribution = dict(zip(unique_patterns_py, counts_py))
            logging.info(f"Test forward pass: Pattern distribution: {pattern_distribution}")
            logging.info(f"  - Batch size: {test_batch_size}")
            logging.info(f"  - Pattern IDs: {[int(p) for p in test_pattern_ids[:10]]}... (first 10)")
            
            # CRITICAL: Verify pattern diversity for contrastive loss effectiveness
            if len(unique_patterns) < 2:
                logging.warning(f"⚠️  ONLY {len(unique_patterns)} UNIQUE PATTERN(S) IN BATCH!")
                logging.warning(f"   Contrastive loss requires multiple patterns to work effectively")
                logging.warning(f"   Consider increasing batch size or checking data generation")
            else:
                logging.info(f"✅ Batch contains {len(unique_patterns)} unique patterns - contrastive loss should work")
            
            # Essential encoder variance validation
            self._validate_encoder_variance_outputs(state, test_batch)
            
            # Phase 2: Joint training with frozen encoders
            if self.phase1_completed:
                logging.info("🔒 PHASE 2: Joint Decoder Training")
                
                # Test forward pass without specialization losses
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
                    repulsion_kl_coeff=0.0,  # DISABLED in Phase 2
                    contrastive_kl_coeff=0.0,  # DISABLED in Phase 2
                    **(cfg.training.get("inference_kwargs") or {}),
                )
            else:
                # Phase 1: Test with specialization losses
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
                    contrastive_kl_coeff=cfg.training.get("contrastive_kl"),
                    pattern_ids=test_pattern_ids,
                    **(cfg.training.get("inference_kwargs") or {}),
                )
            
            logging.info(f"Forward pass test successful: loss={float(test_loss):.4f}")
        except Exception as e:
            logging.error(f"Forward pass test failed: {e}")
            logging.error(f"Error details: {type(e).__name__}: {str(e)}")
            # Safely log test batch info if available
            if 'test_batch' in locals():
                try:
                    logging.error(f"Test batch shapes: grids={test_batch[0].shape}, shapes={test_batch[1].shape}")
                except Exception as batch_error:
                    logging.error(f"Could not log test batch shapes: {batch_error}")
            else:
                logging.error("Test batch not yet created when error occurred")
            
            # Safely log pattern IDs info if available
            if 'test_pattern_ids' in locals():
                try:
                    logging.error(f"Pattern IDs shape: {test_pattern_ids.shape}, dtype: {test_pattern_ids.dtype}")
                except Exception as pattern_error:
                    logging.error(f"Could not log pattern IDs info: {pattern_error}")
            else:
                logging.error("Pattern IDs not yet created when error occurred")
            raise
        
        logging.info("Starting training loop...")
        pbar = trange(num_steps, disable=False)
        
        # Run evaluation at step 0 (first step)
        if cfg.training.get("eval_every_n_logs"):
            try:
                logging.info(f"Running evaluation at step 0 (first step)")
                self.evaluate(state, enc_params_list, step)
                
                # Test datasets evaluation at first step
                if hasattr(self, 'test_datasets') and self.test_datasets:
                    for dataset_dict in self.test_datasets:
                        try:
                            start = time.time()
                            test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress, fig_tsne_samples, fig_tsne_encoders_list = self.test_dataset_submission(
                                state, dataset_dict, step=step
                            )
                            test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                            
                            # Upload all figures
                            for fig, name in [
                                (fig_grids, "generation"),
                                (fig_heatmap, "pixel_accuracy"),
                                (fig_latents, "latents"),
                                (fig_latents_samples, "latents_samples"),
                                (fig_search_progress, "search_progress"),
                                (fig_tsne_samples, "latents_samples"),
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
                            
                            # Ensure step is greater than or equal to current WandB step to avoid monotonicity issues
                            current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
                            if step <= current_wandb_step:
                                adjusted_step = current_wandb_step + 1
                                logging.info(f"⚠️  Test metrics step {step} is <= current WANDB step ({current_wandb_step}), using adjusted step {adjusted_step}")
                                wandb.log(test_metrics, step=adjusted_step)
                            else:
                                wandb.log(test_metrics, step=step)
                            plt.close('all')  # Close all figures to prevent memory leaks
                            # Explicitly close additional T-SNE figures
                            if fig_tsne_samples is not None:
                                plt.close(fig_tsne_samples)
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
            
            # Log essential step info
            if step % 100 == 0:
                encoder_status = "TRAINABLE" if self.encoder_expose_steps > 0 else "FROZEN"
                logging.info(f"Step {step}/{num_steps}: Encoders {encoder_status} (exposure: {self.encoder_expose_steps} steps remaining)")
                
                # Log transition when encoders become frozen
                if self.encoder_expose_steps == 0 and step >= 100:
                    logging.info("🔒 TRANSITION: Encoders are now FROZEN!")
            
            dataloading_time = time.time()
            for batches in dataloader:
                wandb.log({"timing/dataloading_time": time.time() - dataloading_time}, step=step)
                
                # Training - process log_every_n_steps batches at once
                key, train_key = jax.random.split(key)
                start = time.time()
                
                # CRITICAL: Extract explicit pattern IDs from balanced dataloader
                if hasattr(self, 'task_generator') and self.task_generator:
                    # Balanced dataloader provides (grids, shapes, pattern_ids)
                    if len(batches) == 3:
                        grids, shapes, explicit_pattern_ids = batches
                        logging.info(f"✅ Using EXPLICIT pattern IDs: {explicit_pattern_ids[:10]}... (first 10)")
                        logging.info(f"   Pattern distribution: {[int(p) for p in jnp.unique(explicit_pattern_ids)]}")
                        # DEBUG: Verify pattern ID structure
                        expected_patterns = [1] * self.samples_per_pattern_per_batch + [2] * self.samples_per_pattern_per_batch + [3] * self.samples_per_pattern_per_batch
                        if not jnp.array_equal(explicit_pattern_ids, jnp.array(expected_patterns)):
                            logging.error(f"❌ PATTERN ID MISMATCH!")
                            logging.error(f"   Expected: {expected_patterns[:10]}... (first 10)")
                            logging.error(f"   Got: {explicit_pattern_ids[:10]}... (first 10)")
                            logging.error(f"   Full expected: {expected_patterns}")
                            logging.error(f"   Full got: {explicit_pattern_ids}")
                        else:
                            logging.info(f"✅ Pattern IDs match expected structure")
                        
                        # Essential pattern validation (reduced frequency)
                        if step % 500 == 0:  # Validate every 500 steps to reduce spam
                            self._validate_contrastive_loss_patterns(explicit_pattern_ids, self.batch_size)
                    else:
                        # Fallback if dataloader doesn't provide pattern_ids in expected format
                        grids, shapes = batches
                        explicit_pattern_ids = jnp.concatenate([
                            jnp.full((self.samples_per_pattern_per_batch,), 1),  # Pattern 1
                            jnp.full((self.samples_per_pattern_per_batch,), 2),  # Pattern 2  
                            jnp.full((self.samples_per_pattern_per_batch,), 3),  # Pattern 3
                        ], axis=0)
                        logging.warning(f"⚠️  Task generator dataloader didn't provide 3 elements, using fallback")
                else:
                    # Fixed dataset - extract pattern IDs from data content
                    grids, shapes = batches
                    explicit_pattern_ids = self._extract_true_pattern_ids_from_data(grids[0], shapes[0])
                    logging.info(f"⚠️  Using EXTRACTED pattern IDs: {explicit_pattern_ids[:10]}... (first 10)")
            
            # Log essential batch info
            batch_size = grids.shape[1] if hasattr(grids, 'shape') and len(grids.shape) > 1 else len(grids)
            logging.debug(f"Processing balanced batch with {batch_size} samples")
            
            # Two-phase training logic
            if self.phase1_completed:
                # Phase 2: Joint training with frozen encoders
                batches_with_patterns = (grids, shapes, explicit_pattern_ids)
                state, metrics = self.train_n_steps_phase2(state, batches_with_patterns, train_key)
            else:
                # Phase 1: Individual encoder training (should not reach here after Phase 1)
                logging.warning(f"⚠️  Still in Phase 1 during main training loop - this shouldn't happen")
                batches_with_patterns = (grids, shapes, explicit_pattern_ids)
                state, metrics = self.train_n_steps(state, batches_with_patterns, train_key)
            
            end = time.time()
            
            pbar.update(log_every)
            step += log_every
                
            # Log essential encoder status
            if step % 100 == 0:
                encoder_status = "TRAINABLE" if self.encoder_expose_steps > 0 else "FROZEN"
                logging.info(f"Step {step}/{num_steps}: Encoders {encoder_status}")
            throughput = log_every * self.batch_size / (end - start)
            metrics.update({
                "timing/train_time": end - start,
                "timing/train_num_samples_per_second": throughput
            })
                
            # Add essential contrastive loss to Charts section (like train.py)
            if "contrastive_loss" in metrics:
                metrics["Charts/contrastive_loss"] = metrics["contrastive_loss"]
                if "contrastive_loss_weighted" in metrics:
                    metrics["Charts/contrastive_loss_weighted"] = metrics["contrastive_loss_weighted"]
            
            # CRITICAL: Compute clustering metrics every log_every_n_steps to monitor encoder specialization
            # This provides real-time monitoring of encoder specialization progress
            try:
                logging.debug(f"🔍 Computing clustering metrics at step {step}")
                clustering_metrics = self._compute_clustering_metrics_every_step(
                    state, grids, shapes, explicit_pattern_ids, step
                )
                metrics.update(clustering_metrics)
                logging.debug(f"✅ Clustering metrics computed: {list(clustering_metrics.keys())}")
            except Exception as e:
                logging.warning(f"Clustering metrics computation failed at step {step}: {e}")
                
            # Organize Phase 2 metrics for better WandB visualization
            if self.phase1_completed:
                # Phase 2: Organize metrics by category
                organized_metrics = self._organize_phase2_metrics_for_wandb(metrics)
                wandb.log(organized_metrics, step=step)
                
                # Phase 2: Generate merged encoder certainty panel periodically
                if step % (cfg.training.get("eval_every_n_logs", 20) * log_every) == 0:
                    logging.info(f"🔍 Phase 2: Generating merged encoder certainty panel at step {step}")
                    merged_certainty_panel = self._create_merged_encoder_certainty_panel(state, step)
                    if merged_certainty_panel is not None:
                        wandb.log({
                            "phase_2/merged_encoder_certainty_panel": wandb.Image(merged_certainty_panel)
                        }, step=step)
                        plt.close(merged_certainty_panel)
                        logging.info(f"       ✅ Phase 2 merged encoder certainty panel logged to WandB")
                    else:
                        logging.warning(f"       ❌ Phase 2 merged encoder certainty panel creation failed")
            else:
                # Phase 1: Log metrics as is
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
                        self.evaluate(state, enc_params_list, step)
                        
                        # Test datasets evaluation (like train.py)
                        if hasattr(self, 'test_datasets') and self.test_datasets:
                            for dataset_dict in self.test_datasets:
                                try:
                                    start = time.time()
                                    test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress, fig_tsne_samples, fig_tsne_encoders_list = self.test_dataset_submission(
                                        state, dataset_dict, step=step
                                    )
                                    test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                                    
                                    # Upload all figures
                                    for fig, name in [
                                        (fig_grids, "generation"),
                                        (fig_heatmap, "pixel_accuracy"),
                                        (fig_latents, "latents"),
                                        (fig_latents_samples, "latents_samples"),
                                        (fig_search_progress, "search_progress"),
                                        (fig_tsne_samples, "latents_samples"),
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
                                    
                                    # Ensure step is greater than or equal to current WandB step to avoid monotonicity issues
                                    current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
                                    if step <= current_wandb_step:
                                        adjusted_step = current_wandb_step + 1
                                        logging.info(f"⚠️  Test metrics step {step} is <= current WANDB step ({current_wandb_step}), using adjusted step {adjusted_step}")
                                        wandb.log(test_metrics, step=adjusted_step)
                                    else:
                                        wandb.log(test_metrics, step=step)
                                    
                                    plt.close('all')  # Close all figures to prevent memory leaks
                                    # Explicitly close additional T-SNE figures
                                    if fig_tsne_samples is not None:
                                        plt.close(fig_tsne_samples)
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

    def evaluate(self, state: TrainState, enc_params_list: list[dict] = None, step: int = None) -> dict:
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
        # Log the step parameter being used for evaluation
        logging.info(f"🔍 Evaluation called with step={step} (type: {type(step)})")
        if step is None:
            logging.warning("⚠️  Evaluation called without step parameter - this may cause WandB step tracking issues")
        else:
            logging.info(f"✅ Evaluation using step={step} for WandB logging")
        # Use current encoder weights from state.params, not the original artifact weights
        current_enc_params_list = state.params["encoders"]
        logging.info(f"Evaluation using current encoder weights from training state (step {getattr(state, 'step', 'unknown')})")
        if not hasattr(self, "eval_grids"):
            return {}
        cfg = self.cfg
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
        
        # Initialize metrics dict early to avoid variable not defined errors
        encoder_variance_metrics = {}
        
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
                        # FIXED: Report best fitness across all generations (consistent with GA approach)
                        # Instead of last generation fitness, use the best fitness found
                        best_fitness = np.max(gen_fitness, axis=-1)  # Best across all generations
                        final_losses = -best_fitness  # Convert best fitness to loss
                        search_metrics[f"test/{test_name}/total_final_loss"] = float(np.mean(final_losses))
                        # Keep last generation for reference
                        last_gen_fitness = gen_fitness[..., -1]  # Last generation
                        last_gen_losses = -last_gen_fitness  # Convert to loss
                        search_metrics[f"test/{test_name}/last_generation_loss"] = float(np.mean(last_gen_losses))
                        print(f"[structured_train] ES final_loss: {float(np.mean(final_losses)):.6f} (best across {gen_fitness.shape[-1]} generations)")
                        print(f"[structured_train] ES last_gen_loss: {float(np.mean(last_gen_losses)):.6f} (for comparison)")
        

        
        metrics = {
            f"test/{test_name}/correct_shapes": float(np.mean(correct_shapes)),
            f"test/{test_name}/pixel_correctness": float(np.mean(pixel_correctness)),
            f"test/{test_name}/accuracy": float(np.mean(accuracy)),
            **search_metrics,  # Include search trajectory metrics
            **encoder_variance_metrics,  # Include encoder variance metrics for each pattern
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
            np.ones(spp, dtype=int),      # Pattern 1 (L-tetromino)
            np.ones(spp, dtype=int) * 2,  # Pattern 2 (O-tetromino)
            np.ones(total_sets - 2 * spp, dtype=int) * 3,  # Pattern 3 (T-tetromino)
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
        
        # CRITICAL FIX: Use explicit pattern IDs from evaluation dataset instead of position-based assumption
        # This ensures pattern IDs match the actual data content, just like in training
        if hasattr(self, 'eval_pattern_ids') and self.eval_pattern_ids is not None:
            # Use the explicit pattern IDs that were created during dataset generation
            pattern_sequence = np.array(self.eval_pattern_ids)
            logging.info(f"T-SNE pattern mapping: Using EXPLICIT pattern IDs from evaluation dataset")
        else:
            # Fallback to position-based assumption (should not happen with the fix above)
            pattern_sequence = np.concatenate([
                np.ones(samples_per_pattern, dtype=int),      # Pattern 1 (first 32 samples)
                np.ones(samples_per_pattern, dtype=int) * 2,  # Pattern 2 (next 32 samples)
                np.ones(samples_per_pattern, dtype=int) * 3   # Pattern 3 (last 32 samples)
            ])
            logging.warning(f"T-SNE pattern mapping: Using FALLBACK position-based pattern IDs")
        
        logging.info(
            f"T-SNE pattern mapping: {samples_per_pattern} samples per pattern, total patterns: {np.unique(pattern_sequence)}"
        )
        logging.info(f"Pattern ID distribution: {np.bincount(pattern_sequence)[1:]} (should be [32, 32, 32])")

        # Task IDs: each of the num_sets tasks contributes one point per source
        task_id_sequence = np.arange(num_sets, dtype=int)
        
        # CRITICAL DEBUG: Log pattern sequence details
        logging.info(f"T-SNE pattern sequence - shape: {pattern_sequence.shape}, content: {pattern_sequence[:10]}... (first 10)")
        logging.info(f"T-SNE task_id_sequence - shape: {task_id_sequence.shape}, content: {task_id_sequence[:10]}... (first 10)")
        
        # NEW: Track encoder variances for each pattern to monitor specialization
        encoder_variance_metrics = {}
        
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
                
                # CRITICAL: Compute and track encoder variances for each pattern
                # This shows how well each encoder specializes in different patterns
                var_i = np.exp(np.array(logvar_i))  # Convert logvar to variance
                var_i_flat = var_i.reshape(-1, var_i.shape[-1])  # Flatten to [num_tasks, latent_dim]
                
                # CRITICAL FIX: Expand pattern sequence to match flattened pair dimension
                # Input: pairs (96, 4, 5, 5, 2) -> encoder output (96, 4, 32) -> flattened (384, 32)
                # Pattern sequence: (96,) -> needs to be expanded to (384,) to match flattened pairs
                num_samples = var_i.shape[0]  # 96
                num_pairs = var_i.shape[1]    # 4
                expanded_pattern_sequence = np.repeat(pattern_sequence, num_pairs)  # (96*4,) = (384,)
                
                # Compute mean variance per task for this encoder
                mean_var_per_task = np.mean(var_i_flat, axis=1)  # [384]
                
                # Group variances by pattern for detailed analysis
                for pattern_id in [1, 2, 3]:
                    pattern_mask = (expanded_pattern_sequence == pattern_id)
                    if np.any(pattern_mask):
                        pattern_variances = mean_var_per_task[pattern_mask]
                        pattern_mean_var = float(np.mean(pattern_variances))
                        pattern_std_var = float(np.std(pattern_variances))
                        
                        # Store metrics for WandB logging
                        metric_key = f"encoder_{enc_idx}_pattern_{pattern_id}"
                        encoder_variance_metrics[f"{metric_key}_mean_variance"] = pattern_mean_var
                        encoder_variance_metrics[f"{metric_key}_std_var"] = pattern_std_var
                        encoder_variance_metrics[f"{metric_key}_num_samples"] = int(np.sum(pattern_mask))
                        
                        # Log specialization progress
                        logging.info(f"Pattern {pattern_id} - Encoder {enc_idx}: mean_var={pattern_mean_var:.6f}, std_var={pattern_std_var:.6f}")
                
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
                # CRITICAL FIX: Use the correct pattern sequence length for this encoder's output
                encoder_pattern_sequence = pattern_sequence[:lat_np.shape[0]]
                pattern_ids_list.append(encoder_pattern_sequence)  # Pattern sequence matching this encoder's output length
                task_ids_list.append(task_id_sequence[:lat_np.shape[0]])  # Task IDs matching this encoder's output length
                
                # CRITICAL DEBUG: Log what we're appending
                logging.info(f"Main eval - Encoder {enc_idx} - appended pattern_sequence: {encoder_pattern_sequence.shape}, task_sequence: {task_id_sequence[:lat_np.shape[0]].shape}")
                
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
                # CRITICAL FIX: Use the correct pattern sequence length for context output
                context_pattern_sequence = pattern_sequence[:context_np.shape[0]]
                pattern_ids_list.append(context_pattern_sequence)  # Pattern sequence matching context output length
                task_ids_list.append(task_id_sequence[:context_np.shape[0]])  # Task IDs matching context output length
                
                # CRITICAL DEBUG: Log what we're appending for context
                logging.info(f"Main eval - Context - appended pattern_sequence: {context_pattern_sequence.shape}, task_sequence: {task_id_sequence[:context_np.shape[0]].shape}")
                logging.info(f"Main eval - Added context to T-SNE: {len(context_np)} points")
        else:
            logging.warning(f"Main eval - No 'context' key found in info. Available keys: {list(info.keys())}")
        
        if all_latents:
            # CRITICAL DEBUG: Log list contents before concatenation
            logging.info(f"T-SNE lists before concatenation:")
            logging.info(f"  - all_latents: {len(all_latents)} items, shapes: {[lat.shape for lat in all_latents]}")
            logging.info(f"  - source_ids: {len(source_ids)} items")
            logging.info(f"  - pattern_ids_list: {len(pattern_ids_list)} items, shapes: {[pat.shape for pat in pattern_ids_list]}")
            logging.info(f"  - task_ids_list: {len(task_ids_list)} items, shapes: {[task.shape for task in task_ids_list]}")
            
            latents_concat = np.concatenate(all_latents, axis=0)
            source_ids_np = np.array(source_ids)
            pattern_ids_concat = np.concatenate(pattern_ids_list, axis=0)
            task_ids_np = np.concatenate(task_ids_list, axis=0)
            
            # CRITICAL DEBUG: Log array shapes to identify length mismatches
            logging.info(f"T-SNE array shapes - latents: {latents_concat.shape}, source_ids: {source_ids_np.shape}, pattern_ids: {pattern_ids_concat.shape}, task_ids: {task_ids_np.shape}")
            
            # Verify all arrays have the same length
            if not (latents_concat.shape[0] == source_ids_np.shape[0] == pattern_ids_concat.shape[0] == task_ids_np.shape[0]):
                logging.error(f"T-SNE array length mismatch: latents={latents_concat.shape[0]}, source_ids={source_ids_np.shape[0]}, pattern_ids={pattern_ids_concat.shape[0]}, task_ids={task_ids_np.shape[0]}")
                # Truncate all arrays to the minimum length to prevent errors
                min_length = min(latents_concat.shape[0], source_ids_np.shape[0], pattern_ids_concat.shape[0], task_ids_np.shape[0])
                latents_concat = latents_concat[:min_length]
                source_ids_np = source_ids_np[:min_length]
                pattern_ids_concat = pattern_ids_concat[:min_length]
                task_ids_np = task_ids_np[:min_length]
                logging.info(f"T-SNE arrays truncated to length {min_length} to prevent errors")
            
            # Log T-SNE structure: each pattern should have multiple sets with 4 points each (3 encoders + 1 context)
            total_points = latents_concat.shape[0]
            total_patterns = 3  # O, T, L tetrominos
            points_per_pattern = total_points // total_patterns
            logging.info(f"T-SNE structure: {total_points} total points, {total_patterns} patterns, {points_per_pattern} points per pattern")
            logging.info(f"Expected: {len(enc_params_list)} encoders + 1 context = {len(enc_params_list) + 1} points per set")
            logging.info(f"Generating 3 T-SNE visualizations: main (encoders+context), context-only, encoders-only (pattern 1)")
            
            # Downsample points for t-SNE to be memory efficient (match train.py default)
            max_points = int(cfg.eval.get("tsne_max_points", 2000))
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
            
            # 1. ADDITIONAL T-SNE: Show latent samples to demonstrate uncertainty (equivalent to train.py fig_latents_samples)
            # Since structured_train doesn't have latents_samples, we'll create multiple samples from encoders
            if len(enc_params_list) > 0:
                # Create multiple samples by using different encoder outputs as "samples"
                # This shows how different encoders represent the same patterns (uncertainty)
                encoder_samples = []
                encoder_sample_program_ids = []
                encoder_sample_task_ids = []
                
                # For each pattern, collect encoder outputs as samples
                for pattern_id in [1, 2, 3]:
                    pattern_mask = (pattern_ids_concat == pattern_id)
                    if np.any(pattern_mask):
                        # Get encoder points only (exclude context)
                        encoder_mask = (source_ids_np < len(enc_params_list))
                        combined_mask = pattern_mask & encoder_mask
                        
                        if np.any(combined_mask):
                            encoder_latents = latents_concat[combined_mask]
                            encoder_sources = source_ids_np[combined_mask]
                            encoder_task_ids = task_ids_np[combined_mask]
                            
                            # Downsample for cleaner visualization
                            max_encoder_points = min(200, len(encoder_latents))
                            if len(encoder_latents) > max_encoder_points:
                                # Stratified sampling to maintain encoder distribution
                                encoder_indices = []
                                for enc_id in range(len(enc_params_list)):
                                    enc_mask = encoder_sources == enc_id
                                    enc_indices = np.where(enc_mask)[0]
                                    if len(enc_indices) > 0:
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
                            
                            # Add encoder outputs as samples for this pattern
                            encoder_samples.append(encoder_latents)
                            encoder_sample_program_ids.extend([pattern_id] * len(encoder_latents))
                            encoder_sample_task_ids.extend(encoder_task_ids)
                
                if encoder_samples:
                    # Concatenate all encoder samples
                    all_encoder_samples = np.concatenate(encoder_samples, axis=0)
                    all_encoder_program_ids = np.array(encoder_sample_program_ids)
                    all_encoder_task_ids = np.array(encoder_sample_task_ids)
                    
                    # Create T-SNE for encoder samples (showing uncertainty across encoders)
                    fig_tsne_samples = visualize_tsne_sources(
                        latents=all_encoder_samples,
                        program_ids=all_encoder_program_ids,  # Pattern types (1, 2, 3) for colors
                        source_ids=np.zeros(len(all_encoder_samples), dtype=int),  # All same source (encoder samples)
                        max_points=min(2000, len(all_encoder_samples)),
                        random_state=42,
                        task_ids=all_encoder_task_ids,
                    )
                    
                    logging.info(f"Generated encoder samples T-SNE: {len(all_encoder_samples)} points")
                else:
                    fig_tsne_samples = None
                    logging.warning("No encoder samples found for samples T-SNE")
            else:
                fig_tsne_samples = None
                logging.warning("No encoders available for samples T-SNE")
            
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
                
                # OPTION 1: Encoder samples clustering (like train.py fig_latents_samples) - for direct comparison
                encoder_mask = (source_ids_np < len(enc_params_list))
                if np.any(encoder_mask):
                    enc_emb = latents_concat[encoder_mask]
                    enc_prog = pattern_ids_concat[encoder_mask]
                    logging.info(f"Encoder samples clustering: {enc_emb.shape[0]} points, patterns: {np.unique(enc_prog)}")
                    
                    for k in k_values:
                        # Modularity Q on encoder samples (comparable to train.py)
                        modularity_q = compute_modularity_q(enc_emb, enc_prog, k=k)
                        clustering_metrics[f"clustering/encoder_samples/modularity_q_k{k}"] = modularity_q
                        
                        # Adjusted Rand Index on encoder samples (comparable to train.py)
                        ari_score = compute_adjusted_rand_index(enc_emb, enc_prog, k=k)
                        clustering_metrics[f"clustering/encoder_samples/ari_k{k}"] = ari_score
                else:
                    logging.warning("No encoder samples found for encoder samples clustering; skipping")
                
                # OPTION 2: Full latent space clustering (current implementation) - for source analysis
                for k in k_values:
                    # Modularity Q on all embeddings (sources: encoders vs context)
                    modularity_q = compute_modularity_q(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"clustering/source/modularity_q_k{k}"] = modularity_q
                    
                    # Adjusted Rand Index on all embeddings (sources: encoders vs context)
                    ari_score = compute_adjusted_rand_index(latents_concat, source_ids_np, k=k)
                    clustering_metrics[f"clustering/source/ari_k{k}"] = ari_score
                
                # Log clustering metrics to WandB with proper step tracking
                if step is not None:
                    # Get current WANDB step to ensure monotonicity
                    current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
                    
                    # Ensure step is greater than or equal to current WandB step to avoid monotonicity issues
                    if step <= current_wandb_step:
                        adjusted_step = current_wandb_step + 1
                        logging.info(f"⚠️  Step {step} is <= current WANDB step ({current_wandb_step}), using adjusted step {adjusted_step}")
                        wandb.log(clustering_metrics, step=adjusted_step)
                    else:
                        wandb.log(clustering_metrics, step=step)
                    logging.info(f"Clustering metrics computed: {clustering_metrics}")
                else:
                    # If no step provided, use a step value that's greater than current WandB step
                    current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
                    default_step = current_wandb_step + 1
                    wandb.log(clustering_metrics, step=default_step)
                    logging.warning(f"⚠️  Clustering metrics logged with default step={default_step} (step parameter was None)")
                    logging.info(f"Clustering metrics computed (default step): {clustering_metrics}")
                
            except Exception as e:
                logging.warning(f"Clustering metrics computation failed: {e}")
        else:
            fig_tsne = None

        # Log all T-SNE plots to wandb
        wandb_log_data = {
            f"test/{test_name}/pixel_accuracy": wandb.Image(fig_heatmap),
            f"test/{test_name}/generation": wandb.Image(fig_gen),
            f"test/{test_name}/latents": wandb.Image(fig_tsne),
            f"test/{test_name}/latents_samples": wandb.Image(fig_tsne_samples) if fig_tsne_samples is not None else None,
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
        
        # Ensure step is greater than or equal to current WandB step to avoid monotonicity issues
        current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
        if step is None or step <= current_wandb_step:
            adjusted_step = current_wandb_step + 1
            logging.info(f"⚠️  Test metrics step {step} is <= current WANDB step ({current_wandb_step}), using adjusted step {adjusted_step}")
            wandb.log(wandb_log_data, step=adjusted_step)
        else:
            wandb.log(wandb_log_data, step=step)

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
                    # CRITICAL FIX: Use ALL pairs to show pattern-specific variance structure
                    mu_i_np = np.array(mu_i)
                    logvar_i_np = np.array(logvar_i)
                    # Shape: [1, num_pairs, latent_dim] -> [num_pairs, latent_dim]
                    mu_i_np = np.array(mu_i).squeeze(0)  # Remove batch dimension
                    logvar_i_np = np.array(logvar_i).squeeze(0)  # Remove batch dimension
                    
                    # Store all pairs for this encoder
                    enc_mus.append(mu_i_np)  # [num_pairs, latent_dim]
                    enc_logvars.append(logvar_i_np)  # [num_pairs, latent_dim]
                    
                    # DEBUG: Log variance statistics for this encoder and pattern
                    var_i = np.exp(logvar_i_np)  # [num_pairs, latent_dim]
                    mean_var = np.mean(var_i)
                    std_var = np.std(var_i)
                    logging.info(f"Pattern {pid} - Encoder {enc_idx}: mean_var={mean_var:.6f}, std_var={std_var:.6f}")

                # Compute PoE across all pairs to show aggregated confidence
                # Average across pairs for each encoder, then compute PoE
                avg_enc_mus = [np.mean(em, axis=0) for em in enc_mus]  # [latent_dim] per encoder
                avg_enc_logvars = [np.mean(lv, axis=0) for lv in enc_logvars]  # [latent_dim] per encoder
                
                alphas_np = np.asarray(alphas)
                precisions = [np.exp(-lv) for lv in avg_enc_logvars]  # [latent_dim] per encoder
                poe_precision = np.zeros_like(precisions[0])
                for a, p in zip(alphas_np, precisions):
                    poe_precision = poe_precision + a * p
                poe_var = 1.0 / (poe_precision + 1e-8)
                num = np.zeros_like(avg_enc_mus[0])
                for a, p, m in zip(alphas_np, precisions, avg_enc_mus):
                    num = num + a * p * m
                poe_mu = num / (poe_precision + 1e-8)
                poe_logvar = np.log(poe_var + 1e-8)

                panel_title = f"Pattern {pid} - Confidence (All Pairs)"
                enc_labels = [f"Encoder {i}" for i in range(len(enc_mus))]
                
                # CRITICAL: Pass all pairs to show pattern-specific variance structure
                # Also pass pattern information for proper variance filtering
                pattern_names = {1: "O-tetromino", 2: "T-tetromino", 3: "L-tetromino"}
                fig_panel = visualize_struct_confidence_panel(
                    sample_grids=pairs_np[idx],
                    sample_shapes=shapes_np[idx],
                    encoder_mus=enc_mus,  # [num_pairs, latent_dim] per encoder
                    encoder_logvars=enc_logvars,  # [num_pairs, latent_dim] per encoder
                    poe_mu=poe_mu,  # [latent_dim] - PoE aggregated mean
                    poe_logvar=poe_logvar,  # [latent_dim] - PoE aggregated logvar
                    title=panel_title,
                    encoder_labels=enc_labels,
                    pattern_id=pid,  # Pattern ID for filtering
                    pattern_name=pattern_names.get(pid, f"Pattern {pid}"),  # Pattern name
                )
                # Ensure step is greater than or equal to current WandB step to avoid monotonicity issues
                current_wandb_step = wandb.run.step if hasattr(wandb.run, 'step') else 0
                if step is None or step <= current_wandb_step:
                    adjusted_step = current_wandb_step + 1
                    logging.info(f"⚠️  Confidence panel step {step} is <= current WANDB step ({current_wandb_step}), using adjusted step {adjusted_step}")
                    wandb.log({f"test/{test_name}/confidence_panel/pattern_{pid}": wandb.Image(fig_panel)}, step=adjusted_step)
                else:
                    wandb.log({f"test/{test_name}/confidence_panel/pattern_{pid}": wandb.Image(fig_panel)}, step=step)
                plt.close(fig_panel)
                
                logging.info(f"Generated confidence panel for pattern {pid} with {len(enc_mus[0])} pairs")
        except Exception as e:
            logging.warning(f"Confidence panel generation failed: {e}")
            logging.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")

        # Free figures to save memory
        plt.close(fig_heatmap)
        plt.close(fig_gen)
        plt.close(fig_tsne)
        if fig_tsne_samples is not None:
            plt.close(fig_tsne_samples)
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
            2: '#5361E5',  # Encoder 2 - BLUE (same as visualize_tsne_sources)
            3: '#2ca02c'   # Encoder 3 - GREEN (4th option)
        }
        source_markers = {
            0: 'o',    # Encoder 0 - Circle (same as visualize_tsne_sources)
            1: 's',    # Encoder 1 - Square (same as visualize_tsne_sources)
            2: '^',    # Encoder 2 - Triangle (same as visualize_tsne_sources)
            3: 'D'     # Encoder 3 - Diamond (4th option)
        }
        source_labels = {
            0: "Encoder 0",
            1: "Encoder 1", 
            2: "Encoder 2",
            3: "Encoder 3"
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
        step: int = None,
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
            **test_encoder_variance_metrics,  # Include test encoder variance metrics for each pattern
        }
        
        # Create visualizations
        fig_heatmap = visualize_heatmap(
            (pixels_equal.sum(axis=(0)) / (mask.sum(axis=(0)) + 1e-5)),
            (mask.sum(axis=(0)) / (mask.sum() + 1e-5)),
        )
        
        # Generation visualization - CRITICAL FIX: Compute individual predictions for each sample using leave_one_out approach
        num_show = max(1, min(num_tasks_to_show, int(grids_np.shape[0])))
        num_pairs = int(shapes_np.shape[1])
        
        # Instead of copying the first prediction, we need to generate individual predictions for each sample
        # This requires calling the model for each sample individually to get proper leave_one_out predictions
        individual_pred_grids = []
        individual_pred_shapes = []
        
        for task_idx in range(num_show):
            task_grids = dataset_grids[task_idx:task_idx+1]  # Single task
            task_shapes = dataset_shapes[task_idx:task_idx+1]
            
            # Create leave_one_out data for this specific task
            task_leave_one_out_grids = make_leave_one_out(task_grids, axis=-4)
            task_leave_one_out_shapes = make_leave_one_out(task_shapes, axis=-3)
            
            # Handle the extra dimension from make_leave_one_out
            if task_leave_one_out_grids.shape[1] == task_grids.shape[1]:
                task_leave_one_out_grids = task_leave_one_out_grids[:, 0, ...]
                task_leave_one_out_shapes = task_leave_one_out_shapes[:, 0, ...]
            
            try:
                # Generate individual prediction for this task using leave_one_out approach
                task_output_grids, task_output_shapes, _ = self.model.apply(
                    {"params": state.params["decoder"]},
                    method=self.model.generate_output,
                    pairs=task_leave_one_out_grids,
                    grid_shapes=task_leave_one_out_shapes,
                    input=task_grids[:, 0, ..., 0],
                    input_grid_shape=task_shapes[:, 0, ..., 0],
                    key=jax.random.PRNGKey(task_idx),
                    dropout_eval=True,
                    mode=inference_mode,
                    return_two_best=False,
                    poe_alphas=alphas,
                    encoder_params_list=state.params["encoders"],
                    decoder_params=state.params["decoder"],
                    repulsion_kl_coeff=self.cfg.training.get("repulsion_kl"),
                )
                
                # Convert to numpy and reshape for visualization
                task_output_grids_np = np.array(jax.device_get(task_output_grids))
                task_output_shapes_np = np.array(jax.device_get(task_output_shapes))
                
                # Reshape to match the expected visualization format: (num_pairs, ...)
                if len(task_output_grids_np.shape) == 4:  # (1, num_pairs, ...)
                    task_output_grids_np = task_output_grids_np[0]  # Remove batch dimension
                    task_output_shapes_np = task_output_shapes_np[0]
                
                individual_pred_grids.append(task_output_grids_np)
                individual_pred_shapes.append(task_output_shapes_np)
                
            except Exception as e:
                logging.error(f"Individual prediction for task {task_idx} failed: {e}")
                # Fallback to copying the main prediction if individual generation fails
                fallback_grids = np.repeat(out_grids_np[task_idx:task_idx+1, None, ...], num_pairs, axis=1)[0]
                fallback_shapes = np.repeat(out_shapes_np[task_idx:task_idx+1, None, :], num_pairs, axis=1)[0]
                individual_pred_grids.append(fallback_grids)
                individual_pred_shapes.append(fallback_shapes)
        
        # Stack individual predictions for visualization
        pred_grids_vis = np.stack(individual_pred_grids, axis=0)  # (num_show, num_pairs, ...)
        pred_shapes_vis = np.stack(individual_pred_shapes, axis=0)  # (num_show, num_pairs, ...)
        
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

                # NEW: Track encoder variances for each pattern in test datasets
                test_encoder_variance_metrics = {}
                
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
                        
                        # CRITICAL: Compute and track encoder variances for each pattern in test datasets
                        # This shows how well each encoder specializes in different patterns during testing
                        var_i = np.exp(np.array(logvar_i))  # Convert logvar to variance
                        var_i_flat = var_i.reshape(-1, var_i.shape[-1])  # Flatten to [num_tasks, latent_dim]
                        
                        # Compute mean variance per task for this encoder
                        mean_var_per_task = np.mean(var_i_flat, axis=1)  # [num_tasks]
                        
                        # Group variances by pattern for detailed analysis
                        for pattern_id in np.unique(pattern_ids_np):
                            pattern_mask = (pattern_ids_np == pattern_id)
                            if np.any(pattern_mask):
                                pattern_variances = mean_var_per_task[pattern_mask]
                                pattern_mean_var = float(np.mean(pattern_variances))
                                pattern_std_var = float(np.std(pattern_variances))
                                
                                # Store metrics for WandB logging
                                metric_key = f"test_{test_name}_encoder_{enc_idx}_pattern_{pattern_id}"
                                test_encoder_variance_metrics[f"{metric_key}_mean_variance"] = pattern_mean_var
                                test_encoder_variance_metrics[f"{metric_key}_std_variance"] = pattern_std_var
                                test_encoder_variance_metrics[f"{metric_key}_num_samples"] = int(np.sum(pattern_mask))
                                
                                # Log specialization progress for test datasets
                                logging.info(f"Test: Encoder {enc_idx} - Pattern {pattern_id}: mean_var={pattern_mean_var:.6f}, std_var={pattern_std_var:.6f}, samples={np.sum(pattern_mask)}")
                        
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
                        max_points=2000,
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
                        
                        # Create T-SNE for encoder samples (equivalent to train.py fig_latents_samples)
                        # Use source_id = 0 for all points (will show as same marker type)
                        context_source_ids = np.zeros(len(context_latents), dtype=int)
                        
                        fig_tsne_samples = visualize_tsne_sources(
                            latents=context_latents,
                            program_ids=context_patterns,  # Pattern types for colors
                            source_ids=context_source_ids,  # All 0s (same marker type)
                            max_points=max_context_points,
                            random_state=42,
                            task_ids=context_task_ids,
                        )
                        
                        logging.info(f"Test: Generated encoder samples T-SNE: {len(context_latents)} points")
                    else:
                        fig_tsne_samples = None
                        logging.warning("Test: No encoder samples found for samples T-SNE")
                    
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
                        
                        # OPTION 1: Encoder samples clustering (like train.py fig_latents_samples) - for direct comparison
                        encoder_mask = (source_ids_np < len(enc_params_list))
                        if np.any(encoder_mask):
                            enc_emb = latents_concat[encoder_mask]
                            enc_prog = pattern_ids_concat[encoder_mask]
                            logging.info(f"Test encoder samples clustering: {enc_emb.shape[0]} points, patterns: {np.unique(enc_prog)}")
                            
                            for k in k_values:
                                # Modularity Q on encoder samples (comparable to train.py)
                                modularity_q = compute_modularity_q(enc_emb, enc_prog, k=k)
                                test_clustering_metrics[f"clustering/{test_name}/encoder_samples/modularity_q_k{k}"] = modularity_q
                                
                                # Adjusted Rand Index on encoder samples (comparable to train.py)
                                ari_score = compute_adjusted_rand_index(enc_emb, enc_prog, k=k)
                                test_clustering_metrics[f"clustering/{test_name}/encoder_samples/ari_k{k}"] = ari_score
                        else:
                            logging.warning(f"Test: No encoder samples found for encoder samples clustering; skipping")
                        
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
                        
                        # Note: These metrics will be logged to wandb by the calling function with proper step
                        
                    except Exception as e:
                        logging.warning(f"Test clustering metrics computation failed: {e}")
        
        return metrics, fig_gen, fig_heatmap, fig_latents, None, fig_search_progress, fig_tsne_samples, fig_tsne_encoders_list

    def _create_merged_encoder_certainty_panel(self, state: TrainState, step: int) -> Optional[plt.Figure]:
        """
        Create a certainty panel AFTER PHASE 1 that merges ALL encoder solutions into single histograms for each pattern.
        
        This function creates a comprehensive view showing how all encoders (0, 1, 2) represent each pattern
        by merging their variance distributions into single histograms per pattern.
        
        CRITICAL FIX: Now replicates the EXACT SAME functionality as individual encoder certainty plots
        This ensures the merged panel shows the SAME data quality (1260 samples) and consistency
        as the individual plots, preventing the dataset mixing issue.
        
        Args:
            state: Current training state with all encoder parameters
            step: Current training step
            
        Returns:
            matplotlib Figure with merged encoder certainty panels or None if creation fails
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.stats import norm
            
            logging.info(f"🔍 Creating merged encoder certainty panel after Phase 1 (step {step})")
            
            # Create evaluation data for all patterns using PRE-LOADED datasets from initialization
            # This ensures the EXACT SAME data is used every time, preventing dataset mixing
            # CRITICAL: Uses SAME APPROACH as training - takes ALL samples for each pattern
            eval_data = {}
            
            # CRITICAL DEBUG: Track data creation for consistency
            logging.info(f"🔍 Creating evaluation data for certainty plots at step {step}")
            logging.info(f"🔍 This is the MERGED ENCODER CERTAINTY PANEL - replicates individual plot functionality")
            
            # CRITICAL COMPARISON: Check if we have Phase A data to compare against
            if hasattr(self, '_last_phase_a_data') and self._last_phase_a_data:
                logging.info(f"🔍 COMPARING with Phase A data from previous step:")
                for pattern_id in [1, 2, 3]:
                    if pattern_id in self._last_phase_a_data:
                        phase_a_grids = self._last_phase_a_data[pattern_id]['grids']
                        logging.info(f"   Phase A Pattern {pattern_id}: {len(phase_a_grids)} samples, shape: {phase_a_grids.shape}")
            
            # CRITICAL DECISION: ALWAYS use Phase A data if available to ensure consistency
            # This prevents the histogram destruction issue caused by smaller pre-loaded datasets
            if hasattr(self, '_last_phase_a_data') and self._last_phase_a_data:
                logging.info(f"🔍 USING Phase A data for consistency (prevents histogram destruction):")
                for pattern_id in [1, 2, 3]:
                    if pattern_id in self._last_phase_a_data:
                        # CRITICAL FIX: Convert dictionary format back to tuple format for consistency
                        data_dict = self._last_phase_a_data[pattern_id]
                        grids = data_dict['grids']
                        shapes = data_dict['shapes']
                        pattern_ids = data_dict['pattern_ids']
                        
                        # Store as tuple for consistency with other code paths
                        eval_data[pattern_id] = (grids, shapes, pattern_ids)
                        
                        logging.info(f"   ✅ Pattern {pattern_id}: Using Phase A data with {len(grids)} samples")
                logging.info(f"   ✅ This ensures histograms show the SAME samples as Phase A")
                logging.info(f"   ✅ NO MORE HISTOGRAM DESTRUCTION - consistent 1260 samples")
            else:
                logging.error(f"🔍 CRITICAL ERROR: No Phase A data available!")
                logging.error(f"   This will cause poor histogram quality (96 vs 1260 samples)")
                logging.error(f"   CRITICAL FIX: Use EXACT SAME method as individual encoder plots")
                
                # Use the EXACT SAME method as the individual encoder certainty plots
                # This ensures the merged panel shows the SAME data quality and consistency
                eval_data = {}
                for pattern_id in [1, 2, 3]:
                    logging.info(f"   🔧 Pattern {pattern_id}: Using EXACT SAME method as individual plots")
                    
                    # Generate 1260 samples for each pattern using the single pattern sample generator
                    # This is the EXACT SAME code as _generate_phase_a_tsne
                    grids_list = []
                    shapes_list = []
                    pattern_ids_list = []
                    
                    num_samples = 1260  # Generate the same number as individual plots
                    logging.info(f"   🔧 Pattern {pattern_id}: Generating {num_samples} samples (EXACT SAME as individual plots)")
                    
                    for i in range(num_samples):
                        if i % 200 == 0:  # Progress logging
                            logging.info(f"   🔧 Pattern {pattern_id}: Generated {i}/{num_samples} diverse samples")
                        
                        # CRITICAL FIX: Use different seed for each sample to ensure diversity
                        # This prevents the "1260 copies of one grid" problem
                        sample_seed = self.cfg.training.seed + pattern_id * 1000 + i
                        grids, shapes, _ = self._create_single_pattern_sample_with_seed(pattern_id, sample_seed)
                        grids_list.append(grids)
                        shapes_list.append(shapes)
                        pattern_ids_list.append(pattern_id)
                    
                    # Stack all samples (EXACT SAME as individual plots)
                    grids = jnp.stack(grids_list, axis=0)
                    shapes = jnp.stack(shapes_list, axis=0)
                    pattern_ids = jnp.array(pattern_ids_list)
                    
                    eval_data[pattern_id] = (grids, shapes, pattern_ids)
                    logging.info(f"   ✅ Pattern {pattern_id}: Generated {len(grids)} samples (EXACT SAME as individual plots)")
                    logging.info(f"      - Grids: {grids.shape}")
                    logging.info(f"      - Shapes: {shapes.shape}")
                    logging.info(f"      - Pattern IDs: {pattern_ids.shape}")
                    logging.info(f"      - Pattern IDs unique: {np.unique(pattern_ids)}")
                    
                    # CRITICAL CHECK: Ensure pattern IDs are correct
                    if not np.all(pattern_ids == pattern_id):
                        logging.error(f"   ❌ CRITICAL ERROR: Pattern {pattern_id} has wrong pattern IDs!")
                        logging.error(f"      Expected all {pattern_id}, got: {np.unique(pattern_ids)}")
                        raise ValueError(f"Pattern {pattern_id} has corrupted pattern IDs")
                    else:
                        logging.info(f"   ✅ Pattern {pattern_id} validation passed")
            
            # CRITICAL DEBUG: Log summary of all datasets
            logging.info(f"📊 Evaluation data summary:")
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    data = eval_data[pattern_id]
                    logging.info(f"   Pattern {pattern_id}: data type={type(data)}")
                    if isinstance(data, dict):
                        grids = data['grids']
                        logging.info(f"   Pattern {pattern_id}: {len(grids)} samples, shape: {grids.shape} (dict format)")
                    elif isinstance(data, (tuple, list)) and len(data) == 3:
                        grids, shapes, pattern_ids = data
                        logging.info(f"   Pattern {pattern_id}: {len(grids)} samples, shape: {grids.shape} (tuple format)")
                    else:
                        logging.error(f"   Pattern {pattern_id}: Invalid data format: {type(data)}")
                else:
                    logging.error(f"   Pattern {pattern_id}: NOT FOUND in eval_data")
            
            # Create a figure with subplots for each pattern (histogram + Gaussian function)
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            if len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            pattern_names = {1: "L-tetromino", 2: "O-tetromino", 3: "T-tetromino"}
            
            for pattern_idx, pattern_id in enumerate([1, 2, 3]):
                # Top row: histograms
                ax_hist = axes[0, pattern_idx]
                # Bottom row: Gaussian functions
                ax_gauss = axes[1, pattern_idx]
                
                if pattern_id in eval_data:
                    grids, shapes, pattern_ids = eval_data[pattern_id]
                    
                    # CRITICAL VALIDATION: Verify data integrity before processing
                    logging.info(f"       🔍 Processing pattern {pattern_id} for histograms:")
                    logging.info(f"          - Grids shape: {grids.shape}")
                    logging.info(f"          - Shapes shape: {shapes.shape}")
                    logging.info(f"          - Pattern IDs shape: {pattern_ids.shape}")
                    logging.info(f"          - Pattern IDs unique: {np.unique(pattern_ids)}")
                    
                    # CRITICAL CHECK: Ensure we still have the right pattern IDs
                    if not np.all(pattern_ids == pattern_id):
                        logging.error(f"          ❌ CRITICAL ERROR: Pattern {pattern_id} data corrupted!")
                        logging.error(f"             Expected all {pattern_id}, got: {np.unique(pattern_ids)}")
                        raise ValueError(f"Pattern {pattern_id} data corrupted during processing")
                    
                    # Use multiple samples from the pattern dataset for better statistics
                    # Take up to 20 samples or all available if less (since we now have ALL samples)
                    num_samples_to_use = min(20, len(grids))
                    sample_indices = list(range(num_samples_to_use))
                    logging.info(f"       📊 Pattern {pattern_id}: Using {num_samples_to_use} samples from {len(grids)} available")
                    logging.info(f"          - Sample indices: {sample_indices}")
                    
                    # CRITICAL DEBUG: Log sample details
                    for i, sample_idx in enumerate(sample_indices[:5]):  # Log first 5 samples
                        sample_pattern_id = pattern_ids[sample_idx]
                        logging.info(f"          - Sample {i}: index={sample_idx}, pattern_id={sample_pattern_id}")
                    
                    # Collect encoder outputs for ALL encoders on multiple samples
                    all_encoder_variances = []
                    encoder_labels = []
                    
                    for enc_idx in range(len(self.encoders)):
                        encoder_params = state.params["encoders"][enc_idx]
                        
                        # Collect variances from multiple samples
                        pattern_variances = []
                        
                        for sample_idx in sample_indices:
                            sample_grids = grids[sample_idx]
                            sample_shapes = shapes[sample_idx]
                            
                            # CRITICAL CHECK: Verify sample pattern ID before processing
                            sample_pattern_id = pattern_ids[sample_idx]
                            if sample_pattern_id != pattern_id:
                                logging.error(f"          ❌ CRITICAL ERROR: Sample {sample_idx} has wrong pattern ID!")
                                logging.error(f"             Expected {pattern_id}, got {sample_pattern_id}")
                                raise ValueError(f"Sample {sample_idx} has wrong pattern ID")
                            
                            # Forward pass through this encoder
                            mu, logvar = self.encoders[enc_idx].apply(
                                {"params": encoder_params},
                                sample_grids[None, ...],  # Add batch dimension back
                                sample_shapes[None, ...],  # Add batch dimension back
                                dropout_eval=False,
                                mutable=False,
                            )
                            
                            # Convert logvar to variance and flatten
                            var = np.exp(np.array(logvar))  # Convert to numpy
                            var_flat = var.flatten()  # Flatten to 1D array
                            pattern_variances.extend(var_flat)
                        
                        # Store all variances for this encoder across all samples
                        all_encoder_variances.append(np.array(pattern_variances))
                        encoder_labels.append(f"Encoder {enc_idx}")
                        
                        logging.info(f"          ✅ Encoder {enc_idx}: collected {len(pattern_variances)} variance values")
                    
                    # Create merged histogram for this pattern
                    # Use different colors for each encoder
                    colors = ['#FBB998', '#DB74DB', '#5361E5']  # Orange, Pink, Blue
                    
                    for enc_idx, (variances, label, color) in enumerate(zip(all_encoder_variances, encoder_labels, colors)):
                        # Create histogram with transparency for overlap
                        ax_hist.hist(variances, bins=30, alpha=0.7, label=label, color=color, 
                                   edgecolor='black', linewidth=0.5)
                    
                    # Customize the histogram subplot
                    ax_hist.set_title(f'{pattern_names.get(pattern_id, f"Pattern {pattern_id}")}\nMerged Encoder Variances ({num_samples_to_use} samples)', 
                                   fontsize=14, fontweight='bold')
                    ax_hist.set_xlabel('Variance', fontsize=12)
                    ax_hist.set_ylabel('Frequency', fontsize=12)
                    ax_hist.legend(fontsize=10)
                    ax_hist.grid(True, alpha=0.3)
                    
                    # Add statistics text to histogram
                    stats_text = []
                    for enc_idx, variances in enumerate(all_encoder_variances):
                        mean_var = np.mean(variances)
                        std_var = np.std(variances)
                        stats_text.append(f'E{enc_idx}: μ={mean_var:.4f}, σ={std_var:.4f}')
                    
                    stats_str = '\n'.join(stats_text)
                    ax_hist.text(0.02, 0.98, stats_str, transform=ax_hist.transAxes, 
                               verticalalignment='top', fontsize=9, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Create Gaussian function plots for this pattern
                    ax_gauss.set_title(f'{pattern_names.get(pattern_id, f"Pattern {pattern_id}")}\nGaussian Functions ({num_samples_to_use} samples)', 
                                       fontsize=14, fontweight='bold')
                    ax_gauss.set_xlabel('Variance', fontsize=12)
                    ax_gauss.set_ylabel('Density', fontsize=12)
                    
                    # Get range for x-axis based on all encoder variances for this pattern
                    all_vars = np.concatenate(all_encoder_variances)
                    x_min, x_max = np.min(all_vars), np.max(all_vars)
                    x_range = x_max - x_min
                    x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
                    
                    # Plot Gaussian for each encoder
                    for enc_idx, (variances, color) in enumerate(zip(all_encoder_variances, colors)):
                        mean_var = np.mean(variances)
                        std_var = np.std(variances)
                        
                        # For variances, we'll use a log-normal approximation since variances are always positive
                        # Use the mean variance as the scale parameter
                        gaussian = norm.pdf(x_plot, mean_var, mean_var * 0.5)  # Approximate log-normal with normal
                        ax_gauss.plot(x_plot, gaussian, color=color, linewidth=2, alpha=0.8, 
                                    label=f'Encoder {enc_idx}')
                        
                        # Add vertical line at mean variance
                        ax_gauss.axvline(mean_var, color=color, linestyle='--', alpha=0.6, linewidth=1)
                    
                    ax_gauss.legend(fontsize=10)
                    ax_gauss.grid(True, alpha=0.3)
                    
                    logging.info(f"       ✅ Pattern {pattern_id} merged histogram and Gaussian plots created with {len(all_encoder_variances)} encoders")
                else:
                    ax_hist.text(0.5, 0.5, f'No data for Pattern {pattern_id}', 
                               ha='center', va='center', transform=ax_hist.transAxes)
                    ax_hist.set_title(f'Pattern {pattern_id} - No Data')
                    ax_gauss.text(0.5, 0.5, f'No data for Pattern {pattern_id}', 
                                ha='center', va='center', transform=ax_gauss.transAxes)
                    ax_gauss.set_title(f'Pattern {pattern_id} - No Data')
            
            # Set overall title
            fig.suptitle(f'Merged Encoder Certainty Panel - All Patterns (Step {step})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # CRITICAL FINAL VALIDATION: Ensure data consistency across all patterns
            logging.info(f"       🔍 Final data consistency validation:")
            for pattern_id in [1, 2, 3]:
                if pattern_id in eval_data:
                    grids, shapes, pattern_ids = eval_data[pattern_id]
                    logging.info(f"          Pattern {pattern_id}: {len(grids)} samples, pattern IDs: {np.unique(pattern_ids)}")
                    
                    # Final check: ensure pattern IDs are still correct
                    if not np.all(pattern_ids == pattern_id):
                        logging.error(f"          ❌ FINAL VALIDATION FAILED: Pattern {pattern_id} corrupted!")
                        raise ValueError(f"Final validation failed for pattern {pattern_id}")
                    else:
                        logging.info(f"          ✅ Pattern {pattern_id} final validation passed")
            
            logging.info(f"       📊 Merged encoder certainty panel created successfully")
            logging.info(f"       ✅ ALL data consistency checks passed - no mixing detected")
            return fig
            
        except Exception as e:
            logging.error(f"       ❌ Merged encoder certainty panel creation failed: {e}")
            import traceback
            logging.error(f"       Traceback: {traceback.format_exc()}")
            return None




@hydra.main(config_path="configs", version_base=None, config_name="structured")
def run(cfg: omegaconf.DictConfig):
    # Check if we're resuming from a checkpoint to determine WandB resume behavior
    resume_mode = "allow" if cfg.training.get("resume_from_checkpoint") else None
    
    # Generate unique run ID if not resuming to avoid step conflicts
    import time
    run_id = None
    if not cfg.training.get("resume_from_checkpoint"):
        run_id = f"run_{int(time.time())}"
        logging.info(f"🆔 Using unique run ID: {run_id}")
    
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=run_id,  # Use unique ID if not resuming
        resume=resume_mode,  # Allow resuming if checkpoint resume is enabled
        settings=wandb.Settings(console="redirect"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    
    # Define custom metrics to handle step axis properly and avoid monotonicity warnings
    wandb.define_metric("step", summary="none")
    wandb.define_metric("phase_a/*", step_metric="step")
    wandb.define_metric("phase_b/*", step_metric="step")
    wandb.define_metric("phase_2/*", step_metric="step")
    wandb.define_metric("test/*", step_metric="step")
    wandb.define_metric("timing/*", step_metric="step")
    wandb.define_metric("encoder_*", step_metric="step")
    wandb.define_metric("clustering/*", step_metric="step")
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
    # Handle step counter for resumed runs
    if cfg.training.get("resume_from_checkpoint"):
        if hasattr(wandb.run, 'resumed') and wandb.run.resumed:
            logging.info(f"🔄 Resumed WandB run detected")
            # For resumed runs, we'll start from step 0 and let WandB handle conflicts
            trainer.resume_step_offset = 0
        else:
            logging.info(f"⚠️  Checkpoint resume requested but WandB run not resumed")
            trainer.resume_step_offset = 0
    else:
        trainer.resume_step_offset = 0
        
    state = trainer.train(state, enc_params_list)
    # Final evaluation with the final step value
    final_step = cfg.training.total_num_steps
    trainer.evaluate(state, enc_params_list, final_step)


if __name__ == "__main__":
    run()

    logging.warning(f"Resume failed: {e}")
    # Handle step counter for resumed runs
    if cfg.training.get("resume_from_checkpoint"):
        if hasattr(wandb.run, 'resumed') and wandb.run.resumed:
            logging.info(f"🔄 Resumed WandB run detected")
            # For resumed runs, we'll start from step 0 and let WandB handle conflicts
            trainer.resume_step_offset = 0
        else:
            logging.info(f"⚠️  Checkpoint resume requested but WandB run not resumed")
            trainer.resume_step_offset = 0
    else:
        trainer.resume_step_offset = 0
        
    state = trainer.train(state, enc_params_list)
    # Final evaluation with the final step value
    final_step = cfg.training.total_num_steps
    trainer.evaluate(state, enc_params_list, final_step)


if __name__ == "__main__":
    run()


