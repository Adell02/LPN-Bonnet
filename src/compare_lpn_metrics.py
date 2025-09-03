#!/usr/bin/env python3
"""
Statistical comparison of ARI and Q-modularity metrics between regular LPN and StructuredLPN.
Downloads W&B artifacts and computes mean, std, p-value, and Cohen's d effect size.
"""

import argparse
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
import wandb
import jax.numpy as jnp
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from datasets.task_gen.dataloader import make_dataset
from models.structured_lpn import StructuredLPN
from models.lpn import LPN
from flax import serialization
import chex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_wandb_artifact(entity: str, project: str, artifact_name: str, version: str = "latest") -> str:
    """Download a W&B artifact and return the local path."""
    logger.info(f"Downloading artifact: {entity}/{project}/{artifact_name}:{version}")
    
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{version}")
    artifact_dir = artifact.download()
    
    # Find the state.msgpack file
    state_file = Path(artifact_dir) / "state.msgpack"
    if not state_file.exists():
        raise FileNotFoundError(f"state.msgpack not found in {artifact_dir}")
    
    logger.info(f"Downloaded artifact to: {state_file}")
    return str(state_file)


def load_model_state(state_path: str, model_class):
    """Load model state from msgpack file."""
    logger.info(f"Loading model state from: {state_path}")
    
    with open(state_path, "rb") as f:
        state_bytes = f.read()
    
    state_dict = serialization.msgpack_restore(state_bytes)
    return state_dict


def create_pattern_mix_dataset(num_samples: int = 96, num_pairs: int = 4) -> tuple:
    """Create a balanced pattern mix dataset (O, T, L tetrominos)."""
    logger.info(f"Creating pattern mix dataset: {num_samples} samples, {num_pairs} pairs")
    
    samples_per_pattern = num_samples // 3  # 32 per pattern
    grids_all, shapes_all = [], []
    
    for pid in (1, 2, 3):  # O, T, L tetrominos
        g, s, _ = make_dataset(
            length=samples_per_pattern,
            num_pairs=num_pairs,
            num_workers=0,
            task_generator_class="STRUCT_PATTERN",
            online_data_augmentation=False,
            seed=42 + pid,  # Different seed per pattern
            pattern=pid,
            pattern_per_task=True,
        )
        grids_all.append(g)
        shapes_all.append(s)
    
    grids = jnp.concatenate(grids_all, axis=0)
    shapes = jnp.concatenate(shapes_all, axis=0)
    
    # Create pattern IDs
    pattern_ids = jnp.concatenate([
        jnp.full((samples_per_pattern,), 1),  # O-tetromino
        jnp.full((samples_per_pattern,), 2),  # T-tetromino
        jnp.full((samples_per_pattern,), 3),  # L-tetromino
    ], axis=0)
    
    logger.info(f"Dataset created: grids={grids.shape}, shapes={shapes.shape}, pattern_ids={pattern_ids.shape}")
    return grids, shapes, pattern_ids


def compute_clustering_metrics(model, state_dict, grids, shapes, pattern_ids, model_type: str):
    """Compute ARI and Q-modularity metrics for a model."""
    logger.info(f"Computing metrics for {model_type}")
    
    # Set up model parameters
    if model_type == "regular":
        params = state_dict["params"]
    else:  # structured
        params = {
            "encoders": state_dict["params"]["encoders"],
            "decoder": state_dict["params"]["decoder"]
        }
    
    # Generate context latents
    if model_type == "regular":
        # Regular LPN: direct model output
        context_latents = model.apply(
            {"params": params},
            grids,
            shapes,
            method=model.generate_output,
            mode="mean",
            return_two_best=False,
            rngs={"dropout": jax.random.PRNGKey(42), "latents": jax.random.PRNGKey(42)}
        )[2]["context"]  # Extract context from info dict
    else:
        # Structured LPN: PoE aggregation
        context_latents = model.apply(
            {"params": params["decoder"]},
            method=model.generate_output,
            grids=grids,
            shapes=shapes,
            mode="mean",
            return_two_best=False,
            poe_alphas=jnp.asarray([0.333, 0.333, 0.334], dtype=jnp.float32),
            encoder_params_list=params["encoders"],
            decoder_params=params["decoder"],
            rngs={"dropout": jax.random.PRNGKey(42), "latents": jax.random.PRNGKey(42)}
        )[2]["context"]  # Extract context from info dict
    
    # Reshape context to (num_samples * num_pairs, latent_dim)
    context_np = np.array(context_latents).reshape(-1, context_latents.shape[-1])
    
    # Repeat pattern IDs for each pair
    repeated_pattern_ids = np.repeat(pattern_ids, 4)  # 4 pairs per sample
    
    # Compute clustering metrics
    from visualization import compute_adjusted_rand_index, compute_modularity_q
    
    ari_scores = []
    q_modularity_scores = []
    
    # Test multiple k values for robustness
    k_values = [3, 5, 10]
    
    for k in k_values:
        try:
            ari = compute_adjusted_rand_index(context_np, repeated_pattern_ids, k=k)
            q_mod = compute_modularity_q(context_np, repeated_pattern_ids, k=k)
            
            ari_scores.append(ari)
            q_modularity_scores.append(q_mod)
            
            logger.info(f"{model_type} k={k}: ARI={ari:.4f}, Q-modularity={q_mod:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute metrics for k={k}: {e}")
            continue
    
    return {
        "ari_scores": ari_scores,
        "q_modularity_scores": q_modularity_scores,
        "context_latents": context_np,
        "pattern_ids": repeated_pattern_ids
    }


def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d


def main():
    parser = argparse.ArgumentParser(description="Compare LPN metrics")
    parser.add_argument("--regular_artifact", required=True, help="Regular LPN artifact name")
    parser.add_argument("--structured_artifact", required=True, help="Structured LPN artifact name")
    parser.add_argument("--entity", default="ga624-imperial-college-london", help="W&B entity")
    parser.add_argument("--project", default="LPN-ARC", help="W&B project")
    parser.add_argument("--num_samples", type=int, default=96, help="Number of samples for evaluation")
    parser.add_argument("--output_file", default="lpn_comparison_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Download artifacts
    regular_state_path = download_wandb_artifact(args.entity, args.project, args.regular_artifact)
    structured_state_path = download_wandb_artifact(args.entity, args.project, args.structured_artifact)
    
    # Create pattern mix dataset
    grids, shapes, pattern_ids = create_pattern_mix_dataset(args.num_samples)
    
    # Initialize models
    model_config = {
        "max_rows": 5,
        "max_cols": 5,
        "num_layers": 2,
        "num_heads": 6,
        "emb_dim_per_head": 12,
        "mlp_dim_factor": 4.0,
        "dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
        "latent_dim": 32,
        "variational": True,
        "latent_projection_bias": False
    }
    
    regular_model = LPN(model_config)
    structured_model = StructuredLPN(model_config)
    
    # Load model states
    regular_state = load_model_state(regular_state_path, regular_model)
    structured_state = load_model_state(structured_state_path, structured_model)
    
    # Compute metrics
    regular_metrics = compute_clustering_metrics(regular_model, regular_state, grids, shapes, pattern_ids, "regular")
    structured_metrics = compute_clustering_metrics(structured_model, structured_state, grids, shapes, pattern_ids, "structured")
    
    # Statistical analysis
    results = []
    
    # ARI comparison
    regular_ari = np.array(regular_metrics["ari_scores"])
    structured_ari = np.array(structured_metrics["ari_scores"])
    
    if len(regular_ari) > 0 and len(structured_ari) > 0:
        # Paired t-test (since same dataset)
        t_stat_ari, p_val_ari = ttest_rel(regular_ari, structured_ari)
        cohens_d_ari = compute_effect_size(structured_ari, regular_ari)
        
        results.append({
            "metric": "ARI",
            "regular_mean": np.mean(regular_ari),
            "regular_std": np.std(regular_ari),
            "structured_mean": np.mean(structured_ari),
            "structured_std": np.std(structured_ari),
            "t_statistic": t_stat_ari,
            "p_value": p_val_ari,
            "cohens_d": cohens_d_ari,
            "effect_size": "large" if abs(cohens_d_ari) > 0.8 else "medium" if abs(cohens_d_ari) > 0.5 else "small"
        })
    
    # Q-modularity comparison
    regular_q = np.array(regular_metrics["q_modularity_scores"])
    structured_q = np.array(structured_metrics["q_modularity_scores"])
    
    if len(regular_q) > 0 and len(structured_q) > 0:
        # Paired t-test (since same dataset)
        t_stat_q, p_val_q = ttest_rel(regular_q, structured_q)
        cohens_d_q = compute_effect_size(structured_q, regular_q)
        
        results.append({
            "metric": "Q_modularity",
            "regular_mean": np.mean(regular_q),
            "regular_std": np.std(regular_q),
            "structured_mean": np.mean(structured_q),
            "structured_std": np.std(structured_q),
            "t_statistic": t_stat_q,
            "p_value": p_val_q,
            "cohens_d": cohens_d_q,
            "effect_size": "large" if abs(cohens_d_q) > 0.8 else "medium" if abs(cohens_d_q) > 0.5 else "small"
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"\n{row['metric']} Comparison:")
        print(f"  Regular LPN:    {row['regular_mean']:.4f} ± {row['regular_std']:.4f}")
        print(f"  Structured LPN: {row['structured_mean']:.4f} ± {row['structured_std']:.4f}")
        print(f"  t-statistic:    {row['t_statistic']:.4f}")
        print(f"  p-value:        {row['p_value']:.4f}")
        print(f"  Cohen's d:      {row['cohens_d']:.4f} ({row['effect_size']} effect)")
        
        if row['p_value'] < 0.05:
            print(f"  Result:         SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result:         No significant difference (p ≥ 0.05)")
    
    print(f"\nResults saved to: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    import jax
    main()
