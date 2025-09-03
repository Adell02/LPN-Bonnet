"""Minimal script to merge pre-trained encoders using a prior-corrected PoE.

The script downloads three encoder checkpoints from W&B, freezes them and
trains a single decoder. Each batch corresponds to one pattern; the encoders
not associated with the current pattern are regularised toward the prior to
encourage high uncertainty. Behaviour follows ``train.py`` but is intentionally
compact.
"""

import logging
from typing import List

import hydra
import jax
import jax.numpy as jnp
import optax
import omegaconf
import wandb
from flax.training.train_state import TrainState

from models.transformer import EncoderTransformer, DecoderTransformer
from models.utils import EncoderTransformerConfig, DecoderTransformerConfig
from models.structured_lpn import StructuredLPN, poe_diag_gaussians
from datasets.task_gen.dataloader import make_task_gen_dataloader
from visualization import visualize_tsne, visualize_tsne_sources


logging.getLogger().setLevel(logging.INFO)


def _instantiate_modules(cfg: omegaconf.DictConfig) -> tuple[List[EncoderTransformer], DecoderTransformer]:
    """Instantiate encoder and decoder modules from config."""
    # Prefer structured.model_config if provided to match artifact shapes
    mc = getattr(cfg.structured, "model_config", None)
    if mc is not None:
        if not getattr(mc, "variational", False):
            raise ValueError("Encoders must be variational; set structured.model_config.variational=true.")
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
        enc_cfg = hydra.utils.instantiate(enc_cfg)
        dec_cfg = hydra.utils.instantiate(dec_cfg)
    else:
        # Fallback to explicit encoder/decoder configs
        if not getattr(cfg.encoder_transformer, "variational", False):
            raise ValueError("Encoders must be variational; set encoder_transformer.variational=true.")
        enc_cfg: EncoderTransformerConfig = hydra.utils.instantiate(cfg.encoder_transformer)
        dec_cfg: DecoderTransformerConfig = hydra.utils.instantiate(cfg.decoder_transformer)
    
    encoders = [EncoderTransformer(enc_cfg) for _ in cfg.structured.artifacts.models]
    decoder = DecoderTransformer(dec_cfg)
    return encoders, decoder


def load_artifact_params(artifact_ref: str, key: str = "params") -> dict:
    """Load parameters from W&B artifact (same as structured_train.py)."""
    art = wandb.use_artifact(artifact_ref)
    art_dir = art.download()
    import os
    from flax.serialization import msgpack_restore
    state_path = os.path.join(art_dir, "state.msgpack")
    with open(state_path, "rb") as f:
        data = f.read()
    restored = msgpack_restore(data)
    if isinstance(restored, dict) and "params" in restored:
        return restored["params"]
    return restored


def build_params_from_artifacts(cfg: omegaconf.DictConfig) -> tuple[list[dict], dict]:
    """Build encoder and decoder parameters from W&B artifacts (same as structured_train.py)."""
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
        raise ValueError("No structured.artifacts.models provided.")
    if len(dec_params_list) == 1:
        avg_decoder_params = dec_params_list[0]
    else:
        # Average decoder parameters across models
        avg_decoder_params = jax.tree_util.tree_map(lambda *xs: sum(xs) / len(xs), *dec_params_list)
    
    return enc_params_list, avg_decoder_params


def _kl_to_prior(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence to standard normal prior."""
    return 0.5 * jnp.mean(jnp.exp(logvar) + mu ** 2 - 1.0 - logvar)


def train_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg, key):
    """Single training step with off-domain KL regularization."""
    off_coeff = cfg.training.get("off_domain_kl", 1.0)

    def loss_fn(dec_params, key):
        scoped = {"params": {"decoder": dec_params}}
        # Call the model directly - it handles encoder application and PoE internally
        loss, metrics = model.apply(
            scoped,
            batch["pairs"],
            batch["shapes"],
            dropout_eval=False,
            mode=cfg.training.inference_mode,
            poe_alphas=jnp.array(cfg.structured.alphas),
            encoder_params_list=enc_params,
            decoder_params=dec_params,
            rngs={"latents": key},
            prior_kl_coeff=cfg.training.prior_kl_coeff,
        )
        
        # Add off-domain KL regularization manually
        enc_out = [
            enc.apply({"params": p}, batch["pairs"], batch["shapes"], dropout_eval=True)
            for enc, p in zip(model.encoders, enc_params)
        ]
        mus, logvars = model._stack_encoder_outputs(enc_out)
        kl_reg = 0.0
        for i in range(len(enc_out)):
            kl_i = _kl_to_prior(mus[i], logvars[i])
            metrics[f"kl_prior/enc{i+1}"] = kl_i
            # Check if this encoder is off-domain for any sample in the batch
            is_off_domain = jnp.any(batch["pattern_id"] != i + 1)  # pattern_ids are 1-indexed
            # Use JAX conditional instead of Python if
            kl_reg += jnp.where(is_off_domain, kl_i, 0.0)
        loss += off_coeff * kl_reg
        metrics["off_domain_kl"] = kl_reg
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params["decoder"], key)
    state = state.apply_gradients(grads={"decoder": grads})
    return state, metrics


def eval_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg):
    """Single evaluation step."""
    def eval_fn():
        scoped = {"params": {"decoder": state.params["decoder"]}}
        loss, metrics = model.apply(
            scoped,
            batch["pairs"],
            batch["shapes"],
            dropout_eval=True,
            mode=cfg.training.inference_mode,
            poe_alphas=jnp.array(cfg.structured.alphas),
            encoder_params_list=enc_params,
            decoder_params=state.params["decoder"],
            rngs={"latents": jax.random.PRNGKey(0)},
            prior_kl_coeff=cfg.training.prior_kl_coeff,
        )
        return loss, metrics
    
    return eval_fn()


def generate_contexts_for_tsne(state: TrainState, batch, enc_params, model: StructuredLPN, cfg, key):
    """Generate contexts from individual encoders and PoE for T-SNE visualization."""
    def generate_fn():
        # Get individual encoder outputs
        individual_contexts = []
        for i, (enc, enc_param) in enumerate(zip(model.encoders, enc_params)):
            # Apply individual encoder
            mu_i, logvar_i = enc.apply(
                {"params": enc_param}, 
                batch["pairs"], 
                batch["shapes"], 
                dropout_eval=True
            )
            # Sample latents from individual encoder
            key_i = jax.random.fold_in(key, i)
            latents_i = mu_i + jnp.exp(0.5 * logvar_i) * jax.random.normal(key_i, mu_i.shape)
            # Create context (mean across pairs)
            context_i = latents_i.mean(axis=-2)  # (batch, latent_dim)
            individual_contexts.append(context_i)
        
        # Get PoE context
        enc_outputs = []
        for enc, enc_param in zip(model.encoders, enc_params):
            mu_i, logvar_i = enc.apply(
                {"params": enc_param}, 
                batch["pairs"], 
                batch["shapes"], 
                dropout_eval=True
            )
            enc_outputs.append((mu_i, logvar_i))
        
        # Stack and compute PoE
        mus, logvars = model._stack_encoder_outputs(enc_outputs)
        alphas = jnp.array(cfg.structured.alphas)
        mu_poe, logvar_poe = poe_diag_gaussians(mus, logvars, alphas)
        
        # Sample PoE latents
        key_poe = jax.random.fold_in(key, 999)
        latents_poe = mu_poe + jnp.exp(0.5 * logvar_poe) * jax.random.normal(key_poe, mu_poe.shape)
        poe_context = latents_poe.mean(axis=-2)  # (batch, latent_dim)
        
        return individual_contexts, poe_context, batch["pattern_id"]
    
    return generate_fn()


@hydra.main(config_path="configs", config_name="structured", version_base=None)
def main(cfg: omegaconf.DictConfig):
    # Initialize wandb
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    
    # Build model and load parameters (same as structured_train.py)
    encoders, decoder = _instantiate_modules(cfg)
    enc_params, dec_params = build_params_from_artifacts(cfg)
    model = StructuredLPN(tuple(encoders), decoder)
    
    # Initialize model with dummy data
    key = jax.random.PRNGKey(cfg.training.seed)
    init_key, train_key = jax.random.split(key)
    
    dummy_pairs = jnp.zeros((1, 1, 5, 5, 2), dtype=jnp.int32)
    dummy_shapes = jnp.array([[[5, 5], [5, 5]]], dtype=jnp.int32)
    
    variables = model.init(
        init_key,
        dummy_pairs,
        dummy_shapes,
        dropout_eval=False,
        mode=cfg.training.inference_mode,
        poe_alphas=jnp.asarray(cfg.structured.alphas, dtype=jnp.float32),
        encoder_params_list=enc_params,
        decoder_params=dec_params,
    )
    
    # Create optimizer and training state
    tx = optax.adam(cfg.training.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params={"decoder": dec_params}, tx=tx)

    # Create data loaders for each pattern
    loaders = []
    for pattern in [1, 2, 3]:
        loader = make_task_gen_dataloader(
            batch_size=cfg.training.batch_size,
            log_every_n_steps=cfg.training.log_every_n_steps,
            num_workers=cfg.training.num_workers,
            task_generator_class="STRUCT_PATTERN",
            num_pairs=cfg.training.struct_num_pairs,
            num_devices=jax.device_count(),
            online_data_augmentation=cfg.training.online_data_augmentation,
            pattern=pattern,
            num_rows=5,
            num_cols=5,
        )
        loaders.append(iter(loader))

    # Training loop
    for step in range(cfg.training.total_num_steps):
        pat = step % 3
        pairs, shapes = next(loaders[pat])
        batch = {
            "pairs": pairs,
            "shapes": shapes,
            "pattern_id": jnp.full((pairs.shape[0],), pat + 1, dtype=jnp.int32),
        }
        train_key, subkey = jax.random.split(train_key)
        state, metrics = train_step(state, batch, enc_params, model, cfg, subkey)
        
        if (step + 1) % cfg.training.log_every_n_steps == 0:
            wandb.log(metrics, step=step + 1)
            
        if cfg.training.eval_every_n_logs and (step + 1) % (
            cfg.training.log_every_n_steps * cfg.training.eval_every_n_logs
        ) == 0:
            eval_metrics = {}
            for p in range(3):
                pairs, shapes = next(loaders[p])
                batch = {"pairs": pairs, "shapes": shapes}
                loss, m = eval_step(state, batch, enc_params, model, cfg)
                eval_metrics[f"eval/p{p+1}_loss"] = loss
                eval_metrics.update({f"eval/p{p+1}_{k}": v for k, v in m.items()})
            wandb.log(eval_metrics, step=step + 1)

    # T-SNE Evaluation: Generate contexts from individual encoders and PoE
    if cfg.training.eval_every_n_logs:
        logging.info("Generating T-SNE visualizations...")
        
        # Import matplotlib only when needed to avoid multiprocessing issues
        import matplotlib.pyplot as plt
        
        # Collect contexts from all patterns
        all_individual_contexts = [[] for _ in range(len(enc_params))]
        all_poe_contexts = []
        all_pattern_ids = []
        all_task_ids = []
        
        # Generate evaluation data from all patterns
        eval_key = jax.random.PRNGKey(42)
        for pattern in [1, 2, 3]:
            eval_loader = make_task_gen_dataloader(
                batch_size=min(32, cfg.training.batch_size),
                log_every_n_steps=cfg.training.log_every_n_steps,
                num_workers=cfg.training.num_workers,
                task_generator_class="STRUCT_PATTERN",
                num_pairs=cfg.training.struct_num_pairs,
                num_devices=jax.device_count(),
                online_data_augmentation=cfg.training.online_data_augmentation,
                pattern=pattern,
                num_rows=5,
                num_cols=5,
            )
            
            for batch_idx, (pairs, shapes) in enumerate(eval_loader):
                if batch_idx >= 3:
                    break
                    
                batch = {
                    "pairs": pairs,
                    "shapes": shapes,
                    "pattern_id": jnp.full((pairs.shape[0],), pattern, dtype=jnp.int32),
                }
                
                eval_key, subkey = jax.random.split(eval_key)
                individual_contexts, poe_context, pattern_ids = generate_contexts_for_tsne(
                    state, batch, enc_params, model, cfg, subkey
                )
                
                for enc_idx, context in enumerate(individual_contexts):
                    all_individual_contexts[enc_idx].append(context)
                
                all_poe_contexts.append(poe_context)
                all_pattern_ids.append(pattern_ids)
                
                task_ids = jnp.full((pairs.shape[0],), pattern * 1000 + batch_idx * 100 + jnp.arange(pairs.shape[0]), dtype=jnp.int32)
                all_task_ids.append(task_ids)
        
        # Concatenate all contexts
        for enc_idx in range(len(all_individual_contexts)):
            all_individual_contexts[enc_idx] = jnp.concatenate(all_individual_contexts[enc_idx], axis=0)
        all_poe_contexts = jnp.concatenate(all_poe_contexts, axis=0)
        all_pattern_ids = jnp.concatenate(all_pattern_ids, axis=0)
        all_task_ids = jnp.concatenate(all_task_ids, axis=0)
        
        # Create T-SNE plots
        try:
            # Individual encoder contexts
            for enc_idx, contexts in enumerate(all_individual_contexts):
                if contexts.shape[0] > 0:
                    fig_individual = visualize_tsne(contexts, all_pattern_ids)
                    if fig_individual is not None:
                        wandb.log({f"tsne/encoder_{enc_idx+1}_contexts": wandb.Image(fig_individual)}, step=cfg.training.total_num_steps)
                        plt.close(fig_individual)
            
            # PoE contexts
            if all_poe_contexts.shape[0] > 0:
                fig_poe = visualize_tsne(all_poe_contexts, all_pattern_ids)
                if fig_poe is not None:
                    wandb.log({f"tsne/poe_contexts": wandb.Image(fig_poe)}, step=cfg.training.total_num_steps)
                    plt.close(fig_poe)
            
            # Combined plot
            combined_contexts = []
            combined_pattern_ids = []
            combined_source_ids = []
            combined_task_ids = []
            
            for enc_idx, contexts in enumerate(all_individual_contexts):
                combined_contexts.append(contexts)
                combined_pattern_ids.append(all_pattern_ids)
                combined_source_ids.append(jnp.full((contexts.shape[0],), enc_idx, dtype=jnp.int32))
                combined_task_ids.append(all_task_ids)
            
            combined_contexts.append(all_poe_contexts)
            combined_pattern_ids.append(all_pattern_ids)
            combined_source_ids.append(jnp.full((all_poe_contexts.shape[0],), len(enc_params), dtype=jnp.int32))
            combined_task_ids.append(all_task_ids)
            
            combined_contexts = jnp.concatenate(combined_contexts, axis=0)
            combined_pattern_ids = jnp.concatenate(combined_pattern_ids, axis=0)
            combined_source_ids = jnp.concatenate(combined_source_ids, axis=0)
            combined_task_ids = jnp.concatenate(combined_task_ids, axis=0)
            
            if combined_contexts.shape[0] > 0:
                fig_combined = visualize_tsne_sources(
                    combined_contexts, 
                    combined_pattern_ids, 
                    combined_source_ids,
                    max_points=cfg.training.get("tsne_max_points", 500),
                    task_ids=combined_task_ids
                )
                if fig_combined is not None:
                    wandb.log({f"tsne/combined_encoders_poe": wandb.Image(fig_combined)}, step=cfg.training.total_num_steps)
                    plt.close(fig_combined)
            
            logging.info("T-SNE visualizations completed and logged to wandb")
            
        except Exception as e:
            logging.warning(f"T-SNE visualization failed: {e}")
            import traceback
            traceback.print_exc()

    wandb.finish()


if __name__ == "__main__":
    main()