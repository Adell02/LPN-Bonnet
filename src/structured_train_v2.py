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
        enc_cfg = hydra.utils.instantiate(enc_cfg)
        dec_cfg = hydra.utils.instantiate(dec_cfg)
    else:
        # Fallback to explicit encoder/decoder configs
        if not getattr(cfg.encoder_transformer, "variational", False):
            raise ValueError(
                "Encoders must be variational; set encoder_transformer.variational=true."
            )
        enc_cfg: EncoderTransformerConfig = hydra.utils.instantiate(cfg.encoder_transformer)
        dec_cfg: DecoderTransformerConfig = hydra.utils.instantiate(cfg.decoder_transformer)
    
    encoders = [EncoderTransformer(enc_cfg) for _ in cfg.structured.artifacts.models]
    decoder = DecoderTransformer(dec_cfg)
    return encoders, decoder


def _load_artifact_params(ref: str) -> dict:
    art = wandb.use_artifact(ref)
    path = art.download()
    import os
    from flax.serialization import msgpack_restore

    with open(os.path.join(path, "state.msgpack"), "rb") as f:
        data = f.read()
    state = msgpack_restore(data)
    return state["params"] if "params" in state else state


def _build_params(cfg: omegaconf.DictConfig) -> tuple[list[dict], dict]:
    enc_params, dec_params = [], []
    for ref in cfg.structured.artifacts.models:
        params = _load_artifact_params(ref)
        enc_params.append(params["encoder"])
        dec_params.append(params["decoder"])
    avg_dec = jax.tree_util.tree_map(lambda *xs: sum(xs) / len(xs), *dec_params)
    return enc_params, avg_dec


def _kl_to_prior(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.mean(jnp.exp(logvar) + mu ** 2 - 1.0 - logvar)


def train_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg, key):
    pattern = int(batch["pattern_id"][0]) - 1
    off_coeff = cfg.training.get("off_domain_kl", 1.0)

    @jax.jit
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
            pattern_ids=batch["pattern_id"],
        )
        
        # Add off-domain KL regularization manually
        # We need to compute this separately since the model doesn't handle it
        enc_out = [
            enc.apply({"params": p}, batch["pairs"], batch["shapes"], dropout_eval=True)
            for enc, p in zip(model.encoders, enc_params)
        ]
        mus, logvars = model._stack_encoder_outputs(enc_out)
        kl_reg = 0.0
        for i in range(len(enc_out)):
            kl_i = _kl_to_prior(mus[i], logvars[i])
            metrics[f"kl_prior/enc{i+1}"] = float(kl_i)
            if i != pattern:
                kl_reg += kl_i
        loss += off_coeff * kl_reg
        metrics["off_domain_kl"] = kl_reg
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params["decoder"], key)
    state = state.apply_gradients(grads={"decoder": grads})
    return state, metrics


def eval_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg):
    @jax.jit
    def eval_fn():
        scoped = {"params": {"decoder": state.params["decoder"]}}
        # Call the model directly - it handles encoder application and PoE internally
        loss, metrics = model.apply(
            scoped,
            batch["pairs"],
            batch["shapes"],
            dropout_eval=True,
            mode=cfg.training.inference_mode,
            poe_alphas=jnp.array(cfg.structured.alphas),
            encoder_params_list=enc_params,
            decoder_params=state.params["decoder"],
            rngs={"latents": jax.random.PRNGKey(0)},  # Use dummy key for eval
            prior_kl_coeff=cfg.training.prior_kl_coeff,
        )
        return loss, metrics
    
    return eval_fn()


@hydra.main(config_path="configs", config_name="structured", version_base=None)
def main(cfg: omegaconf.DictConfig):
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
               config=omegaconf.OmegaConf.to_container(cfg))
    encoders, decoder = _instantiate_modules(cfg)
    enc_params, dec_params = _build_params(cfg)
    model = StructuredLPN(tuple(encoders), decoder)
    
    # Initialize model with dummy data (critical for proper initialization)
    key = jax.random.PRNGKey(cfg.training.seed)
    init_key, train_key = jax.random.split(key)
    
    # Create dummy data for initialization (like train.py and structured_train.py)
    dummy_pairs = jnp.zeros((1, 1, 5, 5, 2), dtype=jnp.int32)  # (batch, pairs, rows, cols, channels)
    dummy_shapes = jnp.array([[[5, 5], [5, 5]]], dtype=jnp.int32)  # (batch, 2, 2)
    
    # Initialize the model
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
    
    tx = optax.adam(cfg.training.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params={"decoder": dec_params}, tx=tx)

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
                eval_metrics[f"eval/p{p+1}_loss"] = float(loss)
                eval_metrics.update({f"eval/p{p+1}_{k}": v for k, v in m.items()})
            wandb.log(eval_metrics, step=step + 1)

    wandb.finish()


if __name__ == "__main__":
    main()
