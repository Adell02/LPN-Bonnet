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


logging.getLogger().setLevel(logging.INFO)


def _instantiate_modules(cfg: omegaconf.DictConfig) -> tuple[List[EncoderTransformer], DecoderTransformer]:
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


@jax.jit
def train_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg, key):
    pattern = int(batch["pattern_id"][0]) - 1
    off_coeff = cfg.training.get("off_domain_kl", 1.0)

    def loss_fn(dec_params, key):
        enc_out = [
            enc.apply({"params": p}, batch["pairs"], batch["shapes"], train=False)
            for enc, p in zip(model.encoders, enc_params)
        ]
        mus, logvars = model._stack_encoder_outputs(enc_out)
        alphas = jnp.array(cfg.structured.alphas)
        mu_poe, logvar_poe = poe_diag_gaussians(mus, logvars, alphas)
        key, sub = jax.random.split(key)
        latents = mu_poe + jnp.exp(0.5 * logvar_poe) * jax.random.normal(sub, mu_poe.shape)
        scoped = {"params": {"decoder": dec_params}}
        loss, metrics = model._core.apply(
            scoped,
            method=model._core_forward_with_fixed_latents,
            latents=latents,
            pairs=batch["pairs"],
            grid_shapes=batch["shapes"],
            dropout_eval=False,
            mode=cfg.training.inference_mode,
            prior_kl_coeff=cfg.training.prior_kl_coeff,
        )
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


@jax.jit
def eval_step(state: TrainState, batch, enc_params, model: StructuredLPN, cfg):
    enc_out = [
        enc.apply({"params": p}, batch["pairs"], batch["shapes"], train=False)
        for enc, p in zip(model.encoders, enc_params)
    ]
    mus, logvars = model._stack_encoder_outputs(enc_out)
    mu_poe, logvar_poe = poe_diag_gaussians(mus, logvars, jnp.array(cfg.structured.alphas))
    scoped = {"params": {"decoder": state.params["decoder"]}}
    loss, metrics = model._core.apply(
        scoped,
        method=model._core_forward_with_fixed_latents,
        latents=mu_poe,
        pairs=batch["pairs"],
        grid_shapes=batch["shapes"],
        dropout_eval=True,
        mode=cfg.training.inference_mode,
        prior_kl_coeff=cfg.training.prior_kl_coeff,
    )
    return loss, metrics


@hydra.main(config_path="configs", config_name="structured")
def main(cfg: omegaconf.DictConfig):
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
               config=omegaconf.OmegaConf.to_container(cfg))
    encoders, decoder = _instantiate_modules(cfg)
    enc_params, dec_params = _build_params(cfg)
    model = StructuredLPN(tuple(encoders), decoder)
    tx = optax.adam(cfg.training.learning_rate)
    key = jax.random.PRNGKey(cfg.training.seed)
    state = TrainState.create(apply_fn=None, params={"decoder": dec_params}, tx=tx)

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
        key, subkey = jax.random.split(key)
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
