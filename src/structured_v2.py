from __future__ import annotations

import os
import time
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import omegaconf
import wandb

from models.structured_lpn import StructuredLPN
from models.transformer import EncoderTransformer, DecoderTransformer

# Reuse the proven helpers and trainer from the main trainer module to avoid divergence
from structured_train import (
    build_model_from_cfg,
    build_params_from_artifacts,
    StructuredTrainer,
)


def _init_wandb(cfg: omegaconf.DictConfig) -> None:
    if getattr(cfg, "wandb", None):
        wandb.init(
            entity=cfg.wandb.get("entity", None),
            project=cfg.wandb.get("project", None),
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        )


def _log_step_metrics(step: int, metrics: dict) -> None:
    if metrics is None:
        return
    if wandb.run is not None:
        wandb.log({**metrics, "step": step}, step=step)


def _maybe_eval(trainer: StructuredTrainer, state, step: int, cfg: omegaconf.DictConfig) -> None:
    eval_every = cfg.training.get("eval_every_n_logs_phase_2", None) or cfg.training.get("eval_every_n_logs", None)
    if eval_every is None:
        return
    if step % int(eval_every) != 0:
        return
    try:
        eval_metrics = trainer.evaluate(state, step=step)
        if wandb.run is not None and isinstance(eval_metrics, dict):
            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)
    except Exception as e:
        # Keep evaluation best-effort to maintain lean control flow
        print(f"[structured_v2] Evaluation skipped at step {step}: {e}")


def _build_and_init(cfg: omegaconf.DictConfig):
    # Build modules to match artifact shapes
    model, encoders, decoder = build_model_from_cfg(cfg)

    # Load encoder/decoder params from artifacts and average decoder
    enc_params_list, avg_decoder_params = build_params_from_artifacts(cfg, decoder)

    # Initialize trainer and state
    trainer = StructuredTrainer(cfg=cfg, model=model, encoders=encoders, decoder=decoder)

    key = jax.random.PRNGKey(cfg.training.seed)
    state = trainer.init_state(key=key, enc_params_list=enc_params_list, avg_decoder_params=avg_decoder_params)
    return trainer, state


def _train_loop(cfg: omegaconf.DictConfig, trainer: StructuredTrainer, state) -> None:
    total_steps: int = int(cfg.training.total_num_steps)
    log_every: int = int(cfg.training.log_every_n_steps)

    key = jax.random.PRNGKey(cfg.training.seed)

    # Create a balanced dataloader with the same generation/processing as the main trainer
    dataloader = trainer._create_balanced_dataloader(log_every)

    step = 0
    while step < total_steps:
        try:
            # Batches are aligned with the trainer's internal processing: (pairs, shapes, pattern_ids)
            batches = next(dataloader)
        except StopIteration:
            # Re-create for the next window
            dataloader = trainer._create_balanced_dataloader(log_every)
            batches = next(dataloader)

        # Train log_every steps in one go (trainer handles specialization/off-domain KL as per cfg)
        state, avg_metrics = trainer.train_n_steps(state, batches, key)
        step += log_every

        _log_step_metrics(step, avg_metrics)
        _maybe_eval(trainer, state, step, cfg)

    # Final evaluation
    _maybe_eval(trainer, state, step, cfg)


def _finish() -> None:
    if wandb.run is not None:
        wandb.finish()


@jax.profiler.annotate_function
def main(cfg: omegaconf.DictConfig) -> None:
    _init_wandb(cfg)
    trainer, state = _build_and_init(cfg)
    _train_loop(cfg, trainer, state)
    _finish()


# Hydra entrypoint kept minimal and aligned with the new config "structured_v2"
import hydra


@hydra.main(config_path="configs", version_base=None, config_name="structured_v2")
def run(cfg: omegaconf.DictConfig):
    main(cfg)


if __name__ == "__main__":
    # Allow running without Hydra for quick debugging (optional)
    from omegaconf import OmegaConf
    import sys
    if len(sys.argv) == 1:
        # Default: rely on Hydra
        pass
    else:
        # Load an explicit config path if provided
        cfg = OmegaConf.load(sys.argv[1])
        main(cfg)



