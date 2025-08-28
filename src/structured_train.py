from __future__ import annotations

import logging
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
    # Instantiate module classes sharing hyperparams with loaded checkpoints
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
    from flax.serialization import from_bytes
    from flax.core import FrozenDict
    from flax.training.train_state import TrainState
    state_path = os.path.join(art_dir, "state.msgpack")
    with open(state_path, "rb") as f:
        data = f.read()
    # Minimal loader: construct a dummy TrainState to extract params
    dummy = TrainState(step=jnp.array(0), apply_fn=lambda *args, **kwargs: None, params=None, tx=optax.adam(1e-3))
    restored = from_bytes(dummy, data)
    return restored.params


def build_params_from_artifacts(cfg: omegaconf.DictConfig, decoder_module: DecoderTransformer) -> tuple[list[dict], dict]:
    enc_params_list = []
    dec_params_list = []
    for art in cfg.structured.artifacts.models:
        full_params = load_artifact_params(art)
        # Expect top-level keys 'encoder' and 'decoder'
        enc_params = full_params["encoder"] if "encoder" in full_params else full_params
        dec_params = full_params["decoder"] if "decoder" in full_params else full_params
        enc_params_list.append(enc_params)
        dec_params_list.append(dec_params)

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

    def _iterate_batches(self, key: chex.PRNGKey, log_every_n_steps: int):
        shuffle_key, aug_key = jax.random.split(key)
        grids, shapes = shuffle_dataset_into_batches(self.train_grids, self.train_shapes, self.batch_size, shuffle_key)
        num_logs = grids.shape[0] // log_every_n_steps
        grids = grids[: num_logs * log_every_n_steps]
        shapes = shapes[: num_logs * log_every_n_steps]
        if self.cfg.training.online_data_augmentation:
            grids, shapes = data_augmentation_fn(grids, shapes, aug_key)
        for i in range(0, grids.shape[0], log_every_n_steps):
            yield grids[i : i + log_every_n_steps], shapes[i : i + log_every_n_steps]

    def train(self, state: TrainState, enc_params_list: list[dict]) -> TrainState:
        cfg = self.cfg
        num_steps = cfg.training.total_num_steps
        log_every = cfg.training.log_every_n_steps
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)

        step = 0
        for batches, shapes in self._iterate_batches(jax.random.PRNGKey(cfg.training.seed), log_every):
            # Grad over decoder params only; encoder params are fed through kwargs and not part of state.params
            def loss_fn(decoder_params, batch_pairs, batch_shapes, rng):
                loss, metrics = self.model.apply(
                    {"params": decoder_params},
                    batch_pairs,
                    batch_shapes,
                    dropout_eval=False,
                    mode=cfg.training.inference_mode,
                    poe_alphas=alphas,
                    encoder_params_list=enc_params_list,
                    decoder_params=decoder_params,
                    rngs={"dropout": rng, "latents": rng},
                    prior_kl_coeff=cfg.training.get("prior_kl_coeff"),
                    pairwise_kl_coeff=cfg.training.get("pairwise_kl_coeff"),
                    **(cfg.training.get("inference_kwargs") or {}),
                )
                return loss, metrics

            batch_pairs, batch_shapes = batches, shapes
            rng = jax.random.PRNGKey(step)
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch_pairs, batch_shapes, rng)
            state = state.apply_gradients(grads=grads)

            # Log
            wandb.log({"train/loss": float(loss), **{f"train/{k}": float(v) for k, v in metrics.items()}}, step=step)

            # Periodic eval and checkpointing parity with unstructured
            if cfg.training.get("eval_every_n_logs") and (step // log_every) % cfg.training.eval_every_n_logs == 0:
                try:
                    self.evaluate(state, enc_params_list)
                except Exception as e:
                    logging.warning(f"Eval failed: {e}")
            if cfg.training.get("save_checkpoint_every_n_logs") and (step // log_every) % cfg.training.save_checkpoint_every_n_logs == 0:
                try:
                    from flax.serialization import msgpack_serialize, to_state_dict
                    with open("state.msgpack", "wb") as outfile:
                        outfile.write(msgpack_serialize(to_state_dict(state)))
                    wandb.save("state.msgpack")
                except Exception as e:
                    logging.warning(f"Checkpoint save failed: {e}")
            step += log_every
            if step >= num_steps:
                break

        return state

    def evaluate(self, state: TrainState, enc_params_list: list[dict]) -> dict:
        if not hasattr(self, "eval_grids"):
            return {}
        cfg = self.cfg
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
        pairs, shapes = self.eval_grids, self.eval_shapes
        key = jax.random.PRNGKey(0)
        # Generate output using PoE latents
        from models.lpn import LPN
        # Reuse StructuredLPN decoding path by calling generate_output from the core via method apply
        output_grids, output_shapes, info = self.model.apply(
            {"params": state.params},
            pairs,
            shapes,
            pairs[:, 0, ..., 0],
            shapes[:, 0, ..., 0],
            key,
            True,
            cfg.eval.inference_mode,
            False,
            method=LPN.generate_output,
            poe_alphas=alphas,
            encoder_params_list=enc_params_list,
            decoder_params=state.params,
            **(cfg.eval.get("inference_kwargs") or {}),
        )

        # Metrics aligned with original train: correctness, pixel_correctness, accuracy
        correct_shapes = jnp.all(output_shapes == shapes[:, 0, ..., 1], axis=-1)
        batch_ndims = len(pairs.shape[:-2])
        row_arange_broadcast = jnp.arange(pairs.shape[-2]).reshape((*batch_ndims * (1,), pairs.shape[-2]))
        col_arange_broadcast = jnp.arange(pairs.shape[-1]).reshape((*batch_ndims * (1,), pairs.shape[-1]))
        input_row_mask = row_arange_broadcast < shapes[:, 0, ..., 1:2]
        input_col_mask = col_arange_broadcast < shapes[:, 0, ..., 1:2]
        input_mask = input_row_mask[..., None] & input_col_mask[..., None, :]
        pixels_equal = jnp.where(input_mask & correct_shapes[..., None, None], (output_grids == pairs[:, 0, ..., 1]), False)
        pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / shapes[:, 0, ..., 1:].prod(axis=(-1))
        accuracy = pixels_equal.sum(axis=(-1, -2)) == shapes[:, 0, ..., 1:].prod(axis=(-1))
        metrics = {
            "eval/correct_shapes": float(jnp.mean(correct_shapes)),
            "eval/pixel_correctness": float(jnp.mean(pixel_correctness)),
            "eval/accuracy": float(jnp.mean(accuracy)),
        }

        # Figures
        fig_heatmap = visualize_heatmap(
            (pixels_equal.sum(axis=(0)) / (input_mask.sum(axis=(0)) + 1e-5)),
            (input_mask.sum(axis=(0)) / (jnp.sum(input_mask) + 1e-5)),
        )
        # Limit number of tasks shown for memory efficiency
        num_show = int(cfg.eval.get("num_tasks_to_show", 5))
        num_show = max(1, min(num_show, int(pairs.shape[0])))
        fig_gen = visualize_dataset_generation(pairs[:num_show], shapes[:num_show], output_grids[:num_show], output_shapes[:num_show], num_show)

        # Latent t-SNE: per-encoder and PoE overlay using marker shapes
        # Compute per-encoder latents once and reuse for PoE
        enc_mus = []
        enc_logvars = []
        all_latents = []
        source_ids = []
        for enc_idx, enc_params in enumerate(enc_params_list):
            mu_i, logvar_i = self.encoders[enc_idx].apply({"params": enc_params}, pairs, shapes, True, mutable=False)
            enc_mus.append(mu_i)
            enc_logvars.append(logvar_i)
            lat = mu_i.mean(axis=-2)
            lat_np = np.array(lat).reshape(-1, lat.shape[-1])
            all_latents.append(lat_np)
            source_ids.extend([enc_idx] * lat_np.shape[0])
        mus = jnp.stack(enc_mus, axis=0)
        logvars = jnp.stack(enc_logvars, axis=0)
        alphas = jnp.asarray(cfg.structured.alphas, dtype=jnp.float32)
        from models.structured_lpn import poe_diag_gaussians
        mu_poe, _ = poe_diag_gaussians(mus, logvars, alphas)
        poe_lat = mu_poe.mean(axis=-2)
        poe_np = np.array(poe_lat).reshape(-1, poe_lat.shape[-1])
        all_latents.append(poe_np)
        source_ids.extend([len(enc_params_list)] * poe_np.shape[0])

        latents_concat = np.concatenate(all_latents, axis=0)
        source_ids_np = np.array(source_ids)

        # Downsample points for t-SNE to be memory efficient
        max_points = int(cfg.eval.get("tsne_max_points", 2000))
        if latents_concat.shape[0] > max_points:
            idx = np.random.RandomState(42).choice(latents_concat.shape[0], size=max_points, replace=False)
            latents_concat = latents_concat[idx]
            source_ids_np = source_ids_np[idx]

        # Compute t-SNE
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=2, max_iter=1000, random_state=42).fit_transform(latents_concat.astype(float))
        # Plot with different markers per source (encoders + poe)
        import matplotlib.pyplot as plt
        marker_list = ['o', 's', '^', 'P', 'X', 'D', 'v', '<', '>', '*']
        fig_tsne, ax = plt.subplots(figsize=(10, 8))
        for src in sorted(set(source_ids_np.tolist())):
            mask = source_ids_np == src
            ax.scatter(emb[mask, 0], emb[mask, 1], marker=marker_list[src % len(marker_list)], label=(f"enc{src}" if src < len(enc_params_list) else "poe"), alpha=0.75, s=40)
        ax.legend()
        ax.set_title("t-SNE of latents: per-encoder vs PoE (markers)")

        wandb.log({
            "eval/fig_heatmap": wandb.Image(fig_heatmap),
            "eval/fig_generation": wandb.Image(fig_gen),
            "eval/fig_tsne_sources": wandb.Image(fig_tsne),
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


