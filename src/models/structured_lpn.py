from __future__ import annotations

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.tree_util import tree_map

from models.transformer import EncoderTransformer, DecoderTransformer
from models.utils import EncoderTransformerConfig, DecoderTransformerConfig
from models.lpn import LPN


def average_params(params_list: Sequence[dict]) -> dict:
    """Compute elementwise average of a list of parameter pytrees."""
    assert len(params_list) > 0, "params_list must be non-empty"
    return tree_map(lambda *xs: sum(xs) / len(xs), *params_list)


def poe_diag_gaussians(
    mus: chex.Array, logvars: chex.Array, alphas: chex.Array, eps: float = 1e-8
) -> Tuple[chex.Array, chex.Array]:
    """Weighted Product of Experts for diagonal Gaussians with prior factor.

    Args:
        mus: (E, *B, N, H)
        logvars: (E, *B, N, H)
        alphas: (E,) weights; sum(alphas) <= 1 recommended
    Returns:
        (mu, logvar): (*B, N, H)
    """
    precisions = jnp.exp(-logvars)
    prior_prec = 1.0 - jnp.sum(alphas)
    a = alphas.reshape((-1,) + (1,) * (mus.ndim - 1))
    prec_sum = prior_prec + jnp.sum(a * precisions, axis=0)
    sigma = 1.0 / (prec_sum + eps)
    mu_num = jnp.sum(a * precisions * mus, axis=0)
    mu = sigma * mu_num
    logvar = jnp.log(sigma + eps)
    return mu, logvar


class StructuredLPN(nn.Module):
    """Structured LPN that combines multiple encoders via PoE and a single decoder.

    - Encoders are used only to produce per-pair latents; they remain frozen during fine-tuning.
    - Decoder is a single module whose weights are the average of input decoders; it is the only trainable part.
    - All downstream logic (leave-one-out, modes, decoding) is delegated to an internal LPN core that
      uses the same DecoderTransformer.
    """

    encoders: Tuple[EncoderTransformer, ...]
    decoder: DecoderTransformer

    def setup(self):
        # Core LPN used for loss/decoding utilities (encoder is unused here)
        dummy_encoder = self.encoders[0]
        self._core = LPN(encoder=dummy_encoder, decoder=self.decoder)

    @staticmethod
    def _stack_encoder_outputs(enc_outputs: Sequence[tuple[chex.Array, Optional[chex.Array]]]) -> tuple[chex.Array, chex.Array]:
        mus, logvars = [], []
        for mu_i, logvar_i in enc_outputs:
            mus.append(mu_i)
            if logvar_i is None:
                # If deterministic, assign confident variance
                logvar_i = jnp.full_like(mu_i, -5.0)
            logvars.append(logvar_i)
        return jnp.stack(mus, axis=0), jnp.stack(logvars, axis=0)

    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        mode: str,
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        poe_alphas: Optional[chex.Array] = None,
        encoder_params_list: Optional[Sequence[dict]] = None,
        decoder_params: Optional[dict] = None,
        **mode_kwargs,
    ) -> tuple[chex.Array, dict]:
        """Forward pass mirroring LPN but with PoE latents from multiple encoders.

        Args mirror LPN, with extras:
            poe_alphas: weights for encoders, shape (E,). If None, use uniform.
            encoder_params_list: list of param pytrees for each encoder (frozen).
            decoder_params: params for the single decoder (trainable); if provided, used during apply.
        """
        assert len(self.encoders) >= 1, "At least one encoder is required"

        # Apply each encoder to get (mu, logvar)
        enc_outputs = []
        for i, enc in enumerate(self.encoders):
            params = None if encoder_params_list is None else encoder_params_list[i]
            mu_i, logvar_i = enc.apply(
                {"params": params} if params is not None else None,
                pairs,
                grid_shapes,
                dropout_eval,
                mutable=False,
            )
            enc_outputs.append((mu_i, logvar_i))

        mus, logvars = self._stack_encoder_outputs(enc_outputs)
        E = mus.shape[0]
        # Accept None or empty alphas → use uniform across encoders
        if poe_alphas is None or (hasattr(poe_alphas, "size") and int(poe_alphas.size) == 0):
            poe_alphas = jnp.ones((E,), dtype=mus.dtype) / max(E, 1)
        mu_poe, logvar_poe = poe_diag_gaussians(mus, logvars, poe_alphas)

        # Sample if variational
        key = self.make_rng("latents")
        latents = mu_poe + jnp.exp(0.5 * logvar_poe) * jax.random.normal(key, mu_poe.shape)

        # Optionally replace latents for ablations
        if mode_kwargs.get("remove_encoder_latents", False):
            key_init = self.make_rng("latents_init")
            latents = jax.random.normal(key_init, latents.shape)

        # Delegate rest to core LPN: build leave-one-out and follow same modes
        # Note: pass through decoder_params so decoding uses the averaged decoder weights
        # Provide params under the correct scope expected by LPN (i.e., {'decoder': ...})
        scoped_params = None
        if decoder_params is not None:
            scoped_params = {"params": {"decoder": decoder_params}}
        loss, metrics = self._core.apply(
            scoped_params,
            method=StructuredLPN._core_forward_with_fixed_latents,
            latents=latents,
            pairs=pairs,
            grid_shapes=grid_shapes,
            dropout_eval=dropout_eval,
            mode=mode,
            prior_kl_coeff=prior_kl_coeff,
            pairwise_kl_coeff=pairwise_kl_coeff,
            **mode_kwargs,
        )

        # Add PoE-specific metrics
        metrics = dict(metrics)
        metrics.update(
            poe_prior_weight=(1.0 - jnp.sum(poe_alphas)),
            poe_num_encoders=E,
            poe_alphas_mean=jnp.mean(poe_alphas),
        )
        return loss, metrics

    @staticmethod
    def _core_forward_with_fixed_latents(
        self_lpn: LPN,
        latents: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        mode: str,
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        **mode_kwargs,
    ) -> tuple[chex.Array, dict]:
        # This mirrors LPN.__call__ from the point where latents are available.
        from data_utils import make_leave_one_out
        from jax.numpy.linalg import norm
        from jax.tree_util import tree_map as _tree_map

        assert pairs.shape[-4] > 1
        kl_metrics = {}
        prior_kl_loss = None
        pairwise_kl_loss = None

        leave_one_out_latents = make_leave_one_out(latents, axis=-2)
        if mode == "mean":
            context = leave_one_out_latents.mean(axis=-2)
            loss, metrics = self_lpn._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        elif mode == "all":
            loss, metrics = jax.vmap(
                self_lpn._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
            )(leave_one_out_latents, pairs, grid_shapes, dropout_eval)
            context = latents
            distance_context_latents = norm(latents[..., None, :] - leave_one_out_latents, axis=-1)
        elif mode == "random_search":
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs
            key = self_lpn.make_rng("random_search")
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
            context, _ = self_lpn._get_random_search_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key, **mode_kwargs
            )
            loss, metrics = self_lpn._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs
            key = self_lpn.make_rng("gradient_ascent_random_perturbation")
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
            first_context, _ = self_lpn._get_gradient_ascent_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key, **mode_kwargs
            )
            context = first_context
            loss, metrics = self_lpn._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        elif mode == "evolutionary_search":
            for arg in ["population_size", "num_generations", "mutation_std"]:
                assert arg in mode_kwargs
            key = self_lpn.make_rng("evolutionary_search")
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)
            context, _ = self_lpn._get_evolutionary_search_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key, **mode_kwargs
            )
            loss, metrics = self_lpn._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        leave_one_out_contexts = make_leave_one_out(context, axis=-2)
        cosine_between_contexts = jnp.einsum("...h,...nh->...n", context, leave_one_out_contexts) / (
            jnp.linalg.norm(context, axis=-1)[..., None] * jnp.linalg.norm(leave_one_out_contexts, axis=-1) + 1e-5
        )
        cosine_between_latents = jnp.einsum("...h,...nh->...n", latents, leave_one_out_latents) / (
            jnp.linalg.norm(latents, axis=-1)[..., None] * jnp.linalg.norm(leave_one_out_latents, axis=-1) + 1e-5
        )
        if mode != "all":
            distance_context_latents = jnp.linalg.norm(context - latents, axis=-1)
        metrics.update(
            latents_norm=jnp.linalg.norm(latents, axis=-1),
            context_norm=jnp.linalg.norm(context, axis=-1),
            distance_context_latents=distance_context_latents,
            distance_between_contexts=jnp.linalg.norm(context[..., None, :] - leave_one_out_contexts, axis=-1),
            cosine_between_contexts=cosine_between_contexts,
            distance_between_latents=jnp.linalg.norm(latents[..., None, :] - leave_one_out_latents, axis=-1),
            cosine_between_latents=cosine_between_latents,
        )
        loss, metrics = tree_map(jnp.mean, (loss, metrics))
        metrics.update(kl_metrics)
        if prior_kl_loss is not None:
            if prior_kl_coeff is None:
                raise ValueError("Prior KL coefficient is required when using variational inference.")
            loss += prior_kl_coeff * prior_kl_loss
            if pairwise_kl_coeff is not None:
                loss += pairwise_kl_coeff * pairwise_kl_loss
        return loss, metrics



