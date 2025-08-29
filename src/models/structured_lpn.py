from __future__ import annotations

import logging
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
    prior_prec = jnp.maximum(1.0 - jnp.sum(alphas), eps)
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
        repulsion_kl_coeff: Optional[float] = None,
        contrastive_kl_coeff: Optional[float] = None,
        pattern_ids: Optional[chex.Array] = None,
        **mode_kwargs,
    ) -> tuple[chex.Array, dict]:
        """Forward pass mirroring LPN but with PoE latents from multiple encoders.

        Args mirror LPN, with extras:
            poe_alphas: weights for encoders, shape (E,). If None, use uniform.
            encoder_params_list: list of param pytrees for each encoder (frozen).
            decoder_params: params for the single decoder (trainable); if provided, used during apply.
        """
        assert len(self.encoders) >= 1, "At least one encoder is required"

        # Check if this is an initialization call (pairs might be dummy data)
        is_initialization = pairs.shape[0] == 1 and pairs.shape[1] == 1
        if is_initialization:
            # Return dummy loss and metrics for initialization
            dummy_loss = jnp.array(0.0)
            dummy_metrics = {
                "poe_prior_weight": jnp.array(0.0),
                "poe_num_encoders": jnp.array(len(self.encoders)),
                "poe_alphas_mean": jnp.array(0.0),
                "repulsion_loss": jnp.array(0.0),
                "repulsion_loss_weighted": jnp.array(0.0),
                "contrastive_loss": jnp.array(0.0),
                "contrastive_loss_weighted": jnp.array(0.0),
            }
            return dummy_loss, dummy_metrics

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

        # Add KL repulsion loss between encoder latents to spread them apart
        repulsion_loss = 0.0
        if repulsion_kl_coeff is not None and repulsion_kl_coeff > 0 and E > 1:
            try:
                # Compute KL divergence between pairs of encoder latents
                # This encourages encoders to produce different latent representations
                repulsion_loss = self._compute_encoder_repulsion_loss(mus, logvars)
                loss += repulsion_kl_coeff * repulsion_loss
            except Exception as e:
                # Gracefully handle any memory or computation errors
                logging.warning(f"Encoder repulsion loss computation failed: {e}. Skipping repulsion loss.")
                repulsion_loss = 0.0

        # Compute contrastive loss to encourage encoder specialization
        contrastive_loss = 0.0
        if (
            contrastive_kl_coeff is not None
            and contrastive_kl_coeff > 0
            and pattern_ids is not None
        ):
            try:
                # Note: Removed debug logging from inside JAX-compiled function
                # as it can cause issues with JAX traced arrays
                
                contrastive_loss, kl_mean, sign_mean = self._compute_contrastive_loss(
                    mus, logvars, mu_poe, logvar_poe, pattern_ids
                )
                loss += contrastive_kl_coeff * contrastive_loss
            except Exception as e:
                logging.warning(
                    f"Contrastive loss computation failed: {e}. Skipping contrastive loss."
                )
                contrastive_loss = 0.0

        # Add PoE-specific metrics
        metrics = dict(metrics)
        metrics.update(
            poe_prior_weight=(1.0 - jnp.sum(poe_alphas)),
            poe_num_encoders=E,
            poe_alphas_mean=jnp.mean(poe_alphas),
        )

        # Add repulsion and contrastive loss metrics
        if repulsion_kl_coeff is not None and repulsion_kl_coeff > 0:
            metrics.update(
                repulsion_loss=repulsion_loss,
                repulsion_loss_weighted=repulsion_kl_coeff * repulsion_loss,
            )
        if contrastive_kl_coeff is not None and contrastive_kl_coeff > 0:
            metrics.update(
                contrastive_loss=contrastive_loss,
                contrastive_loss_weighted=contrastive_kl_coeff * contrastive_loss,
                # Additional contrastive loss metrics for monitoring
                contrastive_kl_mean=kl_mean if 'kl_mean' in locals() else 0.0,
                contrastive_sign_mean=sign_mean if 'sign_mean' in locals() else 0.0,
            )

        return loss, metrics

    def generate_output(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        input: chex.Array,
        input_grid_shape: chex.Array,
        key: Optional[chex.PRNGKey],
        dropout_eval: bool,
        mode: str,
        return_two_best: bool = False,
        poe_alphas: Optional[chex.Array] = None,
        encoder_params_list: Optional[Sequence[dict]] = None,
        decoder_params: Optional[dict] = None,
        **mode_kwargs,
    ) -> tuple:
        """Generate outputs using PoE latents and the core LPN decoder.

        Mirrors LPN.generate_output but sources latents from encoders via PoE and
        uses the single decoder.
        """
        # 1) run all encoders
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
        if poe_alphas is None or (hasattr(poe_alphas, "size") and int(poe_alphas.size) == 0):
            poe_alphas = jnp.ones((E,), dtype=mus.dtype) / max(E, 1)
        mu_poe, logvar_poe = poe_diag_gaussians(mus, logvars, poe_alphas)

        # 2) sample if variational
        assert key is not None, "'key' is required for stochastic generation"
        key, key_lat = jax.random.split(key)
        
        # Generate single sample for generation (like regular LPN)
        latents = mu_poe + jnp.exp(0.5 * logvar_poe) * jax.random.normal(key_lat, mu_poe.shape)

        # 3) optionally replace latents
        if mode_kwargs.get("remove_encoder_latents", False):
            key, key_init = jax.random.split(key)
            latents = jax.random.normal(key_init, latents.shape)

        info = {}
        # 4) select context like in LPN, using core helpers
        if mode == "mean":
            first_context = latents.mean(axis=-2)
            second_context = first_context
            info = {"context": first_context}
        elif mode == "first":
            first_context = latents[..., 0, :]
            second_context = first_context
            info = {"context": first_context}
        elif mode == "random_search":
            assert key is not None
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs
            key, k = jax.random.split(key)
            first_context, second_context = self._core._get_random_search_context(
                latents, pairs, grid_shapes, k, **mode_kwargs
            )
            info = {"context": first_context}
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs
            key, k = jax.random.split(key)
            first_context, second_context = self._core._get_gradient_ascent_context(
                latents, pairs, grid_shapes, k, **mode_kwargs
            )
            info = {"context": first_context}
        elif mode == "evolutionary_search":
            for arg in ["population_size", "num_generations", "mutation_std"]:
                assert arg in mode_kwargs
            key, k = jax.random.split(key)
            first_context, second_context = self._core._get_evolutionary_search_context(
                latents, pairs, grid_shapes, k, **mode_kwargs
            )
            info = {"context": first_context}
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 5) decode using core's generator with the provided decoder params
        scoped_params = None
        if decoder_params is not None:
            scoped_params = {"params": {"decoder": decoder_params}}
        output_grids, output_shapes = self._core.apply(
            scoped_params,
            method=self._core._generate_output_from_context,
            context=first_context,
            input=input,
            input_grid_shape=input_grid_shape,
            dropout_eval=dropout_eval,
        )
        if return_two_best:
            second_output_grids, second_output_shapes = self._core.apply(
                scoped_params,
                method=self._core._generate_output_from_context,
                context=second_context,
                input=input,
                input_grid_shape=input_grid_shape,
                dropout_eval=dropout_eval,
            )
            return output_grids, output_shapes, second_output_grids, second_output_shapes, info
        else:
            return output_grids, output_shapes, info

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

    def _compute_encoder_repulsion_loss(self, mus: chex.Array, logvars: chex.Array) -> chex.Array:
        """Compute KL repulsion loss between encoder latents to spread them apart.
        
        Args:
            mus: (E, *B, N, H) - means from each encoder
            logvars: (E, *B, N, H) - log variances from each encoder
            
        Returns:
            repulsion_loss: scalar - average KL divergence between encoder pairs
        """
        E = mus.shape[0]
        if E <= 1:
            return 0.0
            
        # Compute KL divergence between all pairs of encoders
        # KL(p_i || p_j) where p_i and p_j are the latent distributions from encoders i and j
        total_kl = 0.0
        num_pairs = 0
        
        for i in range(E):
            for j in range(i + 1, E):
                # KL divergence between two Gaussian distributions
                # KL(N(mu_i, var_i) || N(mu_j, var_j))
                mu_i, mu_j = mus[i], mus[j]
                var_i, var_j = jnp.exp(logvars[i]), jnp.exp(logvars[j])
                
                # KL divergence formula: 0.5 * (log(var_j/var_i) + var_i/var_j + (mu_i-mu_j)^2/var_j - 1)
                kl_div = 0.5 * (
                    jnp.log(var_j / (var_i + 1e-8)) + 
                    var_i / (var_j + 1e-8) + 
                    jnp.square(mu_i - mu_j) / (var_j + 1e-8) - 1.0
                )
                
                # Average over batch and latent dimensions
                kl_div = jnp.mean(kl_div)
                total_kl += kl_div
                num_pairs += 1
        
        # Return average KL divergence across all encoder pairs
        return total_kl / max(num_pairs, 1)

    def _compute_contrastive_loss(
        self,
        mus: chex.Array,
        logvars: chex.Array,
        mu_poe: chex.Array,
        logvar_poe: chex.Array,
        pattern_ids: chex.Array,
    ) -> chex.Array:
        """Compute contrastive KL loss for encoder specialization.
        
        This loss encourages each encoder to produce lower KL divergence (better alignment)
        for patterns it should specialize in, and higher KL divergence for other patterns.
        
        Args:
            mus: (E, B, N, H) - means from each encoder
            logvars: (E, B, N, H) - log variances from each encoder  
            mu_poe: (B, N, H) - mean from PoE aggregation
            logvar_poe: (B, N, H) - log variance from PoE aggregation
            pattern_ids: (B,) - pattern ID for each sample in batch
            
        Returns:
            contrastive_loss: scalar - encourages encoder specialization
            kl_mean: scalar - average KL divergence across all encoders/samples
            sign_mean: scalar - average sign value (indicates pattern alignment)
        """
        E = mus.shape[0]
        if E == 0:
            return 0.0, 0.0, 0.0
            
        # Note: Removed debug logging from inside JAX-compiled function
        # as it can cause issues with JAX traced arrays
        
        var_poe = jnp.exp(logvar_poe)
        var_enc = jnp.exp(logvars)
        
        # KL(q_e || p_poe) for each encoder e: measures how well each encoder
        # approximates the PoE posterior for each sample
        kl = 0.5 * (
            (logvar_poe[None, ...] - logvars)
            + (var_enc + jnp.square(mus - mu_poe[None, ...])) / (var_poe[None, ...] + 1e-8)
            - 1.0
        )
        kl = jnp.mean(kl, axis=(-2, -1))  # (E, B) - average over pairs and latent dims

        # Create encoder IDs: 1, 2, 3 for encoders 0, 1, 2
        # This assumes encoder 0 specializes in pattern 1, encoder 1 in pattern 2, etc.
        enc_ids = jnp.arange(1, E + 1, dtype=pattern_ids.dtype)[:, None]
        
        # Create sign matrix: 
        # +1 if pattern matches encoder specialization (encourage lower KL)
        # -1 if pattern doesn't match (encourage higher KL)
        sign = jnp.where(pattern_ids[None, :] == enc_ids, 1.0, -1.0)
        
        # Compute contrastive loss: 
        # - Positive sign * KL: encourages encoders to have lower KL for their specialized patterns
        # - Negative sign * KL: encourages encoders to have higher KL for non-specialized patterns
        # This should lead to encoder specialization
        contrastive = jnp.mean(sign * kl)
        
        # Return additional metrics for monitoring
        return contrastive, jnp.mean(kl), jnp.mean(sign)



