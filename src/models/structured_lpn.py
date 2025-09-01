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
        
        # If encoder_params_list is provided, use only those encoders
        # Otherwise, use all model encoders
        if encoder_params_list is not None:
            # Use only the encoders that have parameters
            encoders_to_use = self.encoders[:len(encoder_params_list)]
        else:
            # Use all model encoders
            encoders_to_use = self.encoders
            
        for i, enc in enumerate(encoders_to_use):
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
                # CRITICAL FIX: Repulsion loss should SUBTRACT from total loss to spread encoders apart
                # A positive coefficient on repulsion_loss drives encoders toward similarity (wrong!)
                # We want to minimize total loss, so we subtract repulsion_loss to drive KL divergence UP
                # This encourages encoders to become more different (higher KL between them)
                repulsion_loss = self._compute_encoder_repulsion_loss(mus, logvars)
                loss -= repulsion_kl_coeff * repulsion_loss  # SUBTRACT to spread encoders apart
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
                # CRITICAL FIX: Block gradients to PoE to prevent "moving together"
                # This ensures each encoder must independently match the fixed PoE target
                # instead of being able to shift the shared PoE distribution
                mu_poe_fixed = jax.lax.stop_gradient(mu_poe)
                logvar_poe_fixed = jax.lax.stop_gradient(logvar_poe)
                
                contrastive_loss, avg_var_target, avg_var_other = self._compute_contrastive_loss(
                    mus, logvars, mu_poe_fixed, logvar_poe_fixed, pattern_ids, contrastive_kl_coeff
                )
                loss += contrastive_kl_coeff * contrastive_loss
            except Exception as e:
                logging.warning(
                    f"Variance control loss computation failed: {e}. Skipping variance control loss."
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
                repulsion_loss_weighted=-repulsion_kl_coeff * repulsion_loss,  # Negative because we subtract from loss
            )
        if contrastive_kl_coeff is not None and contrastive_kl_coeff > 0:
            metrics.update(
                contrastive_loss=contrastive_loss,
                contrastive_loss_weighted=contrastive_kl_coeff * contrastive_loss,
                # Variance control metrics for monitoring
                contrastive_avg_var_target=avg_var_target if 'avg_var_target' in locals() else 0.0,
                contrastive_avg_var_other=avg_var_other if 'avg_var_other' in locals() else 0.0,
                # Specialization quality metrics
                contrastive_specialization_ratio=avg_var_target / (avg_var_other + 1e-8) if 'avg_var_target' in locals() and 'avg_var_other' in locals() else 1.0,
                contrastive_specialization_score=jnp.log(avg_var_target / (avg_var_other + 1e-8) + 1e-8) if 'avg_var_target' in locals() and 'avg_var_other' in locals() else 0.0,
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
        mu_poe: chex.Array,  # Not used in this implementation but kept for compatibility
        logvar_poe: chex.Array,  # Not used in this implementation but kept for compatibility
        pattern_ids: chex.Array,
        contrastive_kl_coeff: float = 1.0,  # Add configurable coefficient
    ) -> chex.Array:
        """Compute direct variance control loss for encoder specialization.
        
        This loss directly controls the variance of each encoder:
        - Target pattern: variance → 0 (high certainty)
        - Other patterns: variance → ∞ (low certainty)
        
        The loss function is now per-encoder weighted:
        L_total = Σ_e [λ_pos * avg_var_target_e + λ_neg * avg_var_other_e]
        
        Where for each encoder e:
        - avg_var_target_e: average variance of target pattern samples for encoder e
        - avg_var_other_e: average variance of non-target pattern samples for encoder e
        - λ_pos: coefficient for target pattern variance (positive = minimize target variance)
        - λ_neg: coefficient for other pattern variance (negative = maximize other variance)
        
        This encourages:
        - Encoder e to have LOW variance (high certainty) on pattern p_e
        - Encoder e to have HIGH variance (low certainty) on other patterns
        
        Args:
            mus: (E, B, N, H) - means from each encoder (not used in this implementation)
            logvars: (E, B, N, H) - log variances from each encoder  
            mu_poe: (B, N, H) - mean from PoE aggregation (not used)
            logvar_poe: (B, N, H) - log variance from PoE aggregation (not used)
            pattern_ids: (B,) - pattern ID for each sample in batch (1, 2, or 3)
            contrastive_kl_coeff: float - scaling coefficient for the contrastive loss
            
        Returns:
            variance_loss: scalar - encourages encoder specialization through direct variance control
            avg_var_target: scalar - average target pattern variance across all encoders
            avg_var_other: scalar - average other pattern variance across all encoders
        """
        E = mus.shape[0]  # Number of encoders
        B = mus.shape[1]  # Batch size
        
        if E == 0:
            return 0.0, 0.0, 0.0
            
        # CRITICAL: Validate pattern IDs
        unique_patterns = jnp.unique(pattern_ids)
        if len(unique_patterns) < 2:
            # Need at least 2 patterns for contrastive learning to work
            logging.warning(f"Variance control loss requires at least 2 patterns, got {len(unique_patterns)}")
            return 0.0, 0.0, 0.0
        
        # Convert log variances to variances
        var = jnp.exp(logvars)  # (E, B, N, H)
        
        # Average variance over pairs and latent dimensions: (E, B)
        var_per_sample = jnp.mean(var, axis=(-2, -1))
        
        # Create target pattern mask: (E, B)
        # Encoder 0 → Pattern 1, Encoder 1 → Pattern 2, Encoder 2 → Pattern 3
        target_patterns = jnp.arange(1, E + 1, dtype=pattern_ids.dtype)  # [1, 2, 3]
        mask = jnp.where(pattern_ids[None, :] == target_patterns[:, None], 1.0, 0.0)  # (E, B)
        
        # Compute average variance for target and non-target patterns per encoder
        # Target pattern variance: average over samples where mask == 1
        target_var_sum = jnp.sum(var_per_sample * mask, axis=1)  # (E,)
        target_count = jnp.sum(mask, axis=1)  # (E,)
        avg_var_target = jnp.where(target_count > 0, target_var_sum / target_count, 0.0)  # (E,)
        
        # Non-target pattern variance: average over samples where mask == 0
        other_var_sum = jnp.sum(var_per_sample * (1.0 - mask), axis=1)  # (E,)
        other_count = jnp.sum(1.0 - mask, axis=1)  # (E,)
        avg_var_other = jnp.where(other_count > 0, other_var_sum / other_count, 0.0)  # (E,)
        
        # STABILIZATION: Clip variances to prevent extreme values
        # This prevents numerical instability while maintaining the contrastive effect
        clip_threshold = 10.0
        avg_var_target = jnp.clip(avg_var_target, 0.0, clip_threshold)
        avg_var_other = jnp.clip(avg_var_other, 0.0, clip_threshold)
        
        # IMPROVED: Per-encoder weighted loss instead of global averaging
        # This provides stronger specialization pressure for each individual encoder
        
        # Dynamic coefficients based on contrastive_kl_coeff
        # Higher values = more aggressive specialization
        lambda_pos = contrastive_kl_coeff * 0.5   # Positive coefficient for target variance (minimize)
        lambda_neg = -contrastive_kl_coeff * 0.5  # Negative coefficient for other variance (maximize)
        
        # Per-encoder loss: each encoder gets its own specialization signal
        per_encoder_loss = lambda_pos * avg_var_target + lambda_neg * avg_var_other  # (E,)
        
        # Total loss: sum across encoders (not average) for stronger pressure
        variance_loss = jnp.sum(per_encoder_loss)
        
        # Return metrics for monitoring (averaged for logging purposes)
        return variance_loss, jnp.mean(avg_var_target), jnp.mean(avg_var_other)



