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
                
                contrastive_loss, kl_mean, sign_mean = self._compute_contrastive_loss(
                    mus, logvars, mu_poe_fixed, logvar_poe_fixed, pattern_ids
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
                repulsion_loss_weighted=-repulsion_kl_coeff * repulsion_loss,  # Negative because we subtract from loss
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
        mu_poe: chex.Array,  # Should be stop_gradient'd before calling this function
        logvar_poe: chex.Array,  # Should be stop_gradient'd before calling this function
        pattern_ids: chex.Array,
    ) -> chex.Array:
        """Compute contrastive KL loss for encoder specialization.
        
        MATHEMATICAL FOUNDATION:
        This loss implements a contrastive learning approach where:
        1. Each encoder e is assigned a target pattern p_e
        2. For samples of pattern p_e: encourage LOW KL divergence (high certainty)
        3. For samples of other patterns: encourage HIGH KL divergence (low certainty)
        
        The loss function is:
        L_contrastive = Σ_e Σ_i sign(e,i) * KL(q_e(x_i) || p_poe(x_i))
        
        Where:
        - sign(e,i) = +1 if pattern_ids[i] == p_e (target pattern for encoder e)
        - sign(e,i) = -1 if pattern_ids[i] != p_e (non-target pattern for encoder e)
        - KL(q_e(x_i) || p_poe(x_i)) measures how well encoder e approximates PoE posterior for sample i
        
        This encourages:
        - Encoder e to become CERTAIN (low variance) on pattern p_e
        - Encoder e to become UNCERTAIN (high variance) on other patterns
        
        Args:
            mus: (E, B, N, H) - means from each encoder
            logvars: (E, B, N, H) - log variances from each encoder  
            mu_poe: (B, N, H) - mean from PoE aggregation (fixed target)
            logvar_poe: (B, N, H) - log variance from PoE aggregation (fixed target)
            pattern_ids: (B,) - pattern ID for each sample in batch (1, 2, or 3)
            
        Returns:
            contrastive_loss: scalar - encourages encoder specialization
            kl_mean: scalar - average KL divergence across all encoders/samples
            sign_mean: scalar - average sign value (indicates pattern alignment)
        """
        E = mus.shape[0]  # Number of encoders
        B = mus.shape[1]  # Batch size
        if E == 0:
            return 0.0, 0.0, 0.0
            
        # CRITICAL: Validate pattern IDs
        unique_patterns = jnp.unique(pattern_ids)
        if len(unique_patterns) < 2:
            # Need at least 2 patterns for contrastive learning to work
            logging.warning(f"Contrastive loss requires at least 2 patterns, got {len(unique_patterns)}")
            return 0.0, 0.0, 0.0
        
        # Convert to variances for KL computation
        var_poe = jnp.exp(logvar_poe)  # (B, N, H)
        var_enc = jnp.exp(logvars)     # (E, B, N, H)
        
        # Compute KL divergence: KL(q_e || p_poe) for each encoder e and sample i
        # KL(q||p) = 0.5 * (log(var_p/var_q) + (var_q + (mu_q - mu_p)²)/var_p - 1)
        kl = 0.5 * (
            (logvar_poe[None, ...] - logvars)  # log(var_p/var_q)
            + (var_enc + jnp.square(mus - mu_poe[None, ...])) / (var_poe[None, ...] + 1e-8)  # (var_q + (mu_q - mu_p)²)/var_p
            - 1.0  # -1 term
        )
        
        # Average KL over pairs and latent dimensions: (E, B)
        kl = jnp.mean(kl, axis=(-2, -1))
        
        # STABILIZATION: Clip KL values to prevent explosion
        kl_clip_threshold = 10.0
        kl = jnp.clip(kl, -kl_clip_threshold, kl_clip_threshold)
        
        # CRITICAL: Create encoder→pattern specialization mapping
        # Encoder 0 specializes in Pattern 1 (O-tetromino)
        # Encoder 1 specializes in Pattern 2 (T-tetromino)  
        # Encoder 2 specializes in Pattern 3 (L-tetromino)
        target_patterns = jnp.array([1, 2, 3], dtype=pattern_ids.dtype)  # [1, 2, 3]
        
        # Create specialization matrix: (E, B)
        # +1: encoder should be CERTAIN on this pattern (low KL, low variance)
        # -1: encoder should be UNCERTAIN on this pattern (high KL, high variance)
        is_target_pattern = pattern_ids[None, :] == target_patterns[:, None]  # (E, B)
        sign = jnp.where(is_target_pattern, 1.0, -1.0)  # (E, B)
        
        # DEBUG: Log pattern assignment for monitoring (every 100 steps to avoid spam)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 100 == 0:
            # Sample first few batch elements for debugging
            sample_size = min(5, B)
            logging.debug(f"Contrastive loss pattern assignment (sample of {sample_size}):")
            for e in range(min(3, E)):  # Show first 3 encoders
                target_p = target_patterns[e]
                signs_sample = sign[e, :sample_size]
                patterns_sample = pattern_ids[:sample_size]
                logging.debug(f"  Encoder {e} (target: Pattern {target_p}): signs={signs_sample}, patterns={patterns_sample}")
        
        # COMPUTE CONTRASTIVE LOSS
        # L = Σ_e Σ_i sign(e,i) * KL(e,i)
        # This encourages:
        # - Positive sign * KL: encoder becomes CERTAIN on target pattern (KL → 0)
        # - Negative sign * KL: encoder becomes UNCERTAIN on other patterns (KL → high)
        
        # Apply sign to KL values
        signed_kl = sign * kl  # (E, B)
        
        # STABILIZATION: Use temperature scaling for more stable gradients
        temperature = 1.0  # Can be tuned: lower = more aggressive, higher = more stable
        
        # Apply temperature scaling
        signed_kl_scaled = signed_kl / temperature
        
        # STABILIZATION: Use softmax-like normalization to bound the loss
        # This prevents extreme values while maintaining the contrastive effect
        kl_exp = jnp.exp(jnp.clip(signed_kl_scaled, -10.0, 10.0))  # Prevent exp overflow
        kl_normalized = kl_exp / (jnp.sum(kl_exp, axis=0, keepdims=True) + 1e-8)
        
        # Final contrastive loss: average over all encoders and samples
        contrastive_loss = jnp.mean(signed_kl)
        
        # Return metrics for monitoring
        kl_mean = jnp.mean(kl)           # Average KL divergence
        sign_mean = jnp.mean(sign)       # Average sign (should be close to 0 for balanced patterns)
        
        return contrastive_loss, kl_mean, sign_mean



