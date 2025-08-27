#!/usr/bin/env python3
"""
Run gradient_ascent and evolutionary_search on a single W&B model artifact and
store latent search trajectories to NPZ. Then plot both trajectories (2D latents) and
upload comprehensive metrics, artifacts, and plots to W&B for analysis.

Features:
- Automatic PCA projection for high-dimensional latents
- Comprehensive metrics logging (losses, scores, log probabilities)
- Downloadable CSV files for all metrics
- Statistical analysis across multiple runs
- Beautiful 2D visualizations with viridis background

Example usage:

# Basic usage with automatic budget-based calculations:
python src/store_latent_search.py --wandb_artifact_path "entity/project/artifact:v0" --budget 100

# Multiple runs for statistical analysis:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --n_samples 10

# Custom GA and ES parameters:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --ga_steps 50 \
    --ga_lr 0.1 \
    --es_population 20 \
    --es_generations 5 \
    --es_mutation_std 0.3

# With subspace mutation and progress tracking:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --use_subspace_mutation \
    --subspace_dim 16 \
    --ga_step_length 0.5 \
    --track_progress

# For small-scale searches with smooth background:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --background_resolution 800 \
    --background_smoothing \
    --background_knn 7 \
    --background_bandwidth_scale 1.5 \
    --background_global_mix 0.08 \
    --ga_steps 100 \
    --es_population 50 \
    --es_generations 2

W&B Integration:
- Logs comprehensive metrics for both GA and ES methods
- Provides downloadable CSV files for all metrics
- Creates artifacts for latent trajectories and plots
- Supports grouped runs for statistical analysis
- Tracks convergence, improvements, and comparisons
"""

import argparse
import math
import os
import sys
import subprocess
import time
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass


def build_dataset_args(args: argparse.Namespace) -> list[str]:
    src_args: list[str] = []
    if args.json_challenges and args.json_solutions:
        src_args += ["-jc", args.json_challenges, "-js", args.json_solutions]
    elif args.dataset_folder:
        src_args += ["-d", args.dataset_folder]
        if args.dataset_length is not None:
            src_args += ["--dataset-length", str(args.dataset_length)]
        if args.dataset_batch_size is not None:
            src_args += ["--dataset-batch-size", str(args.dataset_batch_size)]
        src_args += ["--dataset-use-hf", args.dataset_use_hf]
        src_args += ["--dataset-seed", str(args.dataset_seed)]
    return src_args


def try_extract_2d_points(npz: np.lib.npyio.NpzFile, prefix: str) -> Optional[np.ndarray]:
    # prefer path first so GA uses the actual trajectory, then best-of-gen, then raw sets
    preferred = [
        f"{prefix}path",
        f"{prefix}best_latents_per_generation",
        f"{prefix}latents",
        f"{prefix}all_latents",
    ]
    for key in preferred:
        if key in npz:
            arr = np.array(npz[key])
            if arr.ndim >= 2:
                # Accept any dimension - PCA will handle reduction to 2D if needed
                print(f"[plot] Using key '{key}' for {prefix} points, shape={arr.shape}")
                return arr
    # Fallback: first array with at least 2 dimensions
    for key in npz.files:
        if not key.startswith(prefix):
            continue
        arr = np.array(npz[key])
        if arr.ndim >= 2:
            return arr
    return None


@dataclass
class Trace:
    pts: Optional[np.ndarray] = None              # GA path or ES path (T,2)
    vals: Optional[np.ndarray] = None             # values for pts (T,)
    best_per_gen: Optional[np.ndarray] = None     # ES best per generation (G,2)
    pop_pts: Optional[np.ndarray] = None          # ES population points (N,2)
    gen_idx: Optional[np.ndarray] = None          # ES generation id (N,)
    pop_vals: Optional[np.ndarray] = None         # ES population values (N,)


def _extract_vals(npz, prefix: str) -> Optional[np.ndarray]:
    # values associated with the *trajectory* points (GA path or ES path)
    for k in [
        f"{prefix}losses",
        f"{prefix}scores",
        f"{prefix}all_losses",
        f"{prefix}all_scores",
        f"{prefix}best_scores_per_generation",
        f"{prefix}best_losses_per_generation",
        f"{prefix}generation_losses",  # ES trajectory values (new key)
        f"{prefix}best_losses_per_generation",  # ES trajectory values (new key)
    ]:
        if k in npz:
            arr = np.array(npz[k]).reshape(-1)
            if arr.ndim == 1 and arr.size > 0:
                print(f"[plot] Using values key '{k}', length={arr.size}")
                return arr
    
    # Special handling for ES: if we have best_latents_per_generation but no values,
    # we need to extract the trajectory from the saved data
    if prefix == "es_" and f"{prefix}best_latents_per_generation" in npz:
        print(f"[plot] ES trajectory found but no values - this indicates a data saving issue")
        print(f"[plot] Available keys: {list(npz.keys())}")
        
        # Try to use generation_losses for ES trajectory
        if f"{prefix}generation_losses" in npz:
            gen_losses = np.array(npz[f"{prefix}generation_losses"]).reshape(-1)
            if gen_losses.ndim == 1 and gen_losses.size > 0:
                print(f"[plot] Using ES generation_losses for trajectory: {gen_losses.shape}")
                return gen_losses
    
    return None


def _extract_best_per_gen(npz, prefix: str) -> Optional[np.ndarray]:
    for k in [f"{prefix}best_latents_per_generation", f"{prefix}elite_latents"]:
        if k in npz:
            arr = np.array(npz[k])
            if arr.ndim >= 2:
                # Accept any dimension - PCA will handle reduction to 2D if needed
                return arr
    return None


def _extract_pop(npz, prefix: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract ES population data with proper loss pairing.
    Prefer per-sample losses to ensure counts match.
    """
    pts = gens = vals = None
    
    if f"{prefix}all_latents" in npz:
        pts = np.array(npz[f"{prefix}all_latents"])
        # Don't reshape here - PCA will handle dimension reduction
    
    if f"{prefix}generation_idx" in npz:
        gens = np.array(npz[f"{prefix}generation_idx"]).reshape(-1)
    
    # Prefer per-sample losses for proper population pairing
    if f"{prefix}losses_per_generation" in npz:
        lp = np.array(npz[f"{prefix}losses_per_generation"])
        # losses_per_generation is typically (G, P) - need to flatten to (G*P,)
        if lp.ndim == 2:
            vals = lp.reshape(-1)  # (G*P,) - flattened per-generation losses
        else:
            vals = lp.reshape(-1)  # fallback for other shapes
        print(f"[plot] Using {prefix}losses_per_generation: {lp.shape} -> {vals.shape}")
    elif f"{prefix}all_losses" in npz and gens is not None:
        # Expand per-gen losses to population using the generation index
        per_gen = np.array(npz[f"{prefix}all_losses"]).reshape(-1)
        vals = per_gen[gens]  # broadcast by index to match all_latents
        print(f"[plot] Using {prefix}all_losses expanded: {per_gen.shape} -> {vals.shape}")
    elif f"{prefix}all_scores" in npz:
        vals = np.array(npz[f"{prefix}all_scores"]).reshape(-1)
        print(f"[plot] Using {prefix}all_scores: {vals.shape}")
    
    # Special handling for ES: if we have generation index but mismatched values,
    # try to reconstruct per-individual losses from available data
    if prefix == "es_" and pts is not None and vals is not None and gens is not None:
        if len(vals) != len(pts):
            print(f"[plot] ES mismatch detected: {len(vals)} values vs {len(pts)} points")
            print(f"[plot] Attempting to reconstruct per-individual losses...")
            
            # Check if we can use generation_losses to expand
            if f"{prefix}generation_losses" in npz:
                gen_losses = np.array(npz[f"{prefix}generation_losses"]).reshape(-1)
                if len(gen_losses) == len(np.unique(gens)):
                    # We have per-generation losses, expand to per-individual
                    # This assumes all individuals in a generation get the same loss
                    expanded_vals = gen_losses[gens]
                    if len(expanded_vals) == len(pts):
                        vals = expanded_vals
                        print(f"[plot] ES reconstruction successful: {len(vals)} values now match {len(pts)} points")
                    else:
                        print(f"[plot] ES reconstruction failed: expanded={len(expanded_vals)}, expected={len(pts)}")
                else:
                    print(f"[plot] ES generation_losses mismatch: {len(gen_losses)} vs {len(np.unique(gens))} generations")
            else:
                print(f"[plot] ES generation_losses not available for reconstruction")
    
    # Debug: show what we're working with for ES
    if prefix == "es_":
        print(f"[plot] ES population debug: pts={pts.shape if pts is not None else None}, "
              f"gens={gens.shape if gens is not None else None}, vals={vals.shape if vals is not None else None}")
        if pts is not None and vals is not None:
            expected_vals = pts.shape[0] if pts.ndim >= 1 else 0
            print(f"[plot] ES expected: {expected_vals} values for {pts.shape[0]} population points")
    
    return pts, gens, vals


def _collapse_to_steps(arr: np.ndarray, steps_len: int) -> np.ndarray:
    """
    Collapse GA latents to one point per step by averaging over non-step axes.
    This ensures T matches loss length for proper pairing.
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        return arr
    
    D = arr.shape[-1]  # latent dimension
    
    # Find axis with size == steps_len among all but last
    axes = [i for i in range(arr.ndim - 1) if arr.shape[i] == steps_len]
    if not axes:
        # Fall back: assume second-to-last is T
        t_axis = arr.ndim - 2
    else:
        t_axis = axes[-1]
    
    print(f"[_collapse_to_steps] Input shape: {arr.shape}, steps_len: {steps_len}, t_axis: {t_axis}")
    
    # Move T to position -2 so final shape is (..., T, D)
    order = [i for i in range(arr.ndim) if i != t_axis and i != arr.ndim - 1] + [t_axis, arr.ndim - 1]
    arr = np.transpose(arr, order)
    print(f"[_collapse_to_steps] After transpose: {arr.shape}")
    
    # Now average over all leading axes except T and D
    lead_axes = tuple(range(arr.ndim - 2))
    if lead_axes:
        print(f"[_collapse_to_steps] Averaging over axes: {lead_axes}")
        arr = arr.mean(axis=lead_axes)
    
    # arr is now (T, D)
    print(f"[_collapse_to_steps] Final shape: {arr.shape}")
    return arr


def _load_trace(npz_path: str, prefix: str) -> Trace:
    t = Trace()
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as f:
            print(f"[plot] Available keys for {prefix}: {list(f.keys())}")
            t.pts = try_extract_2d_points(f, prefix)
            t.vals = _extract_vals(f, prefix)
            
            # Fix GA shape mismatch: collapse latents to one point per step
            if prefix == "ga_" and t.pts is not None and t.vals is not None:
                print(f"[plot] GA shape fix: pts={t.pts.shape}, vals={t.vals.shape}")
                t.pts = _collapse_to_steps(t.pts, steps_len=len(t.vals))
                print(f"[plot] GA shape after fix: pts={t.pts.shape}, vals={t.vals.shape}")
            
            t.best_per_gen = _extract_best_per_gen(f, prefix)
            pop_pts, gen_idx, pop_vals = _extract_pop(f, prefix)
            t.pop_pts = pop_pts
            t.gen_idx = gen_idx
            t.pop_vals = pop_vals
            
            # Debug: show what was extracted for best_per_gen
            if prefix == "es_":
                print(f"[plot] ES best_per_gen extracted: {t.best_per_gen.shape if t.best_per_gen is not None else None}")
                if t.best_per_gen is not None:
                    print(f"[plot] ES best_per_gen content: min={t.best_per_gen.min():.3f}, max={t.best_per_gen.max():.3f}")
            
            # Debug: show what we extracted
            if prefix == "es_":
                print(f"[plot] ES trace: pts={t.pts.shape if t.pts is not None else None}, "
                      f"vals={t.vals.shape if t.vals is not None else None}, "
                      f"pop_pts={t.pop_pts.shape if t.pop_pts is not None else None}, "
                      f"pop_vals={t.pop_vals.shape if t.pop_vals is not None else None}, "
                      f"gen_idx={t.gen_idx.shape if t.gen_idx is not None else None}")
            
            # Debug: show trajectory data for both GA and ES
            if t.pts is not None:
                print(f"[plot] {prefix} trajectory: pts={t.pts.shape}, vals={t.vals.shape if t.vals is not None else None}")
            else:
                print(f"[plot] {prefix} trajectory: NO TRAJECTORY POINTS EXTRACTED!")
                if prefix == "es_":
                    print(f"[plot] ES trajectory extraction failed. Available keys: {list(f.keys())}")
                    print(f"[plot] ES best_per_gen available: {t.best_per_gen is not None}")
                    if t.best_per_gen is not None:
                        print(f"[plot] ES best_per_gen shape: {t.best_per_gen.shape}")
                else:
                    print(f"[plot] {prefix} available keys: {list(np.load(npz_path, allow_pickle=True).keys()) if os.path.exists(npz_path) else 'file not found'}")
    return t


def _fit_unified_pca(all_points: list[np.ndarray], target_dim: int = 2, whiten: bool = True):
    """
    Fit PCA on all data combined to ensure consistent coordinate system.
    Returns PCA transformer that can be applied to individual arrays.
    """
    if not all_points:
        return None
    
    # Collect all points and flatten to 2D
    all_flat = []
    for points in all_points:
        if points is not None and points.size > 0:
            points_flat = points.reshape(-1, points.shape[-1])
            all_flat.append(points_flat)
    
    if not all_flat:
        return None
    
    # Combine all points
    combined_points = np.concatenate(all_flat, axis=0)
    original_dim = combined_points.shape[-1]
    
    if original_dim <= target_dim:
        return None
    
    print(f"[PCA] Fitting unified PCA on {original_dim}D data to project to {target_dim}D")
    print(f"[PCA] Whitening: {whiten}")
    print(f"[PCA] Total points for PCA fitting: {len(combined_points)}")
    
    # Fit PCA on combined data
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, whiten=whiten)
        pca.fit(combined_points)
        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"[PCA] Unified PCA explained variance ratio: {explained_variance_ratio}")
        print(f"[PCA] Cumulative explained variance: {np.sum(explained_variance_ratio):.3f}")
        return pca
    except ImportError:
        print("[PCA] sklearn not available, using manual PCA implementation")
        # Manual PCA implementation
        # Center the data
        mean = np.mean(combined_points, axis=0)
        centered = combined_points - mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Store transformation matrix with optional whitening
        scale = np.sqrt(np.maximum(eigenvalues[:target_dim], 1e-12)) if whiten else np.ones(target_dim)
        pca_transformer = {
            'mean': mean,
            'eigenvectors': eigenvectors[:, :target_dim],
            'scale': scale,
            'explained_variance_ratio': eigenvalues[:target_dim] / np.sum(eigenvalues)
        }
        
        print(f"[PCA] Manual PCA explained variance ratio: {pca_transformer['explained_variance_ratio']}")
        print(f"[PCA] Cumulative explained variance: {np.sum(pca_transformer['explained_variance_ratio']):.3f}")
        print(f"[PCA] Whitening scale factors: {pca_transformer['scale']}")
        
        return pca_transformer


def _apply_fitted_pca(points: np.ndarray, pca_transformer, target_dim: int = 2) -> np.ndarray:
    """
    Apply a fitted PCA transformer to new points.
    """
    if points is None or points.size == 0:
        return points
    
    original_shape = points.shape
    points_flat = points.reshape(-1, points.shape[-1])
    
    if pca_transformer is None:
        return points
    
    # Apply transformation
    if hasattr(pca_transformer, 'transform'):  # sklearn PCA
        points_transformed = pca_transformer.transform(points_flat)
    else:  # manual PCA
        mean = pca_transformer['mean']
        eigenvectors = pca_transformer['eigenvectors']
        centered = points_flat - mean
        points_transformed = centered @ eigenvectors
        # Apply whitening if scale factors are available
        if 'scale' in pca_transformer:
            points_transformed = points_transformed / pca_transformer['scale']
    
    # Reshape back to original shape but with target_dim as last dimension
    new_shape = list(original_shape[:-1]) + [target_dim]
    points_transformed = points_transformed.reshape(new_shape)
    
    return points_transformed


def _nice_bounds(xy: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    px = dx * pad if dx > 0 else 1.0
    py = dy * pad if dy > 0 else 1.0
    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def _splat_background(
    P: np.ndarray,
    V: np.ndarray,
    xlim,
    ylim,
    n: int = 400,
    enable_smoothing: bool = False,
    knn_k: int = 5,
    bandwidth_scale: float = 1.25,
    global_mix: float = 0.05,   # 0 = off, 0.02–0.1 is usually enough
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive Gaussian splatting with per-point bandwidths + optional global blend.
    Guarantees a continuous surface even when samples are irregularly spaced.
    """
    # grid
    xv = np.linspace(xlim[0], xlim[1], n)
    yv = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xv, yv)
    G = np.stack([XX.ravel(), YY.ravel()], axis=1)  # [M,2], M=n*n

    # ensure shapes
    P = P.reshape(-1, 2)
    V = V.reshape(-1)

    N = len(P)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    if N == 1:
        sigma = np.array([0.15 * span], dtype=float)
    else:
        # pairwise distances to set local bandwidths
        d2_pp = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2_pp, np.inf)
        k = int(np.clip(knn_k, 1, N - 1))
        d_knn = np.sqrt(np.partition(d2_pp, k, axis=1)[:, :k])  # [N,k]
        sigma = bandwidth_scale * np.median(d_knn, axis=1)       # [N]

        # clamp bandwidths to reasonable range
        min_sig = 0.02 * span
        max_sig = 0.30 * span
        sigma = np.clip(sigma, min_sig, max_sig)

    # weights to grid with per-point sigmas
    D2 = ((G[:, None, :] - P[None, :, :]) ** 2).sum(-1)          # [M,N]
    W = np.exp(-0.5 * D2 / (sigma[None, :] ** 2)) + 1e-12        # [M,N]

    num = W @ V                                                  # [M]
    den = W.sum(axis=1)                                          # [M]

    if global_mix > 0.0:
        # simple global prior: pulls Z toward the global mean where den is tiny
        Vmean = float(np.mean(V))
        lam = float(global_mix)                                  # acts like an extra, very wide kernel
        Z = (num + lam * Vmean) / (den + lam)
    else:
        Z = num / den

    Z = Z.reshape(n, n)

    if enable_smoothing and N > 1:
        try:
            from scipy.ndimage import gaussian_filter
            # final cosmetic blur, small relative to grid size
            Z = gaussian_filter(Z, sigma=max(0.6, 0.5 * n / 240))
        except ImportError:
            pass

    return XX, YY, Z


def _plot_traj(ax, pts: np.ndarray, color: str, label: str, arrow_every: int = 6, alpha: float = 1.0):
    print(f"[_plot_traj] Plotting {label}: {pts.shape}, color={color}, alpha={alpha}")
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8, alpha=alpha, label=label, zorder=5)
    
    # Add small markers for every step
    ax.scatter(pts[:, 0], pts[:, 1], s=15, color=color, alpha=alpha, zorder=4)
    
    # Add arrows between steps
    for i in range(0, len(pts) - 1, max(1, arrow_every)):
        ax.annotate("", xy=pts[i+1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=color, shrinkA=0, shrinkB=0))
    
    # Special markers for start and end points
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=70, marker="o", edgecolors="black", linewidths=0.7,
               color=color, zorder=6, alpha=alpha)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=70, marker="s", edgecolors="black", linewidths=0.7,
               color=color, zorder=6, alpha=alpha)
    print(f"[_plot_traj] Completed plotting {label}")


def plot_and_save(ga_npz_path: str, es_npz_path: str, out_dir: str, field_name: str = "loss", 
                  background_resolution: int = 400, background_smoothing: bool = False,
                  background_knn: int = 5, background_bandwidth_scale: float = 1.25, 
                  background_global_mix: float = 0.05, ga_steps: int = None, 
                  es_population: int = None, es_generations: int = None) -> tuple[Optional[str], Optional[str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except Exception as e:
        print(f"Plotting unavailable: {e}")
        return None, None

    ga = _load_trace(ga_npz_path, "ga_")
    es = _load_trace(es_npz_path, "es_")

    if ga.pts is None and es.pts is None and es.pop_pts is None:
        print("No 2D latent arrays found to plot.")
        return None, None

    # Apply PCA if needed and track original dimensions
    original_dims = []
    all_points = []
    
    # Collect all points for unified PCA fitting
    if ga.pts is not None:
        all_points.append(ga.pts)
        original_dims.append(ga.pts.shape[-1])
    if es.pts is not None:
        all_points.append(es.pts)
        original_dims.append(es.pts.shape[-1])
    if es.pop_pts is not None:
        all_points.append(es.pop_pts)
        original_dims.append(es.pop_pts.shape[-1])
    if es.best_per_gen is not None:
        all_points.append(es.best_per_gen)
        original_dims.append(es.best_per_gen.shape[-1])
    
    # Get the most common original dimension (or max if different)
    original_dim = max(original_dims) if original_dims else 2
    
    # Initialize PCA transformer early to avoid scope issues
    pca_transformer = None
    
    # Collect background data for loss landscape visualization
    # CRITICAL: We collect points and their corresponding loss values BEFORE PCA projection
    # This ensures that when we project to 2D, each 2D point retains its original loss value
    # The result is an accurate 2D loss landscape that preserves the high-dimensional structure
    bgP_original = []  # Store original high-dimensional points
    bgV_original = []  # Store corresponding loss values
    
    # GA path values
    if ga.pts is not None and ga.vals is not None:
        ga_pts_original = ga.pts.reshape(-1, ga.pts.shape[-1])  # (T, D) where D is original dim
        ga_vals_flat = ga.vals.reshape(-1)
        if len(ga_pts_original) == len(ga_vals_flat):
            print(f"[plot] GA background: original pts={ga_pts_original.shape}, vals={ga_vals_flat.shape}")
            bgP_original.append(ga_pts_original)
            bgV_original.append(ga_vals_flat)
        else:
            print(f"[plot] GA background mismatch: pts={ga_pts_original.shape}, vals={ga_vals_flat.shape}")
    else:
        print(f"[plot] GA background missing: pts={ga.pts is not None}, vals={ga.vals is not None}")
    
    # ES population values
    if es.pop_pts is not None and es.pop_vals is not None:
        es_pop_pts_original = es.pop_pts.reshape(-1, es.pop_pts.shape[-1])  # (N, D) where D is original dim
        es_pop_vals_flat = es.pop_vals.reshape(-1)
        if len(es_pop_pts_original) == len(es_pop_vals_flat):
            print(f"[plot] ES background: pts={es_pop_pts_original.shape}, vals={es_pop_vals_flat.shape}")
            bgP_original.append(es_pop_pts_original)
            bgV_original.append(es_pop_vals_flat)
        else:
            print(f"[plot] ES background mismatch: pts={es_pop_pts_original.shape}, vals={es_pop_vals_flat.shape}")
            print(f"[plot] ES mismatch details: pts length {len(es_pop_pts_original)}, vals length {es_pop_vals_flat.shape}")
    else:
        print(f"[plot] ES background missing: pts={es.pop_pts is not None}, vals={es.pop_vals is not None}")
    
    # Check if we have background data
    have_field = len(bgP_original) > 0
    print(f"[plot] Background field available: {have_field} (bgP_original={len(bgP_original)}, bgV_original={len(bgV_original)})")
    
    # Debug: show what we're working with
    if have_field:
        for i, (pts, vals) in enumerate(zip(bgP_original, bgV_original)):
            print(f"[plot] Background {i}: pts={pts.shape}, vals={vals.shape}")
            if len(pts) != len(vals):
                print(f"[plot] WARNING: Background {i} has mismatched lengths!")
    
    # After PCA, filter background data to only use points with consistent dimension
    if original_dim > 2 and pca_transformer is not None:
        # Filter background to only use points with the PCA dimension
        bgP_original_filtered = []
        bgV_original_filtered = []
        for pts, vals in zip(bgP_original, bgV_original):
            if pts.shape[-1] == original_dim:
                bgP_original_filtered.append(pts)
                bgV_original_filtered.append(vals)
        
        if bgP_original_filtered:
            print(f"[plot] Filtered background to {len(bgP_original_filtered)} arrays with dimension {original_dim}")
            bgP_original = bgP_original_filtered
            bgV_original = bgV_original_filtered
            have_field = len(bgP_original) > 0
        else:
            print(f"[plot] WARNING: No background data with dimension {original_dim}, disabling background")
            have_field = False

    # Apply unified PCA if needed
    if original_dim > 2:
        print(f"[plot] Original latent dimension: {original_dim}D, applying unified PCA to project to 2D")
        
        # Check if all points have the same dimension
        dims = [pts.shape[-1] for pts in all_points if pts is not None]
        if len(set(dims)) > 1:
            print(f"[plot] WARNING: Mixed dimensions detected: {dims}")
            print(f"[plot] This can happen when different runs have different latent dimensions")
            print(f"[plot] Using the most common dimension for PCA")
            # Use the most common dimension
            dim_counter = Counter(dims)
            most_common_dim = dim_counter.most_common(1)[0][0]
            print(f"[plot] Using dimension {most_common_dim} for PCA")
            # Filter to only use points with the most common dimension
            all_points_filtered = [pts for pts in all_points if pts is not None and pts.shape[-1] == most_common_dim]
            if all_points_filtered:
                all_points = all_points_filtered
                original_dim = most_common_dim
                print(f"[plot] Filtered to {len(all_points_filtered)} arrays with dimension {most_common_dim}")
            else:
                print(f"[plot] ERROR: No points with consistent dimension found")
                return None, None
        
        # Fit PCA on all data combined with whitening to prevent stretching
        # Whitening ensures PC1 and PC2 have similar variance, preventing the plot from looking like a pancake
        pca_transformer = _fit_unified_pca(all_points, target_dim=2, whiten=True)
        
        # Apply the same PCA transformation to all arrays
        if ga.pts is not None:
            print(f"[PCA] GA before projection: {ga.pts.shape}")
            ga.pts = _apply_fitted_pca(ga.pts, pca_transformer, target_dim=2)
            print(f"[PCA] GA after projection: {ga.pts.shape}")
        if es.pts is not None:
            print(f"[PCA] ES before projection: {es.pts.shape}")
            es.pts = _apply_fitted_pca(es.pts, pca_transformer, target_dim=2)
            print(f"[PCA] ES after projection: {es.pts.shape}")
        if es.pop_pts is not None:
            print(f"[PCA] ES pop before projection: {es.pop_pts.shape}")
            es.pop_pts = _apply_fitted_pca(es.pop_pts, pca_transformer, target_dim=2)
            print(f"[PCA] ES pop after projection: {es.pop_pts.shape}")
        if es.best_per_gen is not None:
            print(f"[PCA] ES best before projection: {es.best_per_gen.shape}")
            es.best_per_gen = _apply_fitted_pca(es.best_per_gen, pca_transformer, target_dim=2)
            print(f"[PCA] ES best after projection: {es.best_per_gen.shape}")
    else:
        print(f"[plot] Original latent dimension: {original_dim}D, no PCA needed")

    # unified bounds - flatten all arrays to 2D before concatenation
    pts_for_bounds = []
    for p in [ga.pts, es.pts, es.pop_pts, es.best_per_gen]:
        if p is not None:
            # Flatten to 2D: (..., 2) -> (N, 2)
            p_flat = p.reshape(-1, 2)
            pts_for_bounds.append(p_flat)
    
    if not pts_for_bounds:
        print("No valid points found for bounds calculation")
        return None, None
    
    XY = np.concatenate(pts_for_bounds, axis=0)
    
    # Fix the "pancake" look by squaring the view window
    # This ensures PC1 and PC2 have similar visual impact even when PCA variance ratios are imbalanced
    # Instead of letting PC1 dominate the span, we use the maximum range from both components
    cx, cy = XY[:, 0].mean(), XY[:, 1].mean()
    span = max(np.ptp(XY[:, 0]), np.ptp(XY[:, 1]))  # ptp = max - min (NumPy 2.0 compatible)
    pad = 0.10 * span
    xlim = (cx - span/2 - pad, cx + span/2 + pad)
    ylim = (cy - span/2 - pad, cy + span/2 + pad)
    
    print(f"[plot] View window: center=({cx:.3f}, {cy:.3f}), span={span:.3f}, xlim={xlim}, ylim={ylim}")

    # Background data has already been collected above, before PCA projection

    def orient(v: np.ndarray) -> np.ndarray:
        return -v if field_name.lower() == "score" else v

    # normalization across everything we will color
    all_for_norm = []
    if have_field:
        all_for_norm.append(orient(np.concatenate(bgV_original)))
    if ga.vals is not None:
        all_for_norm.append(orient(np.asarray(ga.vals)))
    if es.pop_vals is not None:
        all_for_norm.append(orient(np.asarray(es.pop_vals)))
    vmin, vmax = 0.0, 1.0
    if len(all_for_norm) > 0:
        vv = np.concatenate(all_for_norm)
        vmin, vmax = float(np.nanmin(vv)), float(np.nanmax(vv))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis"

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    title = f"Latent search: GA and ES (Z_dim = {original_dim})"
    ax.set_title(title)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_aspect("equal")  # With whitened PCA, this will look balanced
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # soft heatmap background by splatting losses if available
    if have_field:
        # KEY IMPROVEMENT: Create background using original high-dimensional points and their loss values
        # Project the points to 2D while preserving the loss landscape structure
        # This ensures each 2D point has its correct loss value, creating an accurate loss landscape
        P_original = np.concatenate(bgP_original, axis=0)  # (N, D) where D is original dim
        V_original = orient(np.concatenate(bgV_original, axis=0))  # (N,)
        
        if original_dim > 2 and pca_transformer is not None:
            print(f"[plot] Creating background using {len(P_original)} {original_dim}D points projected to 2D")
            # Project original high-dimensional points to 2D for background creation
            P_2d = _apply_fitted_pca(P_original, pca_transformer, target_dim=2)
            print(f"[plot] Background: {P_original.shape} -> {P_2d.shape}")
        else:
            print(f"[plot] Creating background using {len(P_original)} 2D points directly")
            P_2d = P_original  # Already 2D
        
        # Create the background heatmap using the projected points with their original loss values
        XX, YY, ZZ = _splat_background(
            P_2d, V_original, xlim, ylim, 
            n=background_resolution, 
            enable_smoothing=background_smoothing,
            knn_k=background_knn,
            bandwidth_scale=background_bandwidth_scale,
            global_mix=background_global_mix
        )
        
        # Display the background
        im = ax.pcolormesh(XX, YY, ZZ, shading="auto", cmap=cmap, norm=norm, zorder=0, alpha=0.7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_name)
    else:
        ax.set_facecolor("white")

    # ES population: show all samples in orange with full alpha + translucent generation circles
    if es.pop_pts is not None:
        # Flatten ES population points for plotting
        es_pop_pts_flat = es.pop_pts.reshape(-1, 2)
        
        # Plot ALL ES samples in orange with full alpha
        ax.scatter(es_pop_pts_flat[:, 0], es_pop_pts_flat[:, 1], s=20, alpha=1.0,
                   color="#ff7f0e", linewidths=0, zorder=1, label="ES population (all samples)")
        
        # Then add translucent circles to cluster samples from the same generation
        if es.gen_idx is not None:
            unique_gens = np.unique(es.gen_idx)
            # Complementary colors to viridis (blue-green-yellow): reds, oranges, purples, magentas
            generation_colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', 
                               '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#8e0152']
            
            for gen in unique_gens:
                mask = es.gen_idx == gen
                # Apply the same mask to flattened points
                gen_pts = es_pop_pts_flat[mask]
                color = generation_colors[gen % len(generation_colors)]
                
                # Calculate generation cluster center and radius
                gen_center = np.mean(gen_pts, axis=0)
                gen_radius = np.max(np.linalg.norm(gen_pts - gen_center, axis=1)) * 1.2  # 20% padding
                
                # Draw translucent circle for this generation
                circle = plt.Circle(gen_center, gen_radius, fill=True, linewidth=1, 
                                  edgecolor=color, facecolor=color, alpha=0.15)
                ax.add_patch(circle)
                
                # Add generation label at cluster center
                ax.text(gen_center[0], gen_center[1], f"Gen {gen}", 
                       ha='center', va='center', fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))

    # ES selected path (best per generation if present, otherwise es.pts)
    print(f"[plot] ES trajectory debug: best_per_gen={es.best_per_gen.shape if es.best_per_gen is not None else None}, es.pts={es.pts.shape if es.pts is not None else None}")
    es_sel = es.best_per_gen if es.best_per_gen is not None else es.pts
    print(f"[plot] ES selected for plotting: {es_sel.shape if es_sel is not None else None}")
    
    if es_sel is not None and len(es_sel) > 1:
        # Flatten ES selected path for plotting
        es_sel_flat = es_sel.reshape(-1, 2)
        print(f"[plot] Plotting ES trajectory: {es_sel_flat.shape}, range: x[{es_sel_flat[:, 0].min():.3f}, {es_sel_flat[:, 0].max():.3f}], y[{es_sel_flat[:, 1].min():.3f}, {es_sel_flat[:, 1].max():.3f}]")
        _plot_traj(ax, es_sel_flat, color="#ff7f0e", label="ES selected", alpha=1.0)
    else:
        print(f"[plot] ES trajectory plotting skipped: es_sel={es_sel is not None}, len={len(es_sel) if es_sel is not None else 0}")
        
        # Fallback: try to plot ES trajectory from best_per_gen if available
        if es.best_per_gen is not None and es.best_per_gen.size > 0:
            print(f"[plot] ES fallback: attempting to plot from best_per_gen")
            es_fallback = es.best_per_gen.reshape(-1, es.best_per_gen.shape[-1])
            if es_fallback.shape[0] > 1:
                print(f"[plot] ES fallback plotting: {es_fallback.shape}")
                _plot_traj(ax, es_fallback, color="#ff7f0e", label="ES selected (fallback)", alpha=1.0)
            else:
                print(f"[plot] ES fallback failed: not enough points ({es_fallback.shape[0]})")
        else:
            print(f"[plot] ES fallback: no best_per_gen available")

    # GA path
    if ga.pts is not None and len(ga.pts) > 1:
        # Flatten GA path for plotting
        ga_pts_flat = ga.pts.reshape(-1, 2)
        print(f"[plot] Plotting GA trajectory: {ga_pts_flat.shape}, range: x[{ga_pts_flat[:, 0].min():.3f}, {ga_pts_flat[:, 0].max():.3f}], y[{ga_pts_flat[:, 1].min():.3f}, {ga_pts_flat[:, 1].max():.3f}]")
        _plot_traj(ax, ga_pts_flat, color="#e91e63", label="GA path", alpha=1.0)
    else:
        print(f"[plot] GA plotting skipped: pts={ga.pts is not None}, len={len(ga.pts) if ga.pts is not None else 0}")

    # Create comprehensive legend with all elements
    legend_elements = []
    
    # ES population (all samples in orange)
    if es.pop_pts is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                                       markersize=10, alpha=1.0, label='ES population (all samples)'))
    
    # Generation clusters (general representation)
    if es.pop_pts is not None and es.gen_idx is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='#d73027', markerfacecolor='#d73027', 
                                       markersize=12, alpha=0.3, label='ES generation clusters'))
    
    # Trajectory paths
    legend_elements.append(plt.Line2D([0], [0], color='#ff7f0e', linewidth=2, label='ES selected path'))
    legend_elements.append(plt.Line2D([0], [0], color='#e91e63', linewidth=2, label='GA path'))
    
    # Start/End markers
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='w', 
                                     markersize=10, markeredgewidth=1, label='Start point'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='k', markerfacecolor='w', 
                                     markersize=10, markeredgewidth=1, label='End point'))
    
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "search_trajectories.png")
    svg = os.path.join(out_dir, "search_trajectories.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    
    # Generate loss curves plot
    loss_plot_path = plot_loss_curves(ga, es, out_dir, original_dim, 
                                      ga_steps=ga_steps, 
                                      es_population=es_population, 
                                      es_generations=es_generations)
    
    return png, loss_plot_path


def plot_loss_curves(ga: Trace, es: Trace, out_dir: str, original_dim: int = 2, 
                     ga_steps: int = None, es_population: int = None, es_generations: int = None) -> Optional[str]:
    """
    Generate a plot comparing loss curves for GA and ES methods with budget on x-axis.
    
    Budget calculation:
    - GA: 2 evaluations per step (forward + backward pass)
      Example: 5 steps → [2, 4, 6, 8, 10] budget points (no zero evaluation)
    - ES: Cumulative evaluations at each generation
      Example: 4 generations × 5 population → [5, 10, 15, 20] budget points (no zero evaluation)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Loss curve plotting unavailable: {e}")
        return None
    
    # Check if we have loss data for both methods
    has_ga_loss = ga.vals is not None and len(ga.vals) > 0
    has_es_loss = es.vals is not None and len(es.vals) > 0
    
    print(f"[loss] GA loss data: {has_ga_loss}, ES loss data: {has_es_loss}")
    if has_ga_loss:
        print(f"[loss] GA vals shape: {ga.vals.shape}, type: {type(ga.vals)}")
    if has_es_loss:
        print(f"[loss] ES vals shape: {es.vals.shape}, type: {type(es.vals)}")
    
    if not has_ga_loss and not has_es_loss:
        print("No loss data available for plotting loss curves.")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    title = f"Optimization Progress: Gradient Ascent vs Evolutionary Search (Z_dim = {original_dim})"
    ax.set_title(title)
    ax.set_xlabel("Budget (evaluations: starting from first evaluation)")
    ax.set_ylabel("Loss (lower is better)")
    ax.grid(True, alpha=0.3)
    
    # Plot GA loss curve with budget on x-axis
    if has_ga_loss:
        if ga_steps is not None:
            # GA budget: 2 evaluations per step (forward + backward pass)
            # Start from 2 (first step evaluation), not 0
            ga_budget = 2 * np.arange(1, len(ga.vals) + 1)
            print(f"[loss] GA budget calculation: {len(ga.vals)} steps → budget points: {ga_budget}")
            ax.plot(ga_budget, ga.vals, color="#e91e63", linewidth=2.5, marker='o', 
                    markersize=6, label=f"Gradient Ascent (2×{ga_steps} steps)", zorder=3)
        else:
            # Fallback to step indices if budget info not available
            ga_steps_indices = np.arange(len(ga.vals))
            ax.plot(ga_steps_indices, ga.vals, color="#e91e63", linewidth=2.5, marker='o', 
                    markersize=6, label="Gradient Ascent", zorder=3)
    
    # Plot ES loss curve with budget on x-axis
    if has_es_loss:
        if es_population is not None and es_generations is not None:
            # ES budget: cumulative evaluations at each generation
            # Start from population (first generation evaluation), not 0
            es_budget = np.arange(1, es_generations + 2) * es_population
            print(f"[loss] ES budget calculation: {es_generations} gens × {es_population} pop → budget points: {es_budget}")
            if len(es_budget) == len(es.vals):
                ax.plot(es_budget, es.vals, color="#ff7f0e", linewidth=2.5, marker='s', 
                        markersize=6, label=f"Evolutionary Search ({es_population}×{es_generations})", zorder=3)
            else:
                # Fallback if budget doesn't match values length
                print(f"[loss] ES budget mismatch: budget={es_budget}, values={len(es.vals)}")
                es_steps_indices = np.arange(len(es.vals))
                ax.plot(es_steps_indices, es.vals, color="#ff7f0e", linewidth=2.5, marker='s', 
                        markersize=6, label="Evolutionary Search", zorder=3)
        else:
            # Fallback to generation indices if budget info not available
            es_steps_indices = np.arange(len(es.vals))
            ax.plot(es_steps_indices, es.vals, color="#ff7f0e", linewidth=2.5, marker='s', 
                    markersize=6, label="Evolutionary Search", zorder=3)
    else:
        # Fallback: try to reconstruct ES curve from available data
        print(f"[loss] ES loss data missing - attempting to reconstruct from available data")
        if es.best_per_gen is not None:
            print(f"[loss] ES best_per_gen shape: {es.best_per_gen.shape}")
            # We have the best latents per generation, but no loss values
            # This indicates the loss data wasn't saved properly
            print(f"[loss] Warning: ES trajectory found but loss values are missing!")
            print(f"[loss] The ES curve cannot be plotted without loss data.")
            print(f"[loss] This suggests a data saving issue in evaluate_checkpoint.py")
    
    # Set y-axis to start from a reasonable lower bound
    if has_ga_loss or has_es_loss:
        all_vals = []
        if has_ga_loss:
            all_vals.extend(ga.vals)
        if has_es_loss:
            all_vals.extend(es.vals)
        
        y_min = min(all_vals)
        y_max = max(all_vals)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Add legend
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "loss_curves.png")
    svg = os.path.join(out_dir, "loss_curves.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    
    print(f"Saved loss curves plot to {png}")
    return png


def upload_to_wandb(project: str, entity: Optional[str], cfg: dict, ga_npz: str, es_npz: str, 
                    trajectory_plot: Optional[str], loss_plot: Optional[str], group_name: str = None) -> None:
    """
    Upload comprehensive metrics and artifacts to W&B for a single run.
    
    Logs the following metrics:
    
    GA (Gradient Ascent) Metrics:
    - ga_final_loss: Final loss value after optimization
    - ga_loss_progression: List of loss values at each step
    - ga_loss_improvement: Total improvement from start to end
    - ga_final_score: Final score value
    - ga_score_progression: List of score values at each step
    - ga_score_improvement: Total score improvement
    - ga_final_log_prob: Final log probability
    - ga_log_prob_progression: List of log probability values
    - ga_log_prob_improvement: Total log probability improvement
    
    ES (Evolutionary Search) Metrics:
    - es_final_loss: Final loss value after optimization
    - es_loss_progression: List of loss values at each generation
    - es_loss_improvement: Total improvement from start to end
    - es_best_loss: Best loss achieved across all generations
    - es_best_loss_progression: Best loss at each generation
    - es_population_size: Total number of individuals evaluated
    - es_min_loss: Minimum loss across population
    - es_max_loss: Maximum loss across population
    - es_mean_loss: Mean loss across population
    - es_std_loss: Standard deviation of loss across population
    - es_final_best_fitness: Final best fitness value
    
    Comparison Metrics:
    - ga_vs_es_final_loss_diff: Difference in final losses (GA - ES)
    - ga_vs_es_loss_improvement_diff: Difference in improvements (GA - ES)
    
    Run Summary Metrics:
    - run_budget: Total budget allocated
    - ga_steps: Number of GA steps
    - es_population: ES population size
    - es_generations: Number of ES generations
    - run_seed: Random seed used
    - run_index: Index of this run (for multiple runs)
    - n_samples: Total number of runs
    - dataset_length: Number of dataset samples evaluated
    - ga_return_code: GA execution return code
    - es_return_code: ES execution return code
    - ga_convergence: Whether GA converged or diverged
    - es_convergence: Whether ES converged or diverged
    
    Artifacts:
    - GA latent trajectories (NPZ)
    - ES latent trajectories (NPZ)
    - GA metrics CSV (downloadable)
    - ES metrics CSV (downloadable)
    - Run summary CSV (downloadable)
    - Trajectory plot (PNG)
    - Loss curves plot (PNG)
    """
    try:
        import wandb
    except Exception as e:
        print(f"wandb not available: {e}")
        return
    
    # Create run with optional group for n_samples > 1
    run_kwargs = {
        "project": project,
        "entity": entity,
        "name": f"latent-search-b{cfg.get('budget')}-s{cfg.get('run_idx', 0)}",
        "config": cfg
    }
    
    if group_name:
        run_kwargs["group"] = group_name
        print(f"[wandb] Creating grouped run: {group_name}")
    
    run = wandb.init(**run_kwargs)
    
    # Log artifacts (NPZ files with latent trajectories)
    if os.path.exists(ga_npz):
        ga_art = wandb.Artifact(name=f"ga_latents_b{cfg.get('budget')}_s{cfg.get('run_idx', 0)}", type="latent_trajectories")
        ga_art.add_file(ga_npz)
        run.log_artifact(ga_art)
    if os.path.exists(es_npz):
        es_art = wandb.Artifact(name=f"es_latents_b{cfg.get('budget')}_s{cfg.get('run_idx', 0)}", type="latent_trajectories")
        es_art.add_file(es_npz)
        run.log_artifact(es_art)
    
    # Log plots
    if trajectory_plot and os.path.exists(trajectory_plot):
        wandb.log({"trajectory_plot": wandb.Image(trajectory_plot)})
    if loss_plot and os.path.exists(loss_plot):
        wandb.log({"loss_curves_plot": wandb.Image(loss_plot)})
    
    # Log comprehensive metrics from both methods
    try:
        # Load GA metrics
        if os.path.exists(ga_npz):
            with np.load(ga_npz, allow_pickle=True) as f:
                ga_metrics = {}
                if 'ga_losses' in f:
                    ga_losses = np.array(f['ga_losses'])
                    ga_metrics['ga_final_loss'] = float(ga_losses[-1]) if len(ga_losses) > 0 else None
                    ga_metrics['ga_loss_progression'] = ga_losses.tolist()
                    ga_metrics['ga_loss_improvement'] = float(ga_losses[0] - ga_losses[-1]) if len(ga_losses) > 0 else None
                
                if 'ga_scores' in f:
                    ga_scores = np.array(f['ga_scores'])
                    ga_metrics['ga_final_score'] = float(ga_scores[-1]) if len(ga_scores) > 0 else None
                    ga_metrics['ga_score_progression'] = ga_scores.tolist()
                    ga_metrics['ga_score_improvement'] = float(ga_scores[-1] - ga_scores[0]) if len(ga_scores) > 0 else None
                
                if 'ga_log_probs' in f:
                    ga_log_probs = np.array(f['ga_log_probs'])
                    ga_metrics['ga_final_log_prob'] = float(ga_log_probs[-1]) if len(ga_log_probs) > 0 else None
                    ga_metrics['ga_log_prob_progression'] = ga_log_probs.tolist()
                    ga_metrics['ga_log_prob_improvement'] = float(ga_log_probs[-1] - ga_log_probs[0]) if len(ga_log_probs) > 0 else None
                
                # Log GA metrics
                for key, value in ga_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Create GA metrics CSV for download
                ga_csv_path = f"ga_metrics_run_{cfg.get('run_idx', 0)}.csv"
                ga_metrics_df = pd.DataFrame({
                    'step': range(len(ga_losses)) if 'ga_losses' in f else [],
                    'loss': ga_losses if 'ga_losses' in f else [],
                    'score': ga_scores if 'ga_scores' in f else [],
                    'log_prob': ga_log_probs if 'ga_log_probs' in f else []
                })
                ga_metrics_df.to_csv(ga_csv_path, index=False)
                
                # Upload GA metrics CSV
                ga_csv_art = wandb.Artifact(name=f"ga_metrics_b{cfg.get('budget')}_s{cfg.get('run_idx', 0)}", type="metrics")
                ga_csv_art.add_file(ga_csv_path)
                run.log_artifact(ga_csv_art)
                
                # Clean up temporary CSV
                os.remove(ga_csv_path)
                
        # Load ES metrics
        if os.path.exists(es_npz):
            with np.load(es_npz, allow_pickle=True) as f:
                es_metrics = {}
                
                # Generation-level metrics
                if 'es_generation_losses' in f:
                    gen_losses = np.array(f['es_generation_losses'])
                    es_metrics['es_final_loss'] = float(gen_losses[-1]) if len(gen_losses) > 0 else None
                    es_metrics['es_loss_progression'] = gen_losses.tolist()
                    es_metrics['es_loss_improvement'] = float(gen_losses[0] - gen_losses[-1]) if len(gen_losses) > 0 else None
                    es_metrics['es_best_loss'] = float(np.min(gen_losses)) if len(gen_losses) > 0 else None
                
                if 'es_best_losses_per_generation' in f:
                    best_losses = np.array(f['es_best_losses_per_generation'])
                    es_metrics['es_best_loss_progression'] = best_losses.tolist()
                
                # Population-level metrics
                if 'es_all_losses' in f:
                    all_losses = np.array(f['es_all_losses'])
                    es_metrics['es_population_size'] = len(all_losses)
                    es_metrics['es_min_loss'] = float(np.min(all_losses)) if len(all_losses) > 0 else None
                    es_metrics['es_max_loss'] = float(np.max(all_losses)) if len(all_losses) > 0 else None
                    es_metrics['es_mean_loss'] = float(np.mean(all_losses)) if len(all_losses) > 0 else None
                    es_metrics['es_std_loss'] = float(np.std(all_losses)) if len(all_losses) > 0 else None
                
                if 'es_final_best_fitness' in f:
                    final_fitness = np.array(f['es_final_best_fitness'])
                    es_metrics['es_final_best_fitness'] = float(final_fitness) if final_fitness.size > 0 else None
                
                # Log ES metrics
                for key, value in es_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Create ES metrics CSV for download
                es_csv_path = f"es_metrics_run_{cfg.get('run_idx', 0)}.csv"
                
                # Prepare ES metrics data
                es_data = {}
                if 'es_generation_losses' in f:
                    es_data['generation'] = range(len(gen_losses))
                    es_data['generation_loss'] = gen_losses
                if 'es_best_losses_per_generation' in f:
                    es_data['best_loss'] = best_losses
                if 'es_all_losses' in f and 'es_generation_idx' in f:
                    es_data['population_generation'] = np.array(f['es_generation_idx'])
                    es_data['population_loss'] = all_losses
                
                es_metrics_df = pd.DataFrame(es_data)
                es_metrics_df.to_csv(es_csv_path, index=False)
                
                # Upload ES metrics CSV
                es_csv_art = wandb.Artifact(name=f"es_metrics_b{cfg.get('budget')}_s{cfg.get('run_idx', 0)}", type="metrics")
                es_csv_art.add_file(es_csv_path)
                run.log_artifact(es_csv_art)
                
                # Clean up temporary CSV
                os.remove(es_csv_path)
                
                        # Log comparison metrics
                comparison_metrics = {}
                if 'ga_final_loss' in ga_metrics and 'es_final_loss' in es_metrics:
                    comparison_metrics['ga_vs_es_final_loss_diff'] = ga_metrics['ga_final_loss'] - es_metrics['es_final_loss']
                    comparison_metrics['ga_vs_es_loss_improvement_diff'] = ga_metrics['ga_loss_improvement'] - es_metrics['es_loss_improvement']
                
                # Log comparison metrics
                for key, value in comparison_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Log summary statistics for the run
                summary_metrics = {
                    'run_budget': cfg.get('budget'),
                    'ga_steps': cfg.get('ga_steps'),
                    'es_population': cfg.get('es_population'),
                    'es_generations': cfg.get('es_generations'),
                    'run_seed': cfg.get('dataset_seed'),
                    'run_index': cfg.get('run_idx', 0),
                    'n_samples': cfg.get('n_samples', 1),
                    'dataset_length': cfg.get('dataset_length'),
                    'ga_return_code': cfg.get('ga_return_code'),
                    'es_return_code': cfg.get('es_return_code'),
                }
                
                # Add method-specific summary metrics
                if 'ga_final_loss' in ga_metrics:
                    summary_metrics['ga_convergence'] = 'converged' if ga_metrics['ga_loss_improvement'] > 0 else 'diverged'
                if 'es_final_loss' in es_metrics:
                    summary_metrics['es_convergence'] = 'converged' if es_metrics['es_loss_improvement'] > 0 else 'diverged'
                
                # Log summary metrics
                for key, value in summary_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Create comprehensive summary CSV for the entire run
                summary_csv_path = f"run_summary_{cfg.get('run_idx', 0)}.csv"
                
                # Combine all metrics into one comprehensive summary
                summary_data = {
                    'metric': [],
                    'value': [],
                    'method': [],
                    'description': []
                }
                
                # Add GA metrics
                for key, value in ga_metrics.items():
                    if value is not None:
                        summary_data['metric'].append(key)
                        summary_data['value'].append(value)
                        summary_data['method'].append('GA')
                        summary_data['description'].append('Gradient Ascent metric')
                
                # Add ES metrics
                for key, value in es_metrics.items():
                    if value is not None:
                        summary_data['metric'].append(key)
                        summary_data['value'].append(value)
                        summary_data['method'].append('ES')
                        summary_data['description'].append('Evolutionary Search metric')
                
                # Add comparison metrics
                for key, value in comparison_metrics.items():
                    if value is not None:
                        summary_data['metric'].append(key)
                        summary_data['value'].append(value)
                        summary_data['method'].append('Comparison')
                        summary_data['description'].append('GA vs ES comparison')
                
                # Add summary metrics
                for key, value in summary_metrics.items():
                    if value is not None:
                        summary_data['metric'].append(key)
                        summary_data['value'].append(value)
                        summary_data['method'].append('Run')
                        summary_data['description'].append('Run configuration and status')
                
                # Create and save summary CSV
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_csv_path, index=False)
                
                # Upload comprehensive summary CSV
                summary_csv_art = wandb.Artifact(name=f"run_summary_b{cfg.get('budget')}_s{cfg.get('run_idx', 0)}", type="summary")
                summary_csv_art.add_file(summary_csv_path)
                run.log_artifact(summary_csv_art)
                
                # Clean up temporary CSV
                os.remove(summary_csv_path)
                
    except Exception as e:
        print(f"Warning: Failed to extract metrics from NPZ files: {e}")
    
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store and plot latent search trajectories (GA & ES). "
        "Both methods start from the same mean latent for fair comparison. "
        "Automatically applies PCA to reduce latent dimensions > 2 to 2D for visualization. "
        "Use --ga_steps, --es_population, --es_generations to override automatic budget-based calculations. "
        "Use --n_samples to run multiple experiments with different seeds for statistical analysis. "
        "Use --dataset_length to control the number of samples evaluated from the dataset."
    )
    parser.add_argument("--wandb_artifact_path", required=True, type=str)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--ga_lr", type=float, default=0.5)
    parser.add_argument("--ga_steps", type=int, default=None, help="Number of GA steps (overrides budget/2 calculation)")
    parser.add_argument("--es_mutation_std", type=float, default=0.5)
    parser.add_argument("--es_population", type=int, default=None, help="ES population size (overrides sqrt(budget) calculation)")
    parser.add_argument("--es_generations", type=int, default=None, help="ES number of generations (overrides budget/pop calculation)")
    parser.add_argument("--use_subspace_mutation", action="store_true")
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--ga_step_length", type=float, default=0.5)
    parser.add_argument("--trust_region_radius", type=float, default=None)
    parser.add_argument("--track_progress", action="store_true", help="Enable progress tracking for both GA and ES")
    parser.add_argument("--background_resolution", type=int, default=400, help="Base resolution for background heatmap (higher = smoother)")
    parser.add_argument("--background_smoothing", action="store_true", help="Enable additional Gaussian smoothing for small-scale searches")
    parser.add_argument("--background_knn", type=int, default=5, help="k-NN parameter for adaptive bandwidth (3-7 recommended)")
    parser.add_argument("--background_bandwidth_scale", type=float, default=1.25, help="Bandwidth scaling factor (bigger = softer, more overlap)")
    parser.add_argument("--background_global_mix", type=float, default=0.05, help="Global mixing strength (0.02-0.1 recommended, 0 to disable)")
    parser.add_argument("--out_dir", type=str, default="results/latent_traces")
    parser.add_argument("--wandb_project", type=str, default="latent-search-analysis")
    parser.add_argument("--wandb_entity", type=str, default=None)
    # Data source
    parser.add_argument("--json_challenges", type=str, default=None)
    parser.add_argument("--json_solutions", type=str, default=None)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--dataset_length", type=int, default=None, help="Number of samples in the dataset to evaluate")
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument("--dataset_use_hf", type=str, default="true")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1, help="Number of times to run the script with different random seeds (for statistical analysis)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src_args = build_dataset_args(args)

    # Gradient Ascent config
    ga_steps = args.ga_steps if args.ga_steps is not None else int(math.ceil(args.budget / 2))
    print(f"🔧 GA config: {ga_steps} steps (lr={args.ga_lr})")
    print(f"   🎯 GA starts from mean latent")
    ga_out = os.path.join(args.out_dir, "ga_latents.npz")
    ga_cmd = [
        sys.executable, "src/evaluate_checkpoint.py",
        "-w", args.wandb_artifact_path,
        "-i", "gradient_ascent",
        "--num-steps", str(ga_steps),
        "--lr", str(args.ga_lr),
        "--no-wandb-run", "true",
        "--store-latents", ga_out,
    ]
    if args.track_progress:
        ga_cmd.append("--track-progress")
    ga_cmd += src_args
    print("Running:", " ".join(ga_cmd))
    ga_rc = subprocess.run(ga_cmd, check=False).returncode
    print(f"GA return code: {ga_rc}")

    # Evolutionary Search config
    pop = args.es_population if args.es_population is not None else max(3, min(32, int(round(math.sqrt(args.budget)))))
    gens = args.es_generations if args.es_generations is not None else max(1, int(math.ceil(args.budget / pop)))
    print(f"🧬 ES config: population={pop}, generations={gens} (mutation_std={args.es_mutation_std})")
    print(f"   🎯 ES starts from mean latent (same as GA)")
    es_out = os.path.join(args.out_dir, "es_latents.npz")
    es_cmd = [
        sys.executable, "src/evaluate_checkpoint.py",
        "-w", args.wandb_artifact_path,
        "-i", "evolutionary_search",
        "--population-size", str(pop),
        "--num-generations", str(gens),
        "--mutation-std", str(args.es_mutation_std),
        "--no-wandb-run", "true",
        "--store-latents", es_out,
    ]
    if args.track_progress:
        es_cmd.append("--track-progress")
    es_cmd += src_args
    if args.use_subspace_mutation:
        es_cmd += ["--use-subspace-mutation", "--subspace-dim", str(args.subspace_dim), "--ga-step-length", str(args.ga_step_length)]
        if args.trust_region_radius is not None:
            es_cmd += ["--trust-region-radius", str(args.trust_region_radius)]
    print("Running:", " ".join(es_cmd))
    es_rc = subprocess.run(es_cmd, check=False).returncode
    print(f"ES return code: {es_rc}")

    # Handle multiple runs when n_samples > 1
    # NOTE: n_samples controls how many times to run the script with different seeds
    # dataset_length controls how many samples from the dataset to evaluate in each run
    if args.n_samples > 1:
        print(f"🧪 Running {args.n_samples} experiments with different seeds...")
        
        # Create a group name for W&B
        group_name = f"latent-search-b{args.budget}-n{args.n_samples}-{int(time.time())}"
        
        for run_idx in range(args.n_samples):
            seed = args.dataset_seed + run_idx
            print(f"\n🔬 Run {run_idx + 1}/{args.n_samples} with seed {seed}")
            
            # Build the correct arguments manually to avoid parsing issues
            # Each run uses the same dataset_length but different seeds
            run_src_args = [
                "-d", args.dataset_folder,
                "--dataset-length", str(args.dataset_length) if args.dataset_length else "1",
                "--dataset-batch-size", str(args.dataset_batch_size) if args.dataset_batch_size else "1",
                "--dataset-use-hf", args.dataset_use_hf,
                "--dataset-seed", str(seed)
            ]
            
            # Debug: show the final args
            print(f"[run] Final args: {run_src_args}")
            
            # Run GA for this seed
            ga_cmd = [
                sys.executable, "src/evaluate_checkpoint.py",
                "-w", args.wandb_artifact_path,
                "-i", "gradient_ascent",
                "--num-steps", str(ga_steps),
                "--lr", str(args.ga_lr),
                "--no-wandb-run", "true",
                "--store-latents", ga_out,
            ]
            if args.track_progress:
                ga_cmd.append("--track-progress")
            ga_cmd += run_src_args
            print("Running GA:", " ".join(ga_cmd))
            ga_rc = subprocess.run(ga_cmd, check=False).returncode
            print(f"GA return code: {ga_rc}")
            
            # Run ES for this seed
            es_cmd = [
                sys.executable, "src/evaluate_checkpoint.py",
                "-w", args.wandb_artifact_path,
                "-i", "evolutionary_search",
                "--population-size", str(pop),
                "--num-generations", str(gens),
                "--mutation-std", str(args.es_mutation_std),
                "--no-wandb-run", "true",
                "--store-latents", es_out,
            ]
            if args.track_progress:
                es_cmd.append("--track-progress")
            es_cmd += run_src_args
            if args.use_subspace_mutation:
                es_cmd += ["--use-subspace-mutation", "--subspace-dim", str(args.subspace_dim), "--ga-step-length", str(args.ga_step_length)]
                if args.trust_region_radius is not None:
                    es_cmd += ["--trust-region-radius", str(args.trust_region_radius)]
            print("Running ES:", " ".join(es_cmd))
            es_rc = subprocess.run(es_cmd, check=False).returncode
            print(f"ES return code: {es_rc}")
            
            # Plot for this run
            trajectory_plot, loss_plot = plot_and_save(ga_out, es_out, args.out_dir, 
                                                      background_resolution=args.background_resolution,
                                                      background_smoothing=args.background_smoothing,
                                                      background_knn=args.background_knn,
                                                      background_bandwidth_scale=args.background_bandwidth_scale,
                                                      background_global_mix=args.background_global_mix,
                                                      ga_steps=ga_steps, es_population=pop, es_generations=gens)
            if trajectory_plot:
                print(f"Saved trajectory plot to {trajectory_plot}")
            if loss_plot:
                print(f"Saved loss curves plot to {loss_plot}")
            
            # Upload to W&B with group name
            cfg = {
                "artifact_path": args.wandb_artifact_path,
                "budget": args.budget,
                "ga_steps": ga_steps,
                "ga_lr": args.ga_lr,
                "es_population": pop,
                "es_generations": gens,
                "es_mutation_std": args.es_mutation_std,
                "use_subspace_mutation": args.use_subspace_mutation,
                "subspace_dim": args.subspace_dim if args.use_subspace_mutation else None,
                "ga_step_length": args.ga_step_length if args.use_subspace_mutation else None,
                "trust_region_radius": args.trust_region_radius,
                "track_progress": args.track_progress,
                "background_resolution": args.background_resolution,
                "background_smoothing": args.background_smoothing,
                "background_knn": args.background_knn,
                "background_bandwidth_scale": args.background_bandwidth_scale,
                "background_global_mix": args.background_global_mix,
                "ga_return_code": ga_rc,
                "es_return_code": es_rc,
                "run_idx": run_idx,
                "dataset_seed": seed,
                "n_samples": args.n_samples,
                "dataset_length": args.dataset_length,
            }
            try:
                upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot, group_name)
            except Exception as e:
                print(f"Failed to upload to wandb: {e}")
        
        print(f"\n✅ Completed {args.n_samples} runs in group: {group_name}")
        
    else:
        # Single run (original behavior)
        # Plot (with automatic PCA if latent dimension > 2)
        trajectory_plot, loss_plot = plot_and_save(ga_out, es_out, args.out_dir, 
                                                  background_resolution=args.background_resolution,
                                                  background_smoothing=args.background_smoothing,
                                                  background_knn=args.background_knn,
                                                  background_bandwidth_scale=args.background_bandwidth_scale,
                                                  background_global_mix=args.background_global_mix,
                                                  ga_steps=ga_steps, es_population=pop, es_generations=gens)
        if trajectory_plot:
            print(f"Saved trajectory plot to {trajectory_plot}")
        if loss_plot:
            print(f"Saved loss curves plot to {loss_plot}")

        # Upload to W&B
        cfg = {
            "artifact_path": args.wandb_artifact_path,
            "budget": args.budget,
            "ga_steps": ga_steps,
            "ga_lr": args.ga_lr,
            "es_population": pop,
            "es_generations": gens,
            "es_mutation_std": args.es_mutation_std,
            "use_subspace_mutation": args.use_subspace_mutation,
            "subspace_dim": args.subspace_dim if args.use_subspace_mutation else None,
            "ga_step_length": args.ga_step_length if args.use_subspace_mutation else None,
            "trust_region_radius": args.trust_region_radius,
            "track_progress": args.track_progress,
            "background_resolution": args.background_resolution,
            "background_smoothing": args.background_smoothing,
            "background_knn": args.background_knn,
            "background_bandwidth_scale": args.background_bandwidth_scale,
            "background_global_mix": args.background_global_mix,
            "ga_return_code": ga_rc,
            "es_return_code": es_rc,
        }
        try:
            upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot)
        except Exception as e:
            print(f"Failed to upload to wandb: {e}")


if __name__ == "__main__":
    main()


