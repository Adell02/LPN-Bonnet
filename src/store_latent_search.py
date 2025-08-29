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
    --es_mutation_std 0.3 \
    --es_mutation_std 0.3 \
    --mutation_decay 0.9 \
    --elite_size 10

# With subspace mutation and progress tracking:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --use_subspace_mutation \
    --subspace_dim 16 \
    --ga_step_length 0.5 \
    --track_progress

# With custom run name:
python src/store_latent_search.py \
    --wandb_artifact_path "entity/project/artifact:v0" \
    --budget 100 \
    --run_name "pattern4_experiment" \
    --n_samples 5

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


def safe_array_to_scalar(arr: np.ndarray, default=None):
    """
    Safely convert a numpy array to a scalar value.
    Handles arrays of any size by taking the first element or using a default.
    """
    if arr is None:
        return default
    
    arr = np.asarray(arr)
    if arr.size == 0:
        return default
    elif arr.size == 1:
        return float(arr)
    else:
        # Array has multiple elements, take the first one
        return float(arr.flat[0])


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
    
    # Prefer per-individual losses for exact pairing; then robust fallbacks
    if pts is not None:
        N = int(pts.reshape(-1, pts.shape[-1]).shape[0])
    else:
        N = None

    # Helper to try adopt array if it matches N
    def _try_use(arr: Optional[np.ndarray], label: str) -> Optional[np.ndarray]:
        if arr is None or N is None:
            return None
        a = np.array(arr).reshape(-1)
        if a.size == N:
            print(f"[plot] Using {prefix}{label}: {a.shape}")
            return a
        return None

    # 1) Exact per-individual losses
    if f"{prefix}all_losses" in npz:
        vals = _try_use(npz[f"{prefix}all_losses"], "all_losses")
    if vals is None and f"{prefix}losses_per_generation" in npz:
        lp = np.array(npz[f"{prefix}losses_per_generation"])  # (G,P) or similar
        flat = lp.reshape(-1)
        if N is not None and flat.size == N:
            vals = flat
            print(f"[plot] Using {prefix}losses_per_generation: {lp.shape} -> {vals.shape}")

    # 2) Map per-generation losses to individuals via generation_idx
    if vals is None and gens is not None:
        # generation losses may be stored under multiple names
        gen_losses = None
        for k in (f"{prefix}generation_losses", f"{prefix}best_losses_per_generation"):
            if k in npz:
                gen_losses = np.array(npz[k]).reshape(-1)
                break
        if gen_losses is None and f"{prefix}all_losses" in npz:
            # Sometimes all_losses is per-generation, try that too
            tmp = np.array(npz[f"{prefix}all_losses"]).reshape(-1)
            if N is not None and gens.size > 0 and tmp.size == len(np.unique(gens)):
                gen_losses = tmp
        if gen_losses is not None and gens.size > 0:
            # Safe indexing by generation id
            max_gen = int(np.max(gens))
            if gen_losses.size >= max_gen + 1:
                vals = gen_losses[gens]
                print(f"[plot] Using {prefix}generation_losses expanded by generation_idx: {gen_losses.shape} -> {vals.shape}")

    # 3) As a last resort, if we have scores use them (orientation handled later)
    if vals is None and f"{prefix}all_scores" in npz:
        if N is not None:
            scores = np.array(npz[f"{prefix}all_scores"]).reshape(-1)
            if scores.size == N:
                vals = scores
                print(f"[plot] Using {prefix}all_scores: {vals.shape}")
    
    # Ensure every point has a value; if still mismatched, align robustly
    if prefix == "es_" and pts is not None:
        n_pts = int(pts.reshape(-1, pts.shape[-1]).shape[0])
        if vals is None and gens is not None:
            # Assign a neutral per-generation value if nothing else available
            # Use 0.0 as placeholder to still contribute density; normalization will span GA values too
            unique_gens = np.unique(gens)
            gen_vals = {g: 0.0 for g in unique_gens}
            vals = np.array([gen_vals[g] for g in gens], dtype=float)
            print(f"[plot] ES fallback: assigning neutral per-generation values (0.0) to ensure coverage: {vals.shape}")
        if vals is not None:
            vals = np.array(vals).reshape(-1)
            if vals.size != n_pts:
                print(f"[plot] ES length mismatch after extraction: vals={vals.size}, pts={n_pts}. Will tile/truncate to match.")
                if vals.size == 1:
                    vals = np.repeat(vals, n_pts)
                elif vals.size < n_pts:
                    reps = int(np.ceil(n_pts / vals.size))
                    vals = np.tile(vals, reps)[:n_pts]
                else:
                    vals = vals[:n_pts]
                print(f"[plot] ES values aligned to points: {vals.shape}")
    
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
    global_mix: float = 0.05,   # Restored to 0.05 for smooth gaussian interpolation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive Gaussian splatting with per-point bandwidths.
    Provides smooth interpolation between loss points for beautiful viridis tonality.
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

    # Restore gaussian interpolation for smooth loss landscape
    # Use global mixing for smooth transitions between loss points
    Z = num / den
    
    # Apply global mixing for smooth interpolation
    if global_mix > 0:
        global_mean = np.nanmean(V)
        Z = (1 - global_mix) * Z + global_mix * global_mean
    
    # Reshape to grid
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
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=7.0, alpha=alpha, label=label, zorder=5)
    
    # Add small markers for every step
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color=color, alpha=alpha, zorder=4)
    
    # Add arrows between steps
    for i in range(0, len(pts) - 1, max(1, arrow_every)):
        ax.annotate("", xy=pts[i+1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", lw=6.0, color=color, shrinkA=0, shrinkB=0))
    
    # Special markers for start and end points
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=200, marker="o", edgecolors="black", linewidths=2.0,
               color=color, zorder=6, alpha=alpha)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=200, marker="s", edgecolors="black", linewidths=2.0,
               color=color, zorder=6, alpha=alpha)
    print(f"[_plot_traj] Completed plotting {label}")


def plot_and_save(ga_npz_path: str, es_npz_path: str, out_dir: str, field_name: str = "loss", 
                  background_resolution: int = 400, background_smoothing: bool = False,
                  background_knn: int = 5, background_bandwidth_scale: float = 1.25, 
                  background_global_mix: float = 0.05, ga_steps: int = None, 
                  es_population: int = None, es_generations: int = None, dataset_length: int = None) -> tuple[Optional[str], Optional[str], int]:
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
    # 
    # KEY IMPROVEMENT: ES background now uses ALL population losses, not just winners
    # This creates a comprehensive loss landscape showing the full exploration of the search space
    # including poor-performing regions that help understand the optimization landscape
    bgP_original = []  # Store original high-dimensional points
    bgV_original = []  # Store corresponding loss values
    
    # GA path values
    if ga.pts is not None and ga.vals is not None:
        ga_pts_original = ga.pts.reshape(-1, ga.pts.shape[-1])  # (T, D) where D is original dim
        ga_vals_flat = ga.vals.reshape(-1)
        if len(ga_pts_original) == len(ga_vals_flat):
            print(f"[plot] GA background: original pts={ga_pts_original.shape}, vals={ga_vals_flat.shape}")
            print(f"[plot] GA values range: [{ga_vals_flat.min():.4f}, {ga_vals_flat.max():.4f}]")
            bgP_original.append(ga_pts_original)
            bgV_original.append(ga_vals_flat)
        else:
            print(f"[plot] GA background mismatch: pts={ga_pts_original.shape}, vals={ga_vals_flat.shape}")
    else:
        print(f"[plot] GA background missing: pts={ga.pts is not None}, vals={ga.vals is not None}")
    
    # ES population values - CRITICAL: Use ALL population losses for comprehensive loss landscape
    # This ensures the background shows the full exploration of the search space, not just winners
    # 
    # BEFORE: Only used trajectory values (best per generation) - missed poor regions
    # AFTER: Uses full population losses - shows complete search space exploration
    # This reveals valleys, plateaus, and local minima that help understand optimization difficulty
    if es.pop_pts is not None:
        es_pop_pts_original = es.pop_pts.reshape(-1, es.pop_pts.shape[-1])  # (N, D) where D is original dim
        
        # PREFER: Use full population losses (es.pop_vals) for comprehensive landscape
        if es.pop_vals is not None:
            es_pop_vals_flat = es.pop_vals.reshape(-1)
            if len(es_pop_vals_flat) == len(es_pop_pts_original):
                print(f"[plot] ES background: pts={es_pop_pts_original.shape}, vals={es_pop_vals_flat.shape}")
                print(f"[plot] ES values range: [{es_pop_vals_flat.min():.4f}, {es_pop_vals_flat.max():.4f}]")
                print(f"[plot] ES using FULL population losses for comprehensive landscape")
                bgP_original.append(es_pop_pts_original)
                bgV_original.append(es_pop_vals_flat)
            else:
                print(f"[plot] ES population values length mismatch: pts={len(es_pop_pts_original)}, vals={len(es_pop_vals_flat)}")
                print(f"[plot] ES fallback: attempting to reconstruct from trajectory values")
                # Fallback to trajectory-based reconstruction
                if es.vals is not None:
                    es_trajectory_vals = es.vals.reshape(-1)
                    num_generations = len(es_trajectory_vals)
                    population_per_gen = len(es_pop_pts_original) // num_generations
                    if num_generations > 0 and population_per_gen > 0:
                        es_pop_vals_reconstructed = np.repeat(es_trajectory_vals, population_per_gen)
                        if len(es_pop_vals_reconstructed) == len(es_pop_pts_original):
                            print(f"[plot] ES reconstruction successful: {num_generations} gens × {population_per_gen} pop")
                            bgP_original.append(es_pop_pts_original)
                            bgV_original.append(es_pop_vals_reconstructed)
                        else:
                            print(f"[plot] ES reconstruction failed: expected {len(es_pop_pts_original)}, got {len(es_pop_vals_reconstructed)}")
                    else:
                        print(f"[plot] ES reconstruction failed: invalid generation/population counts")
                else:
                    print(f"[plot] ES trajectory values missing, cannot reconstruct population losses")
        else:
            print(f"[plot] ES population values missing, attempting trajectory-based reconstruction")
            # Fallback: try to reconstruct from trajectory values
            if es.vals is not None:
                es_trajectory_vals = es.vals.reshape(-1)
                num_generations = len(es_trajectory_vals)
                population_per_gen = len(es_pop_pts_original) // num_generations
                if num_generations > 0 and population_per_gen > 0:
                    es_pop_vals_reconstructed = np.repeat(es_trajectory_vals, population_per_gen)
                    if len(es_pop_vals_reconstructed) == len(es_pop_pts_original):
                        print(f"[plot] ES reconstruction successful: {num_generations} gens × {population_per_gen} pop")
                        bgP_original.append(es_pop_pts_original)
                        bgV_original.append(es_pop_vals_reconstructed)
                    else:
                        print(f"[plot] ES reconstruction failed: expected {len(es_pop_pts_original)}, got {len(es_pop_vals_reconstructed)}")
                else:
                    print(f"[plot] ES reconstruction failed: invalid generation/population counts")
            else:
                print(f"[plot] ES trajectory values missing, cannot create background")
    else:
        print(f"[plot] ES background missing: pts={es.pop_pts is not None}")
    
    # Check for value consistency between GA and ES
    # Note: GA uses trajectory losses, ES uses full population losses for comprehensive landscape
    if ga.vals is not None and es.pop_vals is not None:
        ga_range = (ga.vals.min(), ga.vals.max())
        es_pop_range = (es.pop_vals.min(), es.pop_vals.max())
        print(f"[plot] Value consistency check:")
        print(f"  GA trajectory range: [{ga_range[0]:.4f}, {ga_range[1]:.4f}]")
        print(f"  ES population range: [{es_pop_range[0]:.4f}, {es_pop_range[1]:.4f}]")
        print(f"  Note: ES population range may be wider than trajectory range (showing full exploration)")
        
        # Check if ES trajectory values are available for comparison
        if es.vals is not None:
            es_traj_range = (es.vals.min(), es.vals.max())
            print(f"  ES trajectory range: [{es_traj_range[0]:.4f}, {es_traj_range[1]:.4f}]")
            if abs(es_traj_range[0] - es_pop_range[0]) < 1e-3 and abs(es_traj_range[1] - es_pop_range[1]) < 1e-3:
                print(f"  ✅ ES trajectory and population ranges are consistent")
            else:
                print(f"  ℹ️  ES population range differs from trajectory range (expected for comprehensive landscape)")
    elif ga.vals is not None and es.vals is not None:
        # Fallback: compare trajectory values if population values not available
        ga_range = (ga.vals.min(), ga.vals.max())
        es_range = (es.vals.min(), es.vals.max())
        print(f"[plot] Value consistency check (trajectory values only):")
        print(f"  GA range: [{ga_range[0]:.4f}, {ga_range[1]:.4f}]")
        print(f"  ES range: [{es_range[0]:.4f}, {es_range[1]:.4f}]")
        if abs(ga_range[0] - es_range[0]) < 1e-3 and abs(ga_range[1] - es_range[1]) < 1e-3:
            print(f"  ✅ GA and ES trajectory values are consistent")
        else:
            print(f"  ⚠️  GA and ES trajectory values have different ranges")
    
    # Check if we have background data
    have_field = len(bgP_original) > 0
    print(f"[plot] Background field available: {have_field} (bgP_original={len(bgP_original)}, bgV_original={len(bgV_original)})")
    
    # Debug: show what we're working with
    if have_field:
        for i, (pts, vals) in enumerate(zip(bgP_original, bgV_original)):
            print(f"[plot] Background {i}: pts={pts.shape}, vals={vals.shape}")
            print(f"[plot] Background {i} values range: [{vals.min():.4f}, {vals.max():.4f}]")
            if len(pts) != len(vals):
                print(f"[plot] WARNING: Background {i} has mismatched lengths!")
        
        # Show combined background statistics
        all_bg_vals = np.concatenate(bgV_original)
        print(f"[plot] Combined background: {len(all_bg_vals)} values")
        print(f"[plot] Combined background range: [{all_bg_vals.min():.4f}, {all_bg_vals.max():.4f}]")
        print(f"[plot] Combined background mean: {all_bg_vals.mean():.4f}")
        print(f"[plot] Combined background std: {all_bg_vals.std():.4f}")
    
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
        # Determine PCA input feature dimension
        try:
            expected_feat = pca_transformer.n_features_in_
        except Exception:
            expected_feat = len(pca_transformer['mean']) if isinstance(pca_transformer, dict) and 'mean' in pca_transformer else None

        def _maybe_project(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            d = arr.shape[-1]
            if expected_feat is not None and d != expected_feat:
                print(f"[PCA] Skipping projection for {name}: last-dim={d} != expected {expected_feat}")
                return arr
            print(f"[PCA] {name} before projection: {arr.shape}")
            out = _apply_fitted_pca(arr, pca_transformer, target_dim=2)
            print(f"[PCA] {name} after projection: {out.shape}")
            return out

        ga.pts = _maybe_project("GA", ga.pts)
        es.pts = _maybe_project("ES", es.pts)
        es.pop_pts = _maybe_project("ES pop", es.pop_pts)
        es.best_per_gen = _maybe_project("ES best", es.best_per_gen)
    else:
        print(f"[plot] Original latent dimension: {original_dim}D, no PCA needed")

    # unified bounds - flatten all arrays to 2D before concatenation
    # CRITICAL: Include ALL points that will be plotted to ensure complete coverage
    pts_for_bounds = []
    
    # 1. Trajectory points (GA and ES paths)
    for p in [ga.pts, es.pts, es.pop_pts, es.best_per_gen]:
        if p is not None:
            # Flatten to 2D: (..., 2) -> (N, 2)
            p_flat = p.reshape(-1, 2)
            pts_for_bounds.append(p_flat)
    
    # 2. Background loss landscape points (CRITICAL for complete coverage)
    if have_field and bgP_original:
        # Project background points to 2D for bounds calculation
        P_original = np.concatenate(bgP_original, axis=0)  # (N, D) where D is original dim
        if original_dim > 2 and pca_transformer is not None:
            print(f"[bounds] Including background points in bounds calculation: {P_original.shape}")
            # Project original high-dimensional points to 2D for bounds
            P_2d = _apply_fitted_pca(P_original, pca_transformer, target_dim=2)
            print(f"[bounds] Background bounds: {P_original.shape} -> {P_2d.shape}")
        else:
            print(f"[bounds] Background already 2D: {P_original.shape}")
            P_2d = P_original
        
        # Add background points to bounds calculation
        pts_for_bounds.append(P_2d)
        print(f"[bounds] Added {len(P_2d)} background points to bounds calculation")
    
    if not pts_for_bounds:
        print("No valid points found for bounds calculation")
        return None, None
    
    XY = np.concatenate(pts_for_bounds, axis=0)
    print(f"[bounds] Total points for bounds: {len(XY)} (trajectories + background)")
    
    # Use the larger range for both axes to ensure complete loss landscape visibility
    # This prevents "pancake" appearance and ensures all data is visible
    xmin, xmax = XY[:, 0].min(), XY[:, 0].max()
    ymin, ymax = XY[:, 1].min(), XY[:, 1].max()
    
    # Calculate ranges for each axis
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    # Use the larger range for both axes to ensure square-ish plot
    max_range = max(x_range, y_range)
    
    # Ensure minimum range for visibility when trajectories are very small
    min_range = 0.5
    max_range = max(max_range, min_range)
    
    # 🎯 COMPREHENSIVE BOUNDS CALCULATION: GUARANTEE COMPLETE VISIBILITY
    # This ensures ALL plot elements are fully visible:
    # ✅ GA trajectory points (gradient ascent path)
    # ✅ ES trajectory points (evolutionary search path) 
    # ✅ ES population points (all samples across generations)
    # ✅ ES generation circles (clustering visualization)
    # ✅ Loss landscape background (complete coverage)
    # ✅ Proper padding for visual clarity and aesthetics
    
    # COMPREHENSIVE PADDING STRATEGY:
    # 1. Base padding: 5% of the maximum range
    # 2. Circle padding: 20% extra for generation circles and population spread
    # 3. Background padding: Ensure loss landscape is fully visible
    # 4. Minimum padding: Prevent extremely tight bounds
    
    base_padding = 0.05 * max_range
    circle_padding = 0.20 * max_range
    min_padding = 0.10 * max_range
    
    # Use the maximum of all padding strategies
    pad = max(base_padding, circle_padding, min_padding)
    
    # Center the content and create a symmetric square box using the larger range
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half_final = 0.5 * max_range + pad
    xlim = (cx - half_final, cx + half_final)
    ylim = (cy - half_final, cy + half_final)
    
    # Enhanced debugging information
    print(f"[bounds] COMPREHENSIVE BOUNDS CALCULATION:")
    print(f"[bounds] All points bounds: x[{xmin:.3f}, {xmax:.3f}], y[{ymin:.3f}, {ymax:.3f}]")
    print(f"[bounds] Ranges: x_range={x_range:.3f}, y_range={y_range:.3f}")
    print(f"[bounds] Using larger range for both axes: max_range={max_range:.3f}")
    print(f"[bounds] Padding strategy:")
    print(f"[bounds]   - Base padding (5%): {base_padding:.3f}")
    print(f"[bounds]   - Circle padding (20%): {circle_padding:.3f}")
    print(f"[bounds]   - Minimum padding (10%): {min_padding:.3f}")
    print(f"[bounds]   - Final padding: {pad:.3f}")
    print(f"[bounds] Final centered bounds: center=({cx:.3f},{cy:.3f}), half_final={half_final:.3f}")
    print(f"[bounds] Final bounds: xlim={xlim}, ylim={ylim}")
    print(f"[bounds] Coverage: This ensures ALL elements are visible:")
    print(f"[bounds]   ✅ GA trajectory points")
    print(f"[bounds]   ✅ ES trajectory points") 
    print(f"[bounds]   ✅ ES population points")
    print(f"[bounds]   ✅ ES generation circles")
    print(f"[bounds]   ✅ Loss landscape background")
    print(f"[bounds]   ✅ Proper padding for visual clarity")
    print(f"[bounds]   ✅ Square-ish plot using larger range for both axes")

    # Background data has already been collected above, before PCA projection

    # normalization across everything we will color (use original values, not oriented ones)
    all_for_norm = []
    print(f"[normalization] Field type: {field_name}, have_field: {have_field}")
    if have_field:
        if bgV_original:
            bg_vals = np.concatenate(bgV_original)
            print(f"[normalization] Background values: {bg_vals.shape}, range: [{bg_vals.min():.4f}, {bg_vals.max():.4f}]")
            all_for_norm.append(bg_vals)
        else:
            print(f"[normalization] Warning: have_field=True but no background values available")
    if ga.vals is not None:
        print(f"[normalization] GA values: {ga.vals.shape}, range: [{ga.vals.min():.4f}, {ga.vals.max():.4f}]")
        all_for_norm.append(np.asarray(ga.vals))
    
    # ES values: prefer population losses for comprehensive landscape, fallback to trajectory
    if es.pop_vals is not None:
        print(f"[normalization] ES population values: {es.pop_vals.shape}, range: [{es.pop_vals.min():.4f}, {es.pop_vals.max():.4f}]")
        all_for_norm.append(np.asarray(es.pop_vals))
    elif es.vals is not None:
        print(f"[normalization] ES trajectory values (fallback): {es.vals.shape}, range: [{es.vals.min():.4f}, {es.vals.max():.4f}]")
        all_for_norm.append(np.asarray(es.vals))
    
    print(f"[normalization] Total arrays for normalization: {len(all_for_norm)}")
    
    # Set normalization based on the field type
    if field_name.lower() == "score":
        # For scores: higher is better, use original range
        vmin, vmax = 0.0, 1.0
        if len(all_for_norm) > 0:
            vv = np.concatenate(all_for_norm)
            vmin, vmax = float(np.nanmin(vv)), float(np.nanmax(vv))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
    else:
        # For losses: use original range (continuous viridis background, no abs, no white gaps)
        if len(all_for_norm) > 0:
            vv = np.concatenate(all_for_norm)
            vmin, vmax = float(np.nanmin(vv)), float(np.nanmax(vv))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
    
    print(f"[plot] Loss normalization: vmin={vmin:.4f}, vmax={vmax:.4f}")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Use custom colormap from the specified palette
    from matplotlib.colors import LinearSegmentedColormap
    custom_colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
    cmap = LinearSegmentedColormap.from_list('custom_palette', custom_colors, N=256)
    
    def orient(v: np.ndarray) -> np.ndarray:
        """Orient values for visualization: keep original values for losses (no abs)."""
        if field_name.lower() == "score":
            return v  # Scores: higher is better, keep positive
        else:
            return v  # Losses: keep original sign (continuous custom background)

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    title = f"Latent search: GA and ES (Z_dim = {original_dim})"
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel("z1", fontsize=16); ax.set_ylabel("z2", fontsize=16)
    ax.set_aspect("equal")  # With whitened PCA, this will look balanced
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # Initialize background variables to avoid UnboundLocalError
    XX, YY, ZZ = None, None, None
    bg_xmin, bg_xmax, bg_ymin, bg_ymax = None, None, None, None

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
        # Verify that background points are within the calculated bounds
        bg_xmin, bg_xmax = P_2d[:, 0].min(), P_2d[:, 0].max()
        bg_ymin, bg_ymax = P_2d[:, 1].min(), P_2d[:, 1].max()
        print(f"[background] Background points bounds: x[{bg_xmin:.3f}, {bg_xmax:.3f}], y[{bg_ymin:.3f}, {bg_ymax:.3f}]")
        print(f"[background] Plot bounds: xlim={xlim}, ylim={ylim}")
        
        # Check if background points extend beyond plot bounds
        if bg_xmin < xlim[0] or bg_xmax > xlim[1] or bg_ymin < ylim[0] or bg_ymax > ylim[1]:
            print(f"[background] ⚠️  WARNING: Background points extend beyond plot bounds!")
            print(f"[background]   Background x: [{bg_xmin:.3f}, {bg_xmax:.3f}] vs Plot x: {xlim}")
            print(f"[background]   Background y: [{bg_ymin:.3f}, {bg_ymax:.3f}] vs Plot y: {ylim}")
        else:
            print(f"[background] ✅ Background points fully within plot bounds")
        
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
        
        # Set appropriate colorbar label based on field type
        if field_name.lower() == "loss":
            cbar.set_label("Loss", fontsize=14)
        elif field_name.lower() == "score":
            cbar.set_label("Score", fontsize=14)
        else:
            cbar.set_label(field_name, fontsize=14)
        
        # Set unexplored areas to white by setting the background color
        ax.set_facecolor("white")
    else:
        ax.set_facecolor("white")

    # ES population: show all samples in orange with full alpha + translucent generation circles
    if es.pop_pts is not None:
        # Flatten ES population points for plotting
        es_pop_pts_flat = es.pop_pts.reshape(-1, 2)
        
        # Plot ALL ES samples with custom color
        ax.scatter(es_pop_pts_flat[:, 0], es_pop_pts_flat[:, 1], s=80, alpha=1.0,
                   color="#DB74DB", linewidths=0, zorder=1, label="ES population (all samples)")
        
        # Then add translucent circles to cluster samples from the same generation
        if es.gen_idx is not None or es.vals is not None:
            # Try to build a generation index aligned to flattened population
            effective_gen_idx = None
            if es.gen_idx is not None:
                candidate = es.gen_idx.reshape(-1)
                if len(candidate) == len(es_pop_pts_flat):
                    effective_gen_idx = candidate
                else:
                    print(f"[plot] Gen idx length mismatch: gen_idx={len(candidate)} vs pop_pts={len(es_pop_pts_flat)}")
            if effective_gen_idx is None and es.vals is not None:
                num_generations = len(es.vals)
                per_gen = len(es_pop_pts_flat) // num_generations if num_generations > 0 else 0
                if num_generations > 0 and per_gen * num_generations == len(es_pop_pts_flat):
                    print(f"[plot] Reconstructing generation index: {num_generations} gens × {per_gen} pop = {len(es_pop_pts_flat)}")
                    effective_gen_idx = np.repeat(np.arange(num_generations), per_gen)
                else:
                    print(f"[plot] Failed to reconstruct generation index: num_generations={num_generations}, total_pop={len(es_pop_pts_flat)}")

            if effective_gen_idx is not None:
                unique_gens = np.unique(effective_gen_idx)
                # Use custom color palette for generation clusters
                generation_colors = ['#FBB998', '#DB74DB', '#5361E5', '#96DCF8']
                for gen in unique_gens:
                    mask = (effective_gen_idx == gen)
                    if mask.shape[0] != es_pop_pts_flat.shape[0]:
                        print(f"[plot] Skipping gen {gen}: mask/pop mismatch {mask.shape[0]} vs {es_pop_pts_flat.shape[0]}")
                        continue
                    gen_pts = es_pop_pts_flat[mask]
                    if gen_pts.size == 0:
                        continue
                    color = generation_colors[int(gen) % len(generation_colors)]
                    # Calculate generation cluster center and radius
                    gen_center = np.mean(gen_pts, axis=0)
                    gen_radius = np.max(np.linalg.norm(gen_pts - gen_center, axis=1)) * 1.2  # 20% padding
                    # Debug: show generation circle bounds
                    circle_xmin = gen_center[0] - gen_radius
                    circle_xmax = gen_center[0] + gen_radius
                    circle_ymin = gen_center[1] - gen_radius
                    circle_ymax = gen_center[1] + gen_radius
                    print(f"[plot] Gen {gen} circle: center=({gen_center[0]:.3f}, {gen_center[1]:.3f}), radius={gen_radius:.3f}")
                    print(f"[plot] Gen {gen} bounds: x[{circle_xmin:.3f}, {circle_xmax:.3f}], y[{circle_ymin:.3f}, {circle_ymax:.3f}]")
                    # Draw translucent circle for this generation
                    circle = plt.Circle(gen_center, gen_radius, fill=True, linewidth=4, 
                                      edgecolor=color, facecolor=color, alpha=0.15)
                    ax.add_patch(circle)
                    # Generation labels removed as requested
            else:
                print(f"[plot] Skipping generation circles due to unavailable/mismatched generation index")

    # ES selected path (best per generation if present, otherwise es.pts)
    print(f"[plot] ES trajectory debug: best_per_gen={es.best_per_gen.shape if es.best_per_gen is not None else None}, es.pts={es.pts.shape if es.pts is not None else None}")
    es_sel = es.best_per_gen if es.best_per_gen is not None else es.pts
    print(f"[plot] ES selected for plotting: {es_sel.shape if es_sel is not None else None}")
    
    if es_sel is not None and es_sel.size > 0:
        # Check if we have enough trajectory points (more than 1 generation)
        # es_sel shape is typically (1, G, 2) where G is number of generations
        if es_sel.ndim == 3 and es_sel.shape[1] > 1:
            # Flatten ES selected path for plotting: (1, G, 2) -> (G, 2)
            es_sel_flat = es_sel.reshape(-1, 2)
            print(f"[plot] Plotting ES trajectory: {es_sel_flat.shape}, range: x[{es_sel_flat[:, 0].min():.3f}, {es_sel_flat[:, 0].max():.3f}], y[{es_sel_flat[:, 1].min():.3f}, {es_sel_flat[:, 1].max():.3f}]")
            _plot_traj(ax, es_sel_flat, color="#DB74DB", label="ES selected", alpha=1.0)
        elif es_sel.ndim == 2 and es_sel.shape[0] > 1:
            # Already flattened: (G, 2)
            es_sel_flat = es_sel
            print(f"[plot] Plotting ES trajectory (already flat): {es_sel_flat.shape}, range: x[{es_sel_flat[:, 0].min():.3f}, {es_sel_flat[:, 0].max():.3f}], y[{es_sel_flat[:, 1].min():.3f}, {es_sel_flat[:, 1].max():.3f}]")
            _plot_traj(ax, es_sel_flat, color="#DB74DB", label="ES selected", alpha=1.0)
        else:
            print(f"[plot] ES trajectory plotting skipped: es_sel shape={es_sel.shape}, not enough generations")
            
            # Fallback: try to plot ES trajectory from best_per_gen if available
            if es.best_per_gen is not None and es.best_per_gen.size > 0:
                print(f"[plot] ES fallback: attempting to plot from best_per_gen")
                es_fallback = es.best_per_gen.reshape(-1, es.best_per_gen.shape[-1])
                if es_fallback.shape[0] > 1:
                    print(f"[plot] ES fallback plotting: {es_fallback.shape}")
                    _plot_traj(ax, es_fallback, color="#DB74DB", label="ES selected (fallback)", alpha=1.0)
                else:
                    print(f"[plot] ES fallback failed: not enough points ({es_fallback.shape[0]})")
            else:
                print(f"[plot] ES fallback: no best_per_gen available")
    else:
        print(f"[plot] ES trajectory plotting skipped: es_sel is None or empty")

    # GA path
    if ga.pts is not None and len(ga.pts) > 1:
        # Flatten GA path for plotting
        ga_pts_flat = ga.pts.reshape(-1, 2)
        print(f"[plot] Plotting GA trajectory: {ga_pts_flat.shape}, range: x[{ga_pts_flat[:, 0].min():.3f}, {ga_pts_flat[:, 0].max():.3f}], y[{ga_pts_flat[:, 1].min():.3f}, {ga_pts_flat[:, 1].max():.3f}]")
        _plot_traj(ax, ga_pts_flat, color="#FBB998", label="GA path", alpha=1.0)
    else:
        print(f"[plot] GA plotting skipped: pts={ga.pts is not None}, len={len(ga.pts) if ga.pts is not None else 0}")

    # Create comprehensive legend with all elements
    legend_elements = []
    
    # ES population (all samples with custom color)
    if es.pop_pts is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DB74DB', 
                                       markersize=15, alpha=1.0, label='ES population (all samples)'))
    
    # Generation clusters (general representation)
    if es.pop_pts is not None and es.gen_idx is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='#FBB998', markerfacecolor='#FBB998', 
                                       markersize=18, alpha=0.3, label='ES generation clusters'))
    
    # Trajectory paths
    legend_elements.append(plt.Line2D([0], [0], color='#DB74DB', linewidth=5, label='ES selected path'))
    legend_elements.append(plt.Line2D([0], [0], color='#FBB998', linewidth=5, label='GA path'))
    
    # Start/End markers
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='w', 
                                     markersize=15, markeredgewidth=2, label='Start point'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='k', markerfacecolor='w', 
                                     markersize=15, markeredgewidth=2, label='End point'))
    
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=14)
    plt.tight_layout()

    # FINAL VERIFICATION: Ensure all elements are within plot bounds
    print(f"\n[verification] FINAL PLOT BOUNDS VERIFICATION:")
    print(f"[verification] Plot xlim: {xlim}, ylim: {ylim}")
    
    # Check GA trajectory bounds
    if ga.pts is not None:
        ga_flat = ga.pts.reshape(-1, 2)
        ga_xmin, ga_xmax = ga_flat[:, 0].min(), ga_flat[:, 0].max()
        ga_ymin, ga_ymax = ga_flat[:, 1].min(), ga_flat[:, 1].max()
        ga_visible = (ga_xmin >= xlim[0] and ga_xmax <= xlim[1] and ga_ymin >= ylim[0] and ga_ymax <= ylim[1])
        print(f"[verification] GA trajectory: x[{ga_xmin:.3f}, {ga_xmax:.3f}], y[{ga_ymin:.3f}, {ga_ymax:.3f}] - {'✅ VISIBLE' if ga_visible else '❌ OUT OF BOUNDS'}")
    
    # Check ES trajectory bounds
    if es.pts is not None:
        es_flat = es.pts.reshape(-1, 2)
        es_xmin, es_xmax = es_flat[:, 0].min(), es_flat[:, 0].max()
        es_ymin, es_ymax = es_flat[:, 1].min(), es_flat[:, 1].max()
        es_visible = (es_xmin >= xlim[0] and es_xmax <= xlim[1] and es_ymin >= ylim[0] and es_ymax <= ylim[1])
        print(f"[verification] ES trajectory: x[{es_xmin:.3f}, {es_xmax:.3f}], y[{es_ymin:.3f}, {es_ymax:.3f}] - {'✅ VISIBLE' if es_visible else '❌ OUT OF BOUNDS'}")
    
    # Check ES population bounds
    if es.pop_pts is not None:
        es_pop_flat = es.pop_pts.reshape(-1, 2)
        es_pop_xmin, es_pop_xmax = es_pop_flat[:, 0].min(), es_pop_flat[:, 0].max()
        es_pop_ymin, es_pop_ymax = es_pop_flat[:, 1].min(), es_pop_flat[:, 1].max()
        es_pop_visible = (es_pop_xmin >= xlim[0] and es_pop_xmax <= xlim[1] and es_pop_ymin >= ylim[0] and es_pop_ymax <= ylim[1])
        print(f"[verification] ES population: x[{es_pop_xmin:.3f}, {es_pop_xmax:.3f}], y[{es_pop_ymin:.3f}, {es_pop_ymax:.3f}] - {'✅ VISIBLE' if es_pop_visible else '❌ OUT OF BOUNDS'}")
    
    # Check background bounds
    if have_field and bg_xmin is not None:
        bg_visible = (bg_xmin >= xlim[0] and bg_xmax <= xlim[1] and bg_ymin >= ylim[0] and bg_ymax <= ylim[1])
        print(f"[verification] Background landscape: x[{bg_xmin:.3f}, {bg_xmax:.3f}], y[{bg_ymin:.3f}, {bg_ymax:.3f}] - {'✅ VISIBLE' if bg_visible else '❌ OUT OF BOUNDS'}")
    elif have_field:
        print(f"[verification] Background landscape: No background data available")
        bg_visible = False
    else:
        print(f"[verification] Background landscape: No background field requested")
        bg_visible = True  # No background to check
    
    # Overall coverage summary
    all_visible = all([
        ga.pts is None or ga_visible,
        es.pts is None or es_visible,
        es.pop_pts is None or es_pop_visible,
        not have_field or bg_visible
    ])
    
    if all_visible:
        print(f"[verification] 🎉 SUCCESS: All plot elements are fully visible within bounds!")
    else:
        print(f"[verification] ⚠️  WARNING: Some elements may extend beyond plot bounds!")
    
    print(f"[verification] Plot dimensions: {xlim[1] - xlim[0]:.3f} × {ylim[1] - ylim[0]:.3f}")
    print(f"[verification] Aspect ratio: {(xlim[1] - xlim[0]) / (ylim[1] - ylim[0]):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "search_trajectories.png")
    svg = os.path.join(out_dir, "search_trajectories.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    
    # Generate loss curves plot
    loss_plot_path = plot_loss_curves(ga, es, out_dir, original_dim, 
                                      ga_npz_path, es_npz_path,
                                      ga_steps=ga_steps, 
                                      es_population=es_population, 
                                      es_generations=es_generations, dataset_length=dataset_length)
    
    # Generate statistical histograms if dataset_length > 1
    stats_plot_path = None
    if dataset_length and dataset_length > 1:
        stats_plot_path = create_statistical_histograms(ga_npz_path, es_npz_path, out_dir, dataset_length)
        if stats_plot_path:
            print(f"[stats] Created statistical histograms: {stats_plot_path}")
        else:
            print(f"[stats] Failed to create statistical histograms")
    
    return png, loss_plot_path, stats_plot_path, original_dim


def plot_loss_curves(ga: Trace, es: Trace, out_dir: str, original_dim: int = 2, 
                     ga_npz_path: str = None, es_npz_path: str = None,
                     ga_steps: int = None, es_population: int = None, es_generations: int = None, dataset_length: int = None) -> Optional[str]:
    """
    Generate a plot comparing loss curves for GA and ES methods with budget on x-axis.
    
    Budget calculation:
    - GA: 2 evaluations per step (forward + backward pass)
      Example: 5 steps → [2, 4, 6, 8, 10] budget points (no zero evaluation)
    - ES: Cumulative evaluations at each generation
      Example: 4 generations × 5 population → [0, 5, 10, 15, 20] budget points
      Note: Generation 0: 0 evaluations, Generation 1: pop evaluations, Generation 2: 2*pop evaluations, etc.
    
    Also includes final accuracy metrics (accuracy, shape_accuracy, grid_accuracy) as notes.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Loss curve plotting unavailable: {e}")
        return None
    
    # Debug: show input parameters
    print(f"[loss] Function inputs: ga_steps={ga_steps}, es_population={es_population}, es_generations={es_generations}")
    
    # Check if we have loss data for both methods
    has_ga_loss = ga.vals is not None and len(ga.vals) > 0
    has_es_loss = es.vals is not None and len(es.vals) > 0
    
    print(f"[loss] GA loss data: {has_ga_loss}, ES loss data: {has_es_loss}")
    if has_ga_loss:
        print(f"[loss] GA vals shape: {ga.vals.shape}, type: {type(ga.vals)}")
        print(f"[loss] GA vals range: [{ga.vals.min():.4f}, {ga.vals.max():.4f}]")
        print(f"[loss] GA vals content: {ga.vals}")
    if has_es_loss:
        print(f"[loss] ES vals shape: {es.vals.shape}, type: {type(es.vals)}")
        print(f"[loss] ES vals range: [{es.vals.min():.4f}, {es.vals.max():.4f}]")
        print(f"[loss] ES vals content: {es.vals}")
    
    # Check for consistency between GA and ES loss values
    if has_ga_loss and has_es_loss:
        ga_loss_range = (ga.vals.min(), ga.vals.max())
        es_loss_range = (es.vals.min(), es.vals.max())
        print(f"[loss] Loss consistency check:")
        print(f"  GA loss range: [{ga_loss_range[0]:.4f}, {ga_loss_range[1]:.4f}]")
        print(f"  ES loss range: [{es_loss_range[0]:.4f}, {es_loss_range[1]:.4f}]")
        if abs(ga_loss_range[0] - es_loss_range[0]) < 1e-3 and abs(ga_loss_range[1] - es_loss_range[1]) < 1e-3:
            print(f"  ✅ GA and ES loss values are consistent")
        else:
            print(f"  ⚠️  GA and ES loss values have different ranges")
            print(f"  This may indicate they're using different evaluation methods or data")
    
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
    
    # Ensure x-axis shows integers for budget
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Collect all y values that are actually plotted to scale y-axis robustly (single run and multirun)
    y_values_for_limits: list[np.ndarray] = []
    
    # Helper: moving average
    def _moving_average(x: np.ndarray, k: int = 3) -> np.ndarray:
        if x.size == 0 or k <= 1:
            return x
        k = min(k, len(x))
        c = np.convolve(x, np.ones(k)/k, mode='valid')
        # Pad to original length at start
        pad = np.full(len(x) - len(c), c[0]) if len(c) > 0 else np.array([])
        return np.concatenate([pad, c])

    # Plot GA envelope (min/max) with mean line if available in NPZ
    ga_budget = None
    did_ga_overlay = False
    if ga_npz_path and os.path.exists(ga_npz_path):
        try:
            with np.load(ga_npz_path, allow_pickle=True) as f:
                if 'ga_budget' in f:
                    ga_budget = np.array(f['ga_budget']).reshape(-1)
                if (dataset_length is not None and dataset_length > 1) and 'ga_losses_per_sample' in f:
                    L = np.array(f['ga_losses_per_sample'])  # (N, S)
                    x = ga_budget if ga_budget is not None and len(ga_budget) == L.shape[1] else np.arange(L.shape[1])
                    ga_min = np.min(L, axis=0)
                    ga_max = np.max(L, axis=0)
                    ga_mean = np.mean(L, axis=0)
                    ax.fill_between(x, ga_min, ga_max, color="#FBB998", alpha=0.25, label="GA range", zorder=2)
                    ga_mean_ma = _moving_average(ga_mean, k=max(3, L.shape[1]//10))
                    ax.plot(x, ga_mean_ma, color="#FBB998", linewidth=3.0, label=f"GA mean", zorder=4)
                    # Track y extents for axis scaling
                    y_values_for_limits.append(ga_min)
                    y_values_for_limits.append(ga_max)
                    did_ga_overlay = True
        except Exception as _ge:
            print(f"[loss] Failed GA per-sample plotting: {_ge}")

    # Fallback single GA curve if no per-sample available
    if has_ga_loss and not did_ga_overlay:
        if ga_steps is not None:
            ga_budget = 2 * np.arange(1, len(ga.vals) + 1)
            print(f"[loss] GA budget calculation: {len(ga.vals)} steps → budget points: {ga_budget}")
            ax.plot(ga_budget, ga.vals, color="#FBB998", linewidth=3.0, marker='o', 
                    markersize=6, label=f"Gradient Ascent (2×{ga_steps} steps)", zorder=3)
        else:
            ga_steps_indices = np.arange(len(ga.vals))
            ax.plot(ga_steps_indices, ga.vals, color="#FBB998", linewidth=3.0, marker='o', 
                    markersize=6, label="Gradient Ascent", zorder=3)
        # Track y extents
        y_values_for_limits.append(np.asarray(ga.vals).reshape(-1))
    
    # Plot ES envelope (min/max) with mean line if available
    es_budget = None
    did_es_overlay = False
    if es_npz_path and os.path.exists(es_npz_path):
        try:
            with np.load(es_npz_path, allow_pickle=True) as f:
                if 'es_budget' in f:
                    es_budget = np.array(f['es_budget']).reshape(-1)
                if (dataset_length is not None and dataset_length > 1) and 'es_generation_losses_per_sample' in f:
                    L = np.array(f['es_generation_losses_per_sample'])  # (N, G)
                    x = es_budget if es_budget is not None and len(es_budget) == L.shape[1] else np.arange(1, L.shape[1]+1)
                    es_min = np.min(L, axis=0)
                    es_max = np.max(L, axis=0)
                    es_mean = np.mean(L, axis=0)
                    ax.fill_between(x, es_min, es_max, color="#DB74DB", alpha=0.25, label="ES range", zorder=2)
                    es_mean_ma = _moving_average(es_mean, k=max(3, L.shape[1]//4))
                    ax.plot(x, es_mean_ma, color="#DB74DB", linewidth=3.0, label=f"ES mean", zorder=4)
                    # Track y extents
                    y_values_for_limits.append(es_min)
                    y_values_for_limits.append(es_max)
                    did_es_overlay = True
        except Exception as _ee:
            print(f"[loss] Failed ES per-sample plotting: {_ee}")

    # Fallback single ES curve
    if has_es_loss and not did_es_overlay:
        if es_population is not None and es_generations is not None:
            if len(es.vals) == es_generations + 1:
                es_budget = np.arange(es_generations + 1) * es_population
            elif len(es.vals) == es_generations:
                es_budget = np.arange(1, es_generations + 1) * es_population
            else:
                es_budget = np.arange(len(es.vals)) * es_population
            if len(es_budget) == len(es.vals):
                ax.plot(es_budget, es.vals, color="#DB74DB", linewidth=3.0, marker='s', 
                        markersize=6, label=f"Evolutionary Search ({es_population}×{es_generations})", zorder=3)
            else:
                es_steps_indices = np.arange(len(es.vals))
                ax.plot(es_steps_indices, es.vals, color="#DB74DB", linewidth=3.0, marker='s', 
                        markersize=6, label="Evolutionary Search", zorder=3)
        else:
            es_steps_indices = np.arange(len(es.vals))
            ax.plot(es_steps_indices, es.vals, color="#DB74DB", linewidth=3.0, marker='s', 
                    markersize=6, label="Evolutionary Search", zorder=3)
        # Track y extents
        y_values_for_limits.append(np.asarray(es.vals).reshape(-1))
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
    
    # Robust y-axis scaling: use min/max of whatever we actually plotted (handles multirun envelopes)
    try:
        if y_values_for_limits:
            concat_vals = np.concatenate([np.asarray(v).reshape(-1) for v in y_values_for_limits])
            if concat_vals.size > 0:
                y_min = float(np.nanmin(concat_vals))
                y_max = float(np.nanmax(concat_vals))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    if y_max == y_min:
                        pad = max(1e-6, 0.05 * max(1.0, y_max))
                        ax.set_ylim(y_min - pad, y_max + pad)
                    else:
                        y_range = y_max - y_min
                        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    except Exception as _ylim_e:
        print(f"[loss] Y-axis scaling fallback due to error: {_ylim_e}")
    
    # Add legend
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Add accuracy metrics as notes
    if ga_npz_path and os.path.exists(ga_npz_path):
        try:
            with np.load(ga_npz_path, allow_pickle=True) as f:
                ga_accuracies = {}
                # Try different possible accuracy key names
                for key in ['ga_overall_accuracy', 'overall_accuracy', 'ga_accuracy', 'accuracy']:
                    if key in f:
                        ga_accuracies['overall'] = safe_array_to_scalar(np.array(f[key]))
                        break
                
                for key in ['ga_top_1_shape_accuracy', 'top_1_shape_accuracy', 'ga_shape_accuracy', 'correct_shapes', 'shape_accuracy']:
                    if key in f:
                        ga_accuracies['shape'] = safe_array_to_scalar(np.array(f[key]))
                        break
                
                for key in ['ga_top_1_accuracy', 'top_1_accuracy', 'ga_grid_accuracy', 'pixel_correctness', 'grid_accuracy']:
                    if key in f:
                        ga_accuracies['grid'] = safe_array_to_scalar(np.array(f[key]))
                        break

                # Fallback: compute from per-sample arrays if scalars weren't found
                if ('overall' not in ga_accuracies or 'shape' not in ga_accuracies or 'grid' not in ga_accuracies):
                    overall = ga_accuracies.get('overall', None)
                    shape = ga_accuracies.get('shape', None)
                    grid = ga_accuracies.get('grid', None)
                    try:
                        if overall is None and 'per_sample_accuracy' in f:
                            arr = np.array(f['per_sample_accuracy']).reshape(-1)
                            if arr.size > 0:
                                overall = float(np.mean(arr))
                        if shape is None and 'per_sample_shape_accuracy' in f:
                            arr = np.array(f['per_sample_shape_accuracy']).reshape(-1)
                            if arr.size > 0:
                                shape = float(np.mean(arr))
                        if grid is None and 'per_sample_pixel_correctness' in f:
                            arr = np.array(f['per_sample_pixel_correctness']).reshape(-1)
                            if arr.size > 0:
                                grid = float(np.mean(arr))
                    except Exception as _ga_fallback_e:
                        print(f"[loss] GA fallback accuracies failed: {_ga_fallback_e}")

                    if overall is not None:
                        ga_accuracies['overall'] = overall
                    if shape is not None:
                        ga_accuracies['shape'] = shape
                    if grid is not None:
                        ga_accuracies['grid'] = grid
                
                # Add GA accuracy note
                if ga_accuracies:
                    ga_note = "GA Final: "
                    if 'overall' in ga_accuracies:
                        ga_note += f"Acc={ga_accuracies['overall']:.3f} "
                    if 'shape' in ga_accuracies:
                        ga_note += f"Shape={ga_accuracies['shape']:.3f} "
                    if 'grid' in ga_accuracies:
                        ga_note += f"Pixel={ga_accuracies['grid']:.3f}"
                    
                    # Position note in upper left
                    ax.text(0.02, 0.98, ga_note, transform=ax.transAxes, 
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FBB998', alpha=0.8, color='white'))
                    print(f"[loss] GA accuracy note added: {ga_note}")
        except Exception as e:
            print(f"[loss] Failed to extract GA accuracy metrics: {e}")
    
    if es_npz_path and os.path.exists(es_npz_path):
        try:
            with np.load(es_npz_path, allow_pickle=True) as f:
                es_accuracies = {}
                # Try different possible accuracy key names
                for key in ['es_overall_accuracy', 'overall_accuracy', 'es_accuracy', 'accuracy']:
                    if key in f:
                        es_accuracies['overall'] = safe_array_to_scalar(np.array(f[key]))
                        break
                
                for key in ['es_top_1_shape_accuracy', 'top_1_shape_accuracy', 'es_shape_accuracy', 'correct_shapes', 'shape_accuracy']:
                    if key in f:
                        es_accuracies['shape'] = safe_array_to_scalar(np.array(f[key]))
                        break
                
                for key in ['es_top_1_accuracy', 'top_1_accuracy', 'es_grid_accuracy', 'pixel_correctness', 'grid_accuracy']:
                    if key in f:
                        es_accuracies['grid'] = safe_array_to_scalar(np.array(f[key]))
                        break

                # Fallback: compute from per-sample arrays if scalars weren't found
                if ('overall' not in es_accuracies or 'shape' not in es_accuracies or 'grid' not in es_accuracies):
                    overall = es_accuracies.get('overall', None)
                    shape = es_accuracies.get('shape', None)
                    grid = es_accuracies.get('grid', None)
                    try:
                        if overall is None and 'per_sample_accuracy' in f:
                            arr = np.array(f['per_sample_accuracy']).reshape(-1)
                            if arr.size > 0:
                                overall = float(np.mean(arr))
                        if shape is None and 'per_sample_shape_accuracy' in f:
                            arr = np.array(f['per_sample_shape_accuracy']).reshape(-1)
                            if arr.size > 0:
                                shape = float(np.mean(arr))
                        if grid is None and 'per_sample_pixel_correctness' in f:
                            arr = np.array(f['per_sample_pixel_correctness']).reshape(-1)
                            if arr.size > 0:
                                grid = float(np.mean(arr))
                    except Exception as _es_fallback_e:
                        print(f"[loss] ES fallback accuracies failed: {_es_fallback_e}")

                    if overall is not None:
                        es_accuracies['overall'] = overall
                    if shape is not None:
                        es_accuracies['shape'] = shape
                    if grid is not None:
                        es_accuracies['grid'] = grid
                
                # Add ES accuracy note
                if es_accuracies:
                    es_note = "ES Final: "
                    if 'overall' in es_accuracies:
                        es_note += f"Acc={es_accuracies['overall']:.3f} "
                    if 'shape' in es_accuracies:
                        es_note += f"Shape={es_accuracies['shape']:.3f} "
                    if 'grid' in es_accuracies:
                        es_note += f"Pixel={es_accuracies['grid']:.3f}"
                    
                    # Position note below GA note
                    ax.text(0.02, 0.92, es_note, transform=ax.transAxes, 
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='#DB74DB', alpha=0.8, color='white'))
                    print(f"[loss] ES accuracy note added: {es_note}")
        except Exception as e:
            print(f"[loss] Failed to extract ES accuracy metrics: {e}")
    
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


def create_statistical_histograms(ga_npz_path: str, es_npz_path: str, out_dir: str, dataset_length: int) -> Optional[str]:
    """
    Create statistical histograms comparing GA vs ES performance across the four metrics:
    - accuracy (overall)
    - shape correctness 
    - pixel correctness
    - best loss (minimum loss achieved per sample)
    
    Only creates histograms when dataset_length > 1 (multiple samples evaluated).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        print(f"Statistical histogram plotting unavailable: {e}")
        return None
    
    if dataset_length <= 1:
        print(f"[stats] Skipping statistical histograms: dataset_length={dataset_length} (need > 1)")
        return None
    
    print(f"[stats] Creating statistical histograms for {dataset_length} samples...")
    
    # Extract per-sample metrics from both NPZ files
    ga_metrics = {}
    es_metrics = {}
    
    # Load GA metrics
    if os.path.exists(ga_npz_path):
        try:
            with np.load(ga_npz_path, allow_pickle=True) as f:
                print(f"[stats] GA NPZ keys: {list(f.keys())}")
                
                # Extract per-sample metrics
                for key, wandb_key in [
                    ('per_sample_accuracy', 'accuracy'),
                    ('per_sample_shape_accuracy', 'shape_correctness'),
                    ('per_sample_pixel_correctness', 'pixel_correctness'),
                ]:
                    if key in f:
                        arr = np.array(f[key]).reshape(-1)
                        if arr.size > 0:
                            ga_metrics[wandb_key] = arr
                            print(f"[stats] GA {wandb_key}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")
                        else:
                            print(f"[stats] GA {wandb_key}: empty array")
                    else:
                        print(f"[stats] GA {wandb_key}: key '{key}' not found")
                
                # Extract best losses from per-sample loss trajectories
                if 'ga_losses_per_sample' in f:
                    ga_losses_per_sample = np.array(f['ga_losses_per_sample'])
                    if ga_losses_per_sample.size > 0:
                        # Take the minimum loss (best performance) for each sample
                        ga_best_losses = np.min(ga_losses_per_sample, axis=1)
                        ga_metrics['best_loss'] = ga_best_losses
                        print(f"[stats] GA best_loss: shape={ga_best_losses.shape}, mean={ga_best_losses.mean():.4f}, std={ga_best_losses.std():.4f}")
                    else:
                        print(f"[stats] GA best_loss: empty array")
                else:
                    print(f"[stats] GA best_loss: key 'ga_losses_per_sample' not found")
                    
        except Exception as e:
            print(f"[stats] Failed to load GA metrics: {e}")
    
    # Load ES metrics
    if os.path.exists(es_npz_path):
        try:
            with np.load(es_npz_path, allow_pickle=True) as f:
                print(f"[stats] ES NPZ keys: {list(f.keys())}")
                
                # Extract per-sample metrics
                for key, wandb_key in [
                    ('per_sample_accuracy', 'accuracy'),
                    ('per_sample_shape_accuracy', 'shape_correctness'),
                    ('per_sample_pixel_correctness', 'pixel_correctness'),
                ]:
                    if key in f:
                        arr = np.array(f[key]).reshape(-1)
                        if arr.size > 0:
                            es_metrics[wandb_key] = arr
                            print(f"[stats] ES {wandb_key}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")
                        else:
                            print(f"[stats] ES {wandb_key}: empty array")
                    else:
                        print(f"[stats] ES {wandb_key}: key '{key}' not found")
                
                # Extract best losses from per-sample loss trajectories
                if 'es_generation_losses_per_sample' in f:
                    es_losses_per_sample = np.array(f['es_generation_losses_per_sample'])
                    if es_losses_per_sample.size > 0:
                        # Take the minimum loss (best performance) for each sample
                        es_best_losses = np.min(es_losses_per_sample, axis=1)
                        es_metrics['best_loss'] = es_best_losses
                        print(f"[stats] ES best_loss: shape={es_best_losses.shape}, mean={es_best_losses.mean():.4f}, std={es_best_losses.std():.4f}")
                    else:
                        print(f"[stats] ES best_loss: empty array")
                else:
                    print(f"[stats] ES best_loss: key 'es_generation_losses_per_sample' not found")
                    
        except Exception as e:
            print(f"[stats] Failed to load ES metrics: {e}")
    
    # Check if we have enough data to create histograms
    if not ga_metrics and not es_metrics:
        print("[stats] No per-sample metrics found in either NPZ file")
        return None
    
    # Optional diagnostic: warn if GA and ES arrays are (nearly) identical
    def _warn_if_identical(name: str, a: Optional[np.ndarray], b: Optional[np.ndarray]):
        try:
            if a is None or b is None:
                return
            if a.size == 0 or b.size == 0:
                return
            n = min(a.size, b.size)
            aa = a[:n].astype(float)
            bb = b[:n].astype(float)
            if np.allclose(aa, bb, atol=1e-8, rtol=1e-6):
                print(f"[stats][warn] GA and ES {name} arrays are numerically identical (n={n}).")
            else:
                diff = float(np.mean(np.abs(aa - bb)))
                print(f"[stats] Mean |GA-ES| for {name}: {diff:.6f} (n={n})")
        except Exception as _w_e:
            print(f"[stats] Identical-check failed for {name}: {_w_e}")

    _warn_if_identical('accuracy', ga_metrics.get('accuracy'), es_metrics.get('accuracy'))
    _warn_if_identical('shape_correctness', ga_metrics.get('shape_correctness'), es_metrics.get('shape_correctness'))
    _warn_if_identical('pixel_correctness', ga_metrics.get('pixel_correctness'), es_metrics.get('pixel_correctness'))
    _warn_if_identical('best_loss', ga_metrics.get('best_loss'), es_metrics.get('best_loss'))

    # Create figure with 4 subplots (one for each metric)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Statistical Analysis: GA vs ES Performance ({dataset_length} samples)", fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define colors and labels
    ga_color = '#FBB998'  # Custom color for GA
    es_color = '#DB74DB'  # Custom color for ES
    alpha = 0.7
    
    # Metric names and descriptions
    metrics_info = {
        'accuracy': ('Overall Accuracy', 'Fraction of completely correct solutions'),
        'shape_correctness': ('Shape Correctness', 'Fraction of correct shape predictions'),
        'pixel_correctness': ('Pixel Correctness', 'Fraction of correct pixel predictions'),
        'best_loss': ('Best Loss', 'Minimum loss achieved per sample (lower is better)')
    }
    
    # Create histograms for each metric
    for i, (metric, (title, description)) in enumerate(metrics_info.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get data for this metric
        ga_data = ga_metrics.get(metric, None)
        es_data = es_metrics.get(metric, None)
        
        # Determine histogram bins based on available data
        all_data = []
        if ga_data is not None:
            all_data.extend(ga_data)
        if es_data is not None:
            all_data.extend(es_data)
        
        if not all_data:
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{title}\n{description}", fontsize=12)
            continue
        
        # Create histogram bins (10-20 bins, adaptive to data range)
        min_val, max_val = min(all_data), max(all_data)
        if min_val == max_val:
            bins = 10
        else:
            # Adaptive binning: more bins for wider ranges
            range_val = max_val - min_val
            if range_val < 0.1:
                bins = 10
            elif range_val < 0.5:
                bins = 15
            else:
                bins = 20
        
        # Plot GA histogram
        if ga_data is not None and len(ga_data) > 0:
            ax.hist(ga_data, bins=bins, alpha=alpha, color=ga_color, label='GA', 
                   edgecolor='black', linewidth=0.5, density=True)
            
            # Add mean and std lines
            ga_mean = np.mean(ga_data)
            ga_std = np.std(ga_data)
            ax.axvline(ga_mean, color=ga_color, linestyle='--', linewidth=2, 
                      label=f'GA Mean: {ga_mean:.3f}')
            ax.axvline(ga_mean + ga_std, color=ga_color, linestyle=':', linewidth=1, alpha=0.7)
            ax.axvline(ga_mean - ga_std, color=ga_color, linestyle=':', linewidth=1, alpha=0.7)
            
            # Add text annotation
            ax.text(0.02, 0.98, f'GA: μ={ga_mean:.3f}, σ={ga_std:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=ga_color, alpha=0.8, color='white'))
        
        # Plot ES histogram
        if es_data is not None and len(es_data) > 0:
            ax.hist(es_data, bins=bins, alpha=alpha, color=es_color, label='ES', 
                   edgecolor='black', linewidth=0.5, density=True)
            
            # Add mean and std lines
            es_mean = np.mean(es_data)
            es_std = np.std(es_data)
            ax.axvline(es_mean, color=es_color, linestyle='--', linewidth=2, 
                      label=f'ES Mean: {es_mean:.3f}')
            ax.axvline(es_mean + es_std, color=es_color, linestyle=':', linewidth=1, alpha=0.7)
            ax.axvline(es_mean - es_std, color=es_color, linestyle=':', linewidth=1, alpha=0.7)
            
            # Add text annotation
            ax.text(0.02, 0.90, f'ES: μ={es_mean:.3f}, σ={es_std:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=es_color, alpha=0.8, color='white'))
        
        # Customize subplot
        ax.set_title(f"{title}\n{description}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{title} Value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set x-axis limits with some padding
        if all_data:
            data_range = max(all_data) - min(all_data)
            padding = 0.05 * data_range if data_range > 0 else 0.1
            ax.set_xlim(min(all_data) - padding, max(all_data) + padding)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "statistical_analysis.png")
    svg_path = os.path.join(out_dir, "statistical_analysis.svg")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[stats] Saved statistical histograms to {png_path}")
    return png_path


def upload_to_wandb(project: str, entity: Optional[str], cfg: dict, ga_npz: str, es_npz: str, 
                    trajectory_plot: Optional[str], loss_plot: Optional[str], stats_plot: Optional[str] = None, 
                    group_name: str = None, existing_run = None) -> None:
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
    
    # Use existing run if provided, otherwise create new one
    if existing_run is not None:
        run = existing_run
        print(f"[wandb] Using existing run: {run.name}")
        print(f"[wandb] Will upload plots and metrics to existing run: {run.name}")
    else:
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
    print(f"[wandb] Attempting to upload plots...")
    print(f"[wandb] Trajectory plot path: {trajectory_plot}")
    print(f"[wandb] Loss plot path: {loss_plot}")
    print(f"[wandb] Statistical plot path: {stats_plot}")
    
    if trajectory_plot and os.path.exists(trajectory_plot):
        print(f"[wandb] ✅ Uploading trajectory plot: {trajectory_plot}")
        wandb.log({"trajectory_plot": wandb.Image(trajectory_plot)})
    else:
        print(f"[wandb] ❌ Trajectory plot not found or invalid: {trajectory_plot}")
    
    if loss_plot and os.path.exists(loss_plot):
        print(f"[wandb] ✅ Uploading loss curves plot: {loss_plot}")
        wandb.log({"loss_curves_plot": wandb.Image(loss_plot)})
    else:
        print(f"[wandb] ❌ Loss plot not found or invalid: {loss_plot}")
    
    # Log statistical histograms if available
    if stats_plot and os.path.exists(stats_plot):
        print(f"[wandb] ✅ Uploading statistical histograms: {stats_plot}")
        wandb.log({"statistical_analysis": wandb.Image(stats_plot)})
    elif stats_plot:
        print(f"[wandb] ❌ Statistical plot not found: {stats_plot}")
    else:
        print(f"[wandb] ℹ️  No statistical plot to upload (dataset_length <= 1)")
    
    print(f"[wandb] Plot upload completed")
    
    # Log comprehensive metrics from both methods
    try:
        print("[metrics] Extracting metrics from NPZ files...")
        
        # Debug: show what keys are available in the NPZ files
        if os.path.exists(ga_npz):
            with np.load(ga_npz, allow_pickle=True) as f:
                print(f"[metrics] GA NPZ keys: {list(f.keys())}")
        if os.path.exists(es_npz):
            with np.load(es_npz, allow_pickle=True) as f:
                print(f"[metrics] ES NPZ keys: {list(f.keys())}")
        
        # Load GA metrics
        if os.path.exists(ga_npz):
            with np.load(ga_npz, allow_pickle=True) as f:
                ga_metrics = {}
                if 'ga_losses' in f:
                    ga_losses = np.array(f['ga_losses'])
                    if len(ga_losses) > 0:
                        ga_metrics['ga_final_loss'] = safe_array_to_scalar(ga_losses[-1])
                        ga_metrics['ga_loss_progression'] = ga_losses.tolist()
                        ga_metrics['ga_loss_improvement'] = safe_array_to_scalar(ga_losses[0] - ga_losses[-1])
                    else:
                        ga_metrics['ga_final_loss'] = None
                        ga_metrics['ga_loss_progression'] = []
                        ga_metrics['ga_loss_improvement'] = None
                
                if 'ga_scores' in f:
                    ga_scores = np.array(f['ga_scores'])
                    if len(ga_scores) > 0:
                        ga_metrics['ga_final_score'] = safe_array_to_scalar(ga_scores[-1])
                        ga_metrics['ga_score_progression'] = ga_scores.tolist()
                        ga_metrics['ga_score_improvement'] = safe_array_to_scalar(ga_scores[-1] - ga_scores[0])
                    else:
                        ga_metrics['ga_final_score'] = None
                        ga_metrics['ga_score_progression'] = []
                        ga_metrics['ga_score_improvement'] = None
                
                if 'ga_log_probs' in f:
                    ga_log_probs = np.array(f['ga_log_probs'])
                    if len(ga_log_probs) > 0:
                        ga_metrics['ga_final_log_prob'] = safe_array_to_scalar(ga_log_probs[-1])
                        ga_metrics['ga_log_prob_progression'] = ga_log_probs.tolist()
                        ga_metrics['ga_log_prob_improvement'] = safe_array_to_scalar(ga_log_probs[-1] - ga_log_probs[0])
                    else:
                        ga_metrics['ga_final_log_prob'] = None
                        ga_metrics['ga_log_prob_progression'] = []
                        ga_metrics['ga_log_prob_improvement'] = None
                
                # Log GA metrics
                for key, value in ga_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Per-sample histograms (if available)
                for key, wandb_key in [
                    ('per_sample_accuracy', 'ga_per_sample_accuracy'),
                    ('per_sample_shape_accuracy', 'ga_per_sample_shape_accuracy'),
                    ('per_sample_pixel_correctness', 'ga_per_sample_pixel_correctness'),
                ]:
                    if key in f:
                        arr = np.array(f[key]).reshape(-1)
                        if arr.size > 0:
                            try:
                                run.log({f"{wandb_key}_hist": wandb.Histogram(arr)})
                                run.log({f"{wandb_key}_mean": float(np.mean(arr))})
                                run.log({f"{wandb_key}_std": float(np.std(arr))})
                            except Exception as _we:
                                print(f"[metrics] Failed to log GA histogram {wandb_key}: {_we}")

                # Create GA metrics CSV for download
                ga_csv_path = f"ga_metrics_run_{cfg.get('run_idx', 0)}.csv"
                
                # Ensure 1D arrays and align lengths
                losses_1d = np.array(f['ga_losses']).reshape(-1) if 'ga_losses' in f else np.array([])
                scores_1d = np.array(f['ga_scores']).reshape(-1) if 'ga_scores' in f else np.array([])
                logprobs_1d = np.array(f['ga_log_probs']).reshape(-1) if 'ga_log_probs' in f else np.array([])
                
                lengths = [len(losses_1d), len(scores_1d), len(logprobs_1d)]
                max_len = max([l for l in lengths] + [0])
                
                def align_len(arr: np.ndarray, n: int) -> list:
                    if n == 0:
                        return []
                    if arr.size == 0:
                        return [float('nan')] * n
                    if len(arr) == n:
                        return arr.astype(float).tolist()
                    if len(arr) > n:
                        return arr[:n].astype(float).tolist()
                    # pad with NaN
                    pad = [float('nan')] * (n - len(arr))
                    return arr.astype(float).tolist() + pad
                
                data_dict = {
                    'step': list(range(max_len)) if max_len > 0 else [],
                    'loss': align_len(losses_1d, max_len),
                    'score': align_len(scores_1d, max_len),
                    'log_prob': align_len(logprobs_1d, max_len),
                }
                
                ga_metrics_df = pd.DataFrame(data_dict)
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
                gen_losses = None
                if 'es_generation_losses' in f:
                    gen_losses = np.array(f['es_generation_losses'])
                elif 'es_generation_fitness' in f:
                    # Convert fitness to positive losses
                    gen_losses = -np.array(f['es_generation_fitness'])
                if gen_losses is not None:
                    if len(gen_losses) > 0:
                        es_metrics['es_final_loss'] = safe_array_to_scalar(gen_losses[-1])
                        es_metrics['es_loss_progression'] = gen_losses.tolist()
                        es_metrics['es_loss_improvement'] = safe_array_to_scalar(gen_losses[0] - gen_losses[-1])
                        es_metrics['es_best_loss'] = safe_array_to_scalar(np.min(gen_losses))
                    else:
                        es_metrics['es_final_loss'] = None
                        es_metrics['es_loss_progression'] = []
                        es_metrics['es_loss_improvement'] = None
                        es_metrics['es_best_loss'] = None
                
                if 'es_best_losses_per_generation' in f:
                    best_losses = np.array(f['es_best_losses_per_generation'])
                    es_metrics['es_best_loss_progression'] = best_losses.tolist()
                
                # Population-level metrics
                if 'es_all_losses' in f:
                    all_losses = np.array(f['es_all_losses'])
                    es_metrics['es_population_size'] = len(all_losses)
                    if len(all_losses) > 0:
                        es_metrics['es_min_loss'] = safe_array_to_scalar(np.min(all_losses))
                        es_metrics['es_max_loss'] = safe_array_to_scalar(np.max(all_losses))
                        es_metrics['es_mean_loss'] = safe_array_to_scalar(np.mean(all_losses))
                        es_metrics['es_std_loss'] = safe_array_to_scalar(np.std(all_losses))
                    else:
                        es_metrics['es_min_loss'] = None
                        es_metrics['es_max_loss'] = None
                        es_metrics['es_mean_loss'] = None
                        es_metrics['es_std_loss'] = None
                
                if 'es_final_best_loss' in f:
                    es_metrics['es_final_best_loss'] = safe_array_to_scalar(np.array(f['es_final_best_loss']))
                if 'es_final_best_fitness' in f:
                    final_fitness = np.array(f['es_final_best_fitness'])
                    es_metrics['es_final_best_fitness'] = safe_array_to_scalar(final_fitness)
                    if final_fitness.size > 1:
                        print(f"[metrics] Note: es_final_best_fitness had {final_fitness.size} elements, using first: {final_fitness.flat[0]:.4f}")
                
                # Per-sample histograms (if available)
                for key, wandb_key in [
                    ('per_sample_accuracy', 'es_per_sample_accuracy'),
                    ('per_sample_shape_accuracy', 'es_per_sample_shape_accuracy'),
                    ('per_sample_pixel_correctness', 'es_per_sample_pixel_correctness'),
                ]:
                    if key in f:
                        arr = np.array(f[key]).reshape(-1)
                        if arr.size > 0:
                            try:
                                run.log({f"{wandb_key}_hist": wandb.Histogram(arr)})
                                run.log({f"{wandb_key}_mean": float(np.mean(arr))})
                                run.log({f"{wandb_key}_std": float(np.std(arr))})
                            except Exception as _we:
                                print(f"[metrics] Failed to log ES histogram {wandb_key}: {_we}")

                # Log ES metrics
                for key, value in es_metrics.items():
                    if value is not None:
                        wandb.log({key: value})
                
                # Create ES metrics CSV for download
                es_csv_path = f"es_metrics_run_{cfg.get('run_idx', 0)}.csv"
                
                # Prepare ES metrics data (ensure 1D arrays and aligned lengths)
                es_csv_path = f"es_metrics_run_{cfg.get('run_idx', 0)}.csv"
                
                gens_1d = np.array(range(len(gen_losses))) if 'es_generation_losses' in f else np.array([])
                gen_losses_1d = np.array(gen_losses).reshape(-1) if 'es_generation_losses' in f else np.array([])
                best_losses_1d = np.array(best_losses).reshape(-1) if 'es_best_losses_per_generation' in f else np.array([])
                pop_gen_1d = np.array(f['es_generation_idx']).reshape(-1) if 'es_all_losses' in f and 'es_generation_idx' in f else np.array([])
                pop_loss_1d = np.array(all_losses).reshape(-1) if 'es_all_losses' in f else np.array([])
                
                lengths = [len(gens_1d), len(gen_losses_1d), len(best_losses_1d), len(pop_gen_1d), len(pop_loss_1d)]
                max_len = max([l for l in lengths] + [0])
                
                def align_len_es(arr: np.ndarray, n: int) -> list:
                    if n == 0:
                        return []
                    if arr.size == 0:
                        return [float('nan')] * n
                    if len(arr) == n:
                        return arr.astype(float).tolist()
                    if len(arr) > n:
                        return arr[:n].astype(float).tolist()
                    pad = [float('nan')] * (n - len(arr))
                    return arr.astype(float).tolist() + pad
                
                es_data = {
                    'generation': align_len_es(gens_1d, max_len),
                    'generation_loss': align_len_es(gen_losses_1d, max_len),
                    'best_loss': align_len_es(best_losses_1d, max_len),
                    'population_generation': align_len_es(pop_gen_1d, max_len),
                    'population_loss': align_len_es(pop_loss_1d, max_len),
                }
                
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
                    'es_mutation_std': cfg.get('es_mutation_std'),
                    'es_mutation_decay': cfg.get('es_mutation_decay'),
                    'es_elite_size': cfg.get('es_elite_size'),
                    'run_seed': cfg.get('dataset_seed'),
                    'run_index': cfg.get('run_idx', 0),
                    'n_samples': cfg.get('n_samples', 1),
                    'dataset_length': cfg.get('dataset_length'),
                    'dataset_folder': cfg.get('dataset_folder'),  # Dataset folder name
                    'latent_dimension': cfg.get('latent_dimension'),  # Latent space dimension
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
                
                print("[metrics] Successfully extracted and logged all metrics")
                
    except Exception as e:
        print(f"Warning: Failed to extract metrics from NPZ files: {e}")
        print(f"[metrics] Error details: {type(e).__name__}: {str(e)}")
        
        # Try to identify which specific metric extraction failed
        import traceback
        print(f"[metrics] Full traceback:")
        traceback.print_exc()
        
        # Continue with basic metrics even if some fail
        print(f"[metrics] Attempting to continue with basic metrics...")
        
        # Log basic run information
        try:
            if 'run' in locals():
                wandb.log({
                    "metrics_extraction_failed": True,
                    "metrics_error": str(e),
                    "run_budget": cfg.get('budget', 'unknown'),
                    "dataset_folder": cfg.get('dataset_folder', 'unknown'),
                    "latent_dimension": cfg.get('latent_dimension', 'unknown')
                })
        except Exception as log_error:
            print(f"[metrics] Failed to log error to wandb: {log_error}")
    
    # Only finish the run if we created it (not if using existing_run)
    if existing_run is None:
        run.finish()


def main() -> None:
    # Initialize run variable
    run = None
    
    parser = argparse.ArgumentParser(
        description="Store and plot latent search trajectories (GA & ES). "
        "Both methods start from the same mean latent for fair comparison. "
        "Automatically applies PCA to reduce latent dimensions > 2 to 2D for visualization. "
        "Use --ga_steps, --es_population, --es_generations to override automatic budget-based calculations. "
        "Use --n_samples to run multiple experiments with different seeds for statistical analysis. "
        "Use --dataset_length to control the number of samples evaluated from the dataset. "
        "Use --run_name to set a custom name for W&B runs."
    )
    parser.add_argument("--wandb_artifact_path", required=True, type=str)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--ga_lr", type=float, default=0.5)
    parser.add_argument("--ga_steps", type=int, default=None, help="Number of GA steps (overrides budget/2 calculation)")
    parser.add_argument("--es_mutation_std", type=float, default=0.5)
    parser.add_argument("--es_population", type=int, default=None, help="ES population size (overrides sqrt(budget) calculation)")
    parser.add_argument("--es_generations", type=int, default=None, help="ES number of generations (overrides budget/pop calculation)")
    parser.add_argument("--mutation-decay", "--mutation_decay", dest="mutation_decay", type=float, default=None, help="Multiply mutation std by this factor each generation (default 0.95)")
    parser.add_argument("--elite-size", "--elite_size", dest="elite_size", type=int, default=None, help="Number of top candidates preserved each generation (default population//2)")

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
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the W&B run (overrides default naming)")
    # Data source
    parser.add_argument("--json_challenges", type=str, default=None)
    parser.add_argument("--json_solutions", type=str, default=None)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--dataset_length", type=int, default=None, help="Number of samples in the dataset to evaluate")
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument("--dataset_use_hf", type=str, default="true")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1, help="Number of times to run the script with different random seeds (for statistical analysis)")
    parser.add_argument("--aggregate_statistics", action="store_true", help="Aggregate per-sample metrics across n_samples runs and generate an aggregated statistical plot")
    parser.add_argument("--no_files", action="store_true", help="Disable file generation and plotting (faster, just return values)")
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
    # Log GA start
    if run is not None:
        wandb.log({"ga_status": "started"})
    
    print("Running:", " ".join(ga_cmd))
    ga_rc = subprocess.run(ga_cmd, check=False).returncode
    print(f"GA return code: {ga_rc}")
    
    # Log GA completion
    if run is not None:
        wandb.log({"ga_return_code": ga_rc, "ga_status": "completed"})

    # Evolutionary Search config
    pop = args.es_population if args.es_population is not None else max(3, min(32, int(round(math.sqrt(args.budget)))))
    gens = args.es_generations if args.es_generations is not None else max(1, int(math.ceil(args.budget / pop)))
    mutation_decay = args.mutation_decay if args.mutation_decay is not None else 0.95
    elite_size = args.elite_size if args.elite_size is not None else max(1, pop // 2)
    print(
        f"🧬 ES config: population={pop}, generations={gens} (mutation_std={args.es_mutation_std}, mutation_decay={mutation_decay}, elite_size={elite_size})"
    )    print(f"   🎯 ES starts from mean latent (same as GA)")
    es_out = os.path.join(args.out_dir, "es_latents.npz")
    es_cmd = [
        sys.executable, "src/evaluate_checkpoint.py",
        "-w", args.wandb_artifact_path,
        "-i", "evolutionary_search",
        "--population-size", str(pop),
        "--num-generations", str(gens),
        "--mutation-std", str(args.es_mutation_std),
        "--mutation-decay", str(mutation_decay),
        "--elite-size", str(elite_size),
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

    # Log ES start
    if run is not None:
        wandb.log({"es_status": "started"})
    
    print("Running:", " ".join(es_cmd))
    es_rc = subprocess.run(es_cmd, check=False).returncode
    print(f"ES return code: {es_rc}")

    # Log ES completion
    if run is not None:
        wandb.log({"es_return_code": es_rc, "es_status": "completed"})

    # Handle multiple runs when n_samples > 1
    # NOTE: n_samples controls how many times to run the script with different seeds
    # dataset_length controls how many samples from the dataset to evaluate in each run
    if args.n_samples > 1:
        print(f"🧪 Running {args.n_samples} experiments with different seeds...")
        
        # Create a group name for W&B
        if args.run_name:
            group_name = f"{args.run_name}-n{args.n_samples}-{int(time.time())}"
        else:
            group_name = f"latent-search-b{args.budget}-n{args.n_samples}-{int(time.time())}"
        
        # Optional aggregators for cross-run statistics
        ga_agg_acc, ga_agg_shape, ga_agg_pixel, ga_agg_best_loss = [], [], [], []
        es_agg_acc, es_agg_shape, es_agg_pixel, es_agg_best_loss = [], [], [], []

        for run_idx in range(args.n_samples):
            seed = args.dataset_seed + run_idx
            print(f"\n🔬 Run {run_idx + 1}/{args.n_samples} with seed {seed}")
            
            # Start W&B run at the beginning to capture all logs
            cfg = {
                "artifact_path": args.wandb_artifact_path,
                "budget": args.budget,
                "ga_steps": ga_steps,
                "ga_lr": args.ga_lr,
                "es_population": pop,
                "es_generations": gens,
                "es_mutation_std": args.es_mutation_std,
                "es_mutation_decay": mutation_decay,
                "es_elite_size": elite_size,
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
                "run_idx": run_idx,
                "run_name": args.run_name,  # Custom run name if provided
                "dataset_seed": seed,
                "n_samples": args.n_samples,
                "dataset_length": args.dataset_length,
                "dataset_folder": args.dataset_folder,  # Dataset folder name
                "latent_dimension": None,  # Will be updated after plotting
            }
            
            try:
                # Initialize W&B run at the start
                import wandb
                
                # Use custom run name if provided, otherwise use default naming
                if args.run_name:
                    run_name = f"{args.run_name}-s{run_idx}-{int(time.time())}" if args.n_samples > 1 else f"{args.run_name}-{int(time.time())}"
                else:
                    run_name = f"latent-search-b{args.budget}-s{run_idx}-{int(time.time())}"
                
                run_kwargs = {
                    "project": args.wandb_project,
                    "entity": args.wandb_entity,
                    "name": run_name,
                    "config": cfg,
                    "group": group_name
                }
                run = wandb.init(**run_kwargs)
                print(f"[wandb] Started run {run_idx + 1}/{args.n_samples} in group: {group_name}")
                print(f"[wandb] Run name: {run_name}")
                
                # Log run start with dataset info
                wandb.log({
                    "run_status": "started", 
                    "run_index": run_idx,
                    "run_name": run_name,
                    "dataset_folder": args.dataset_folder,
                    "dataset_length": args.dataset_length
                })
                
            except Exception as e:
                print(f"Failed to start wandb run: {e}")
                run = None
            
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
            
            # Log GA start
            if run is not None:
                wandb.log({"ga_status": "started", "ga_seed": seed})
            
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
            
            # Log GA completion
            if run is not None:
                wandb.log({"ga_return_code": ga_rc, "ga_status": "completed"})
            
            # Log ES start
            if run is not None:
                wandb.log({"es_status": "started", "es_seed": seed})
            
            # Run ES for this seed
            es_cmd = [
                sys.executable, "src/evaluate_checkpoint.py",
                "-w", args.wandb_artifact_path,
                "-i", "evolutionary_search",
                "--population-size", str(pop),
                "--num-generations", str(gens),
                "--mutation-std", str(args.es_mutation_std),
                "--mutation-decay", str(mutation_decay),
                "--elite-size", str(elite_size),
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
            
            # Log ES completion
            if run is not None:
                wandb.log({"es_return_code": es_rc, "es_status": "completed"})
            
            # Plot for this run - skip if no_files is set
            if args.no_files:
                print("🚫 File generation disabled - skipping plots and file saves")
                trajectory_plot, loss_plot, stats_plot = None, None, None
                # Try to get latent dimension from existing files if available
                latent_dim = 2  # Default fallback
                try:
                    if os.path.exists(ga_out):
                        with np.load(ga_out, allow_pickle=True) as f:
                            if 'ga_latents' in f:
                                ga_latents = np.array(f['ga_latents'])
                                if ga_latents.size > 0:
                                    latent_dim = ga_latents.shape[-1]
                                    print(f"📊 Extracted latent dimension from existing GA file: {latent_dim}")
                except Exception as e:
                    print(f"⚠️  Could not extract latent dimension: {e}")
            else:
                trajectory_plot, loss_plot, stats_plot, latent_dim = plot_and_save(ga_out, es_out, args.out_dir, 
                                                          background_resolution=args.background_resolution,
                                                          background_smoothing=args.background_smoothing,
                                                          background_knn=args.background_knn,
                                                          background_bandwidth_scale=args.background_bandwidth_scale,
                                                          background_global_mix=args.background_global_mix,
                                                          ga_steps=ga_steps, es_population=pop, es_generations=gens, dataset_length=args.dataset_length)
                if trajectory_plot:
                    print(f"Saved trajectory plot to {trajectory_plot}")
                if loss_plot:
                    print(f"Saved loss curves plot to {loss_plot}")
                if stats_plot:
                    print(f"Saved statistical histograms to {stats_plot}")
                else:
                    print(f"No statistical histograms created (dataset_length={args.dataset_length})")
            
            print(f"📊 Latent space dimension: {latent_dim}")
            
            # Log plotting completion
            if run is not None:
                wandb.log({"plotting_status": "completed", "latent_dimension": latent_dim})
            
            # Upload final results to W&B
            try:
                if run is not None:
                    # Add return codes and latent dimension to config for final upload
                    cfg.update({
                        "ga_return_code": ga_rc,
                        "es_return_code": es_rc,
                        "latent_dimension": latent_dim,  # Latent space dimension
                    })
                    
                    # Debug: show plot paths before upload
                    print(f"[debug] Plot paths before upload:")
                    print(f"[debug] trajectory_plot: {trajectory_plot}")
                    print(f"[debug] loss_plot: {loss_plot}")
                    print(f"[debug] trajectory_plot exists: {os.path.exists(trajectory_plot) if trajectory_plot else False}")
                    print(f"[debug] loss_plot exists: {os.path.exists(loss_plot) if loss_plot else False}")
                    
                    # Upload artifacts and final metrics
                    upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot, stats_plot, group_name, existing_run=run)
                else:
                    # Fallback: create new run for upload
                    upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot, stats_plot, group_name)
            except Exception as e:
                print(f"Failed to upload to wandb: {e}")
            
                        # Finish the W&B run
            if run is not None:
                run.finish()
                print(f"[wandb] Finished run {run_idx + 1}/{args.n_samples}")

            # Aggregate per-sample metrics across runs (if requested later)
            try:
                # Load GA per-sample metrics from NPZ
                if os.path.exists(ga_out):
                    with np.load(ga_out, allow_pickle=True) as fga:
                        if 'per_sample_accuracy' in fga:
                            ga_agg_acc.append(np.array(fga['per_sample_accuracy']).reshape(-1))
                        if 'per_sample_shape_accuracy' in fga:
                            ga_agg_shape.append(np.array(fga['per_sample_shape_accuracy']).reshape(-1))
                        if 'per_sample_pixel_correctness' in fga:
                            ga_agg_pixel.append(np.array(fga['per_sample_pixel_correctness']).reshape(-1))
                        if 'ga_losses_per_sample' in fga:
                            ga_losses_per_sample = np.array(fga['ga_losses_per_sample'])
                            if ga_losses_per_sample.size > 0:
                                ga_best_losses = np.min(ga_losses_per_sample, axis=1)
                                ga_agg_best_loss.append(ga_best_losses)
                # Load ES per-sample metrics from NPZ
                if os.path.exists(es_out):
                    with np.load(es_out, allow_pickle=True) as fes:
                        if 'per_sample_accuracy' in fes:
                            es_agg_acc.append(np.array(fes['per_sample_accuracy']).reshape(-1))
                        if 'per_sample_shape_accuracy' in fes:
                            es_agg_shape.append(np.array(fes['per_sample_shape_accuracy']).reshape(-1))
                        if 'per_sample_pixel_correctness' in fes:
                            es_agg_pixel.append(np.array(fes['per_sample_pixel_correctness']).reshape(-1))
                        if 'es_generation_losses_per_sample' in fes:
                            es_losses_per_sample = np.array(fes['es_generation_losses_per_sample'])
                            if es_losses_per_sample.size > 0:
                                es_best_losses = np.min(es_losses_per_sample, axis=1)
                                es_agg_best_loss.append(es_best_losses)
            except Exception as _agg_e:
                print(f"[aggregate] Skipped aggregation for run {run_idx}: {_agg_e}")

        print(f"\n✅ Completed {args.n_samples} runs in group: {group_name}")

        # Generate aggregated statistics if requested
        if args.aggregate_statistics:
            try:
                print(f"\n📊 Generating aggregated statistics across {args.n_samples} runs...")
                
                # Concatenate across runs if available
                def _concat(lst):
                    return np.concatenate(lst, axis=0) if lst else np.array([])
                ga_acc = _concat(ga_agg_acc)
                ga_shp = _concat(ga_agg_shape)
                ga_pix = _concat(ga_agg_pixel)
                ga_bl = _concat(ga_agg_best_loss)
                es_acc = _concat(es_agg_acc)
                es_shp = _concat(es_agg_shape)
                es_pix = _concat(es_agg_pixel)
                es_bl = _concat(es_agg_best_loss)

                # Save aggregated NPZs (so we can reuse existing plotting function)
                agg_ga_npz = os.path.join(args.out_dir, "ga_aggregated.npz")
                agg_es_npz = os.path.join(args.out_dir, "es_aggregated.npz")
                if ga_acc.size or ga_shp.size or ga_pix.size or ga_bl.size:
                    np.savez_compressed(agg_ga_npz,
                        per_sample_accuracy=ga_acc if ga_acc.size else np.array([]),
                        per_sample_shape_accuracy=ga_shp if ga_shp.size else np.array([]),
                        per_sample_pixel_correctness=ga_pix if ga_pix.size else np.array([]),
                        ga_losses_per_sample=np.array([ga_bl]) if ga_bl.size else np.array([])  # Reshape for compatibility
                    )
                else:
                    # ensure files exist
                    np.savez_compressed(agg_ga_npz)
                if es_acc.size or es_shp.size or es_pix.size or es_bl.size:
                    np.savez_compressed(agg_es_npz,
                        per_sample_accuracy=es_acc if es_acc.size else np.array([]),
                        per_sample_shape_accuracy=es_shp if es_shp.size else np.array([]),
                        per_sample_pixel_correctness=es_pix if es_pix.size else np.array([]),
                        es_generation_losses_per_sample=np.array([es_bl]) if es_bl.size else np.array([])  # Reshape for compatibility
                    )
                else:
                    np.savez_compressed(agg_es_npz)

                # Dataset length for aggregated plot is total number of per-sample entries if present
                agg_len = int(max(ga_acc.size, es_acc.size, ga_shp.size, es_shp.size, ga_pix.size, es_pix.size, ga_bl.size, es_bl.size))
                
                # Create aggregated statistical histograms - skip if no_files is set
                if args.no_files:
                    print("🚫 File generation disabled - skipping aggregated statistical histograms")
                    agg_plot = None
                else:
                    agg_plot = create_statistical_histograms(agg_ga_npz, agg_es_npz, args.out_dir, dataset_length=agg_len if agg_len > 1 else 2)
                    if agg_plot:
                        print(f"[aggregate] Saved aggregated statistical histograms to {agg_plot}")
                    else:
                        print(f"[aggregate] Failed to generate aggregated statistical histograms")
                
                # Create aggregated W&B run (regardless of no_files flag)
                try:
                    import wandb
                    
                    # Create aggregated run name with "agg_" prefix
                    if args.run_name:
                        agg_run_name = f"agg_{args.run_name}-{int(time.time())}"
                    else:
                        agg_run_name = f"agg_latent-search-b{args.budget}-{int(time.time())}"
                    
                    # Create aggregated run configuration
                    agg_cfg = {
                        "artifact_path": args.wandb_artifact_path,
                        "budget": args.budget,
                        "ga_steps": ga_steps,
                        "ga_lr": args.ga_lr,
                        "es_population": pop,
                        "es_generations": gens,
                        "es_mutation_std": args.es_mutation_std,
                        "es_mutation_decay": mutation_decay,
                        "es_elite_size": elite_size,
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
                        "run_name": agg_run_name,
                        "dataset_folder": args.dataset_folder,
                        "n_samples": args.n_samples,
                        "aggregated": True,
                        "total_samples": agg_len,
                        "ga_return_code": 0,  # Aggregated runs don't have return codes
                        "es_return_code": 0,
                        "latent_dimension": None,  # Will be updated after plotting
                    }
                    
                    # Start aggregated W&B run
                    agg_run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=agg_run_name,
                        config=agg_cfg,
                        group=group_name
                    )
                    print(f"[wandb] Started aggregated run: {agg_run_name}")
                    
                    # Log aggregated metrics
                    if ga_acc.size > 0:
                        wandb.log({
                            "aggregated/ga_accuracy_mean": float(np.mean(ga_acc)),
                            "aggregated/ga_accuracy_std": float(np.std(ga_acc)),
                            "aggregated/ga_shape_correctness_mean": float(np.mean(ga_shp)),
                            "aggregated/ga_shape_correctness_std": float(np.std(ga_shp)),
                            "aggregated/ga_pixel_correctness_mean": float(np.mean(ga_pix)),
                            "aggregated/ga_pixel_correctness_std": float(np.std(ga_pix)),
                        })
                    if ga_bl.size > 0:
                        wandb.log({
                            "aggregated/ga_best_loss_mean": float(np.mean(ga_bl)),
                            "aggregated/ga_best_loss_std": float(np.std(ga_bl)),
                        })
                    if es_acc.size > 0:
                        wandb.log({
                            "aggregated/es_accuracy_mean": float(np.mean(es_acc)),
                            "aggregated/es_accuracy_std": float(np.std(es_acc)),
                            "aggregated/es_shape_correctness_mean": float(np.mean(es_shp)),
                            "aggregated/es_shape_correctness_std": float(np.std(es_shp)),
                            "aggregated/es_pixel_correctness_mean": float(np.mean(es_pix)),
                            "aggregated/es_pixel_correctness_std": float(np.std(es_pix)),
                        })
                    if es_bl.size > 0:
                        wandb.log({
                            "aggregated/es_best_loss_mean": float(np.mean(es_bl)),
                            "aggregated/es_best_loss_std": float(np.std(es_bl)),
                        })
                    
                    # Upload aggregated plots and NPZ files
                    upload_to_wandb(args.wandb_project, args.wandb_entity, agg_cfg, agg_ga_npz, agg_es_npz, 
                                  None, None, agg_plot, group_name, existing_run=agg_run)
                    
                    # Finish aggregated run
                    agg_run.finish()
                    print(f"[wandb] Finished aggregated run: {agg_run_name}")
                    
                except Exception as e:
                    print(f"[aggregate] Failed to create aggregated W&B run: {e}")
            except Exception as _agg_plot_e:
                print(f"[aggregate] Aggregation failed: {_agg_plot_e}")
        
    else:
        # Single run (original behavior)
        # Start W&B run at the beginning
        cfg = {
            "artifact_path": args.wandb_artifact_path,
            "budget": args.budget,
            "ga_steps": ga_steps,
            "ga_lr": args.ga_lr,
            "es_population": pop,
            "es_generations": gens,
            "es_mutation_std": args.es_mutation_std,
            "es_mutation_decay": mutation_decay,
            "es_elite_size": elite_size,
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
            "run_name": args.run_name,  # Custom run name if provided
            "dataset_folder": args.dataset_folder,  # Dataset folder name
            "latent_dimension": None,  # Will be updated after plotting
        }
        
        try:
            # Initialize W&B run at the start
            import wandb
            
            # Use custom run name if provided, otherwise use default naming
            if args.run_name:
                run_name = f"{args.run_name}-{int(time.time())}"
            else:
                run_name = f"latent-search-b{args.budget}-{int(time.time())}"
            
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=cfg
            )
            print(f"[wandb] Started single run")
            print(f"[wandb] Run name: {run_name}")
            
            # Log run start with dataset info
            wandb.log({
                "run_status": "started",
                "run_name": run_name,
                "dataset_folder": args.dataset_folder,
                "dataset_length": args.dataset_length
            })
            
        except Exception as e:
            print(f"Failed to start wandb run: {e}")
            run = None
        
        # Plot (with automatic PCA if latent dimension > 2) - skip if no_files is set
        if args.no_files:
            print("🚫 File generation disabled - skipping plots and file saves")
            trajectory_plot, loss_plot, stats_plot = None, None, None
            # Try to get latent dimension from existing files if available
            latent_dim = 2  # Default fallback
            try:
                if os.path.exists(ga_out):
                    with np.load(ga_out, allow_pickle=True) as f:
                        if 'ga_latents' in f:
                            ga_latents = np.array(f['ga_latents'])
                            if ga_latents.size > 0:
                                latent_dim = ga_latents.shape[-1]
                                print(f"📊 Extracted latent dimension from existing GA file: {latent_dim}")
            except Exception as e:
                print(f"⚠️  Could not extract latent dimension: {e}")
        else:
            trajectory_plot, loss_plot, stats_plot, latent_dim = plot_and_save(ga_out, es_out, args.out_dir, 
                                                      background_resolution=args.background_resolution,
                                                      background_smoothing=args.background_smoothing,
                                                      background_knn=args.background_knn,
                                                      background_bandwidth_scale=args.background_bandwidth_scale,
                                                      background_global_mix=args.background_global_mix,
                                                      ga_steps=ga_steps, es_population=pop, es_generations=gens, dataset_length=args.dataset_length)
            if trajectory_plot:
                print(f"Saved trajectory plot to {trajectory_plot}")
            if loss_plot:
                print(f"Saved loss curves plot to {loss_plot}")
            if stats_plot:
                print(f"Saved statistical histograms to {stats_plot}")
            else:
                print(f"No statistical histograms created (dataset_length={args.dataset_length})")

        print(f"📊 Latent space dimension: {latent_dim}")

        # Log plotting completion
        if run is not None:
            wandb.log({"plotting_status": "completed", "latent_dimension": latent_dim})

        # Upload final results to W&B
        try:
            if run is not None:
                # Add return codes and latent dimension to config for final upload
                cfg.update({
                    "ga_return_code": ga_rc,
                    "es_return_code": es_rc,
                    "latent_dimension": latent_dim,  # Latent space dimension
                })
                
                # Debug: show plot paths before upload
                print(f"[debug] Plot paths before upload:")
                print(f"[debug] trajectory_plot: {trajectory_plot}")
                print(f"[debug] loss_plot: {loss_plot}")
                print(f"[debug] trajectory_plot exists: {os.path.exists(trajectory_plot) if trajectory_plot else False}")
                print(f"[debug] loss_plot exists: {os.path.exists(loss_plot) if loss_plot else False}")
                
                # Upload artifacts and final metrics
                upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot, stats_plot, existing_run=run)
            else:
                # Fallback: create new run for upload
                upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, trajectory_plot, loss_plot, stats_plot)
        except Exception as e:
            print(f"Failed to upload to wandb: {e}")
        
        # Finish the W&B run
        if run is not None:
            run.finish()
            print(f"[wandb] Finished single run")


if __name__ == "__main__":
    main()


