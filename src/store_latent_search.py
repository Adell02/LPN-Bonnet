#!/usr/bin/env python3
"""
Run gradient_ascent and evolutionary_search once on a single W&B model artifact and
store latent search trajectories to NPZ. Then plot both trajectories (2D latents) and
upload NPZs and plot to W&B.

Example usage:

# Basic usage with automatic budget-based calculations:
python src/store_latent_search.py --wandb_artifact_path "entity/project/artifact:v0" --budget 100

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
"""

import argparse
import math
import os
import sys
import subprocess
from typing import Optional, Tuple

import numpy as np
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
    
    # Move T to position -2 so final shape is (..., T, D)
    order = [i for i in range(arr.ndim) if i != t_axis and i != arr.ndim - 1] + [t_axis, arr.ndim - 1]
    arr = np.transpose(arr, order)
    
    # Now average over all leading axes except T and D
    lead_axes = tuple(range(arr.ndim - 2))
    if lead_axes:
        arr = arr.mean(axis=lead_axes)
    
    # arr is now (T, D)
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
    
    # Apply unified PCA if needed
    if original_dim > 2:
        print(f"[plot] Original latent dimension: {original_dim}D, applying unified PCA to project to 2D")
        
        # Fit PCA on all data combined with whitening to prevent stretching
        # Whitening ensures PC1 and PC2 have similar variance, preventing the plot from looking like a pancake
        pca_transformer = _fit_unified_pca(all_points, target_dim=2, whiten=True)
        
        # Apply the same PCA transformation to all arrays
        if ga.pts is not None:
            ga.pts = _apply_fitted_pca(ga.pts, pca_transformer, target_dim=2)
        if es.pts is not None:
            es.pts = _apply_fitted_pca(es.pts, pca_transformer, target_dim=2)
        if es.pop_pts is not None:
            es.pop_pts = _apply_fitted_pca(es.pop_pts, pca_transformer, target_dim=2)
        if es.best_per_gen is not None:
            es.best_per_gen = _apply_fitted_pca(es.best_per_gen, pca_transformer, target_dim=2)
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

    # collect samples for the soft heatmap
    bgP, bgV = [], []
    
    # GA path values
    if ga.pts is not None and ga.vals is not None:
        # Flatten GA points to 2D and ensure vals match
        ga_pts_flat = ga.pts.reshape(-1, 2)
        ga_vals_flat = ga.vals.reshape(-1)
        if len(ga_pts_flat) == len(ga_vals_flat):
            print(f"[plot] GA data: pts={ga_pts_flat.shape}, vals={ga_vals_flat.shape}")
            bgP.append(ga_pts_flat)
            bgV.append(ga_vals_flat)
        else:
            print(f"[plot] GA data mismatch: pts={ga_pts_flat.shape}, vals={ga_vals_flat.shape}")
    else:
        print(f"[plot] GA data missing: pts={ga.pts is not None}, vals={ga.vals is not None}")
    
    # ES population values
    if es.pop_pts is not None and es.pop_vals is not None:
        # Flatten ES population points to 2D and ensure vals match
        es_pop_pts_flat = es.pop_pts.reshape(-1, 2)
        es_pop_vals_flat = es.pop_vals.reshape(-1)
        print(f"[plot] ES debug: pop_pts={es.pop_pts.shape}, pop_vals={es.pop_vals.shape}")
        print(f"[plot] ES debug: flattened pts={es_pop_pts_flat.shape}, vals={es_pop_vals_flat.shape}")
        if len(es_pop_pts_flat) == len(es_pop_vals_flat):
            print(f"[plot] ES data: pts={es_pop_pts_flat.shape}, vals={es_pop_vals_flat.shape}")
            bgP.append(es_pop_pts_flat)
            bgV.append(es_pop_vals_flat)
        else:
            print(f"[plot] ES data mismatch: pts={es_pop_pts_flat.shape}, vals={es_pop_vals_flat.shape}")
            print(f"[plot] ES mismatch details: pts length {len(es_pop_pts_flat)}, vals length {len(es_pop_vals_flat)}")
    else:
        print(f"[plot] ES data missing: pts={es.pop_pts is not None}, vals={es.pop_vals is not None}")
    
    # if nothing, background stays white
    have_field = len(bgP) > 0
    print(f"[plot] Background field available: {have_field} (bgP={len(bgP)}, bgV={len(bgV)})")
    
    # Debug: show what we're working with
    if have_field:
        for i, (pts, vals) in enumerate(zip(bgP, bgV)):
            print(f"[plot] Background {i}: pts={pts.shape}, vals={vals.shape}")
            if len(pts) != len(vals):
                print(f"[plot] WARNING: Background {i} has mismatched lengths!")

    def orient(v: np.ndarray) -> np.ndarray:
        return -v if field_name.lower() == "score" else v

    # normalization across everything we will color
    all_for_norm = []
    if have_field:
        all_for_norm.append(orient(np.concatenate(bgV)))
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
    ax.grid(True, alpha=0.25)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # soft heatmap background by splatting losses if available
    if have_field:
        P = np.concatenate(bgP, axis=0)
        V = orient(np.concatenate(bgV, axis=0))
        XX, YY, ZZ = _splat_background(
            P, V, xlim, ylim, 
            n=background_resolution, 
            enable_smoothing=background_smoothing,
            knn_k=background_knn,
            bandwidth_scale=background_bandwidth_scale,
            global_mix=background_global_mix
        )
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
            generation_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
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
    es_sel = es.best_per_gen if es.best_per_gen is not None else es.pts
    if es_sel is not None and len(es_sel) > 1:
        # Flatten ES selected path for plotting
        es_sel_flat = es_sel.reshape(-1, 2)
        _plot_traj(ax, es_sel_flat, color="#ff7f0e", label="ES selected", alpha=1.0)

    # GA path
    if ga.pts is not None and len(ga.pts) > 1:
        # Flatten GA path for plotting
        ga_pts_flat = ga.pts.reshape(-1, 2)
        _plot_traj(ax, ga_pts_flat, color="#e91e63", label="GA path", alpha=1.0)

    # Create comprehensive legend with all elements
    legend_elements = []
    
    # ES population (all samples in orange)
    if es.pop_pts is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                                       markersize=10, alpha=1.0, label='ES population (all samples)'))
    
    # Generation clusters (general representation)
    if es.pop_pts is not None and es.gen_idx is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='#1f77b4', markerfacecolor='#1f77b4', 
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
    """Generate a plot comparing loss curves for GA and ES methods with budget on x-axis."""
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
    ax.set_xlabel("Budget (2×steps for GA, pop×gen for ES)")
    ax.set_ylabel("Loss (lower is better)")
    ax.grid(True, alpha=0.3)
    
    # Plot GA loss curve with budget on x-axis
    if has_ga_loss:
        if ga_steps is not None:
            # Use actual budget: 2 × steps
            ga_budget = 2 * np.arange(len(ga.vals))
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
            # Use actual budget: population × generations
            es_budget = np.arange(es_generations + 1) * es_population  # +1 to include initial
            if len(es_budget) == len(es.vals):
                ax.plot(es_budget, es.vals, color="#ff7f0e", linewidth=2.5, marker='s', 
                        markersize=6, label=f"Evolutionary Search ({es_population}×{es_generations})", zorder=3)
            else:
                # Fallback if budget doesn't match values length
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
                    trajectory_plot: Optional[str], loss_plot: Optional[str]) -> None:
    try:
        import wandb
    except Exception as e:
        print(f"wandb not available: {e}")
        return
    run = wandb.init(project=project, entity=entity, name=f"latent-search-b{cfg.get('budget')}", config=cfg)
    if os.path.exists(ga_npz):
        ga_art = wandb.Artifact(name=f"ga_latents_b{cfg.get('budget')}", type="latent_trajectories")
        ga_art.add_file(ga_npz)
        run.log_artifact(ga_art)
    if os.path.exists(es_npz):
        es_art = wandb.Artifact(name=f"es_latents_b{cfg.get('budget')}", type="latent_trajectories")
        es_art.add_file(es_npz)
        run.log_artifact(es_art)
    if trajectory_plot and os.path.exists(trajectory_plot):
        wandb.log({"trajectory_plot": wandb.Image(trajectory_plot)})
    if loss_plot and os.path.exists(loss_plot):
        wandb.log({"loss_curves_plot": wandb.Image(loss_plot)})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store and plot latent search trajectories (GA & ES). "
        "Both methods start from the same mean latent for fair comparison. "
        "Automatically applies PCA to reduce latent dimensions > 2 to 2D for visualization. "
        "Use --ga_steps, --es_population, --es_generations to override automatic budget-based calculations."
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
    parser.add_argument("--dataset_length", type=int, default=None)
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument("--dataset_use_hf", type=str, default="true")
    parser.add_argument("--dataset_seed", type=int, default=0)
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


