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
            if arr.ndim >= 2 and arr.shape[-1] == 2:
                print(f"[plot] Using key '{key}' for {prefix} points, shape={arr.shape}")
                return arr.reshape(-1, 2)
    # Fallback: first array with last-dim=2
    for key in npz.files:
        if not key.startswith(prefix):
            continue
        arr = np.array(npz[key])
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            return arr.reshape(-1, 2)
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
    ]:
        if k in npz:
            arr = np.array(npz[k]).reshape(-1)
            if arr.ndim == 1 and arr.size > 0:
                print(f"[plot] Using values key '{k}', length={arr.size}")
                return arr
    return None


def _extract_best_per_gen(npz, prefix: str) -> Optional[np.ndarray]:
    for k in [f"{prefix}best_latents_per_generation", f"{prefix}elite_latents"]:
        if k in npz:
            arr = np.array(npz[k])
            if arr.ndim >= 2 and arr.shape[-1] == 2:
                return arr.reshape(-1, 2)
    return None


def _extract_pop(npz, prefix: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    # also read ES population scores so we can make the soft heatmap
    pts = gens = vals = None
    if f"{prefix}all_latents" in npz:
        pts = np.array(npz[f"{prefix}all_latents"]).reshape(-1, 2)
    if f"{prefix}generation_idx" in npz:
        gens = np.array(npz[f"{prefix}generation_idx"]).reshape(-1)
    for k in (f"{prefix}all_scores", f"{prefix}all_losses"):
        if k in npz:
            vals = np.array(npz[k]).reshape(-1)
            break
    return pts, gens, vals


def _load_trace(npz_path: str, prefix: str) -> Trace:
    t = Trace()
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as f:
            t.pts = try_extract_2d_points(f, prefix)
            t.vals = _extract_vals(f, prefix)
            t.best_per_gen = _extract_best_per_gen(f, prefix)
            pop_pts, gen_idx, pop_vals = _extract_pop(f, prefix)
            t.pop_pts = pop_pts
            t.gen_idx = gen_idx
            t.pop_vals = pop_vals
    return t


def _nice_bounds(xy: np.ndarray, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    px = dx * pad if dx > 0 else 1.0
    py = dy * pad if dy > 0 else 1.0
    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def _splat_background(P: np.ndarray, V: np.ndarray, xlim, ylim, n: int = 240, 
                      enable_smoothing: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Adaptive resolution: increase grid density for small-scale searches
    search_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    if len(P) > 1:
        # Calculate typical step size from the data
        d2 = ((P[None, :, :] - P[:, None, :]) ** 2).sum(-1)
        nn = np.sqrt(np.partition(d2 + np.eye(len(P))*1e12, 1, axis=1)[:, 1])
        typical_step = float(np.median(nn) + 1e-9)
        
        # Increase resolution for small steps to create smoother gradients
        if typical_step < search_range * 0.01:  # Very small steps
            n = max(n, 800)  # High resolution
        elif typical_step < search_range * 0.05:  # Small steps
            n = max(n, 500)  # Medium-high resolution
        elif typical_step < search_range * 0.1:   # Medium steps
            n = max(n, 400)  # Medium resolution
    
    xv = np.linspace(xlim[0], xlim[1], n)
    yv = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xv, yv)

    if len(P) == 1:
        sigma = 0.1 * max(xlim[1]-xlim[0], ylim[1]-ylim[0])
    else:
        # Adaptive sigma: use larger radius to cover more space
        if typical_step < search_range * 0.01:
            # Very small steps: use larger sigma to cover more area
            sigma = max(typical_step * 4.0, search_range * 0.02)
        elif typical_step < search_range * 0.05:
            # Small steps: use larger sigma to cover more area
            sigma = max(typical_step * 3.0, search_range * 0.03)
        else:
            # Larger steps: use larger sigma for better coverage
            sigma = max(float(np.median(nn) * 2.0), search_range * 0.05)

    # Create smoother background with multiple passes
    Z = np.zeros_like(XX)
    
    # First pass: Gaussian splatting
    Xg = XX[..., None] - P[None, None, :, 0]
    Yg = YY[..., None] - P[:, None, :, 1]
    W = np.exp(-0.5 * (Xg*Xg + Yg*Yg) / (sigma * sigma)) + 1e-12
    num = (W * V[None, None, :]).sum(axis=-1)
    den = W.sum(axis=-1)
    Z = num / den
    
    # Second pass: Apply Gaussian smoothing for very small-scale searches
    if enable_smoothing and len(P) > 1 and typical_step < search_range * 0.02:
        from scipy.ndimage import gaussian_filter
        try:
            # Smooth the background to create more gradient-like appearance
            smoothing_sigma = max(1.0, n * typical_step / search_range)
            Z = gaussian_filter(Z, sigma=smoothing_sigma)
        except ImportError:
            # Fallback if scipy not available
            pass
    
    return XX, YY, Z


def _plot_traj(ax, pts: np.ndarray, color: str, label: str, arrow_every: int = 6, alpha: float = 1.0):
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8, alpha=alpha, label=label, zorder=5)
    for i in range(0, len(pts) - 1, max(1, arrow_every)):
        ax.annotate("", xy=pts[i+1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=color, shrinkA=0, shrinkB=0))
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=70, marker="o", edgecolors="black", linewidths=0.7,
               color=color, zorder=6, alpha=alpha)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=70, marker="s", edgecolors="black", linewidths=0.7,
               color=color, zorder=6, alpha=alpha)


def plot_and_save(ga_npz_path: str, es_npz_path: str, out_dir: str, field_name: str = "loss", 
                  background_resolution: int = 240, background_smoothing: bool = False) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except Exception as e:
        print(f"Plotting unavailable: {e}")
        return None

    ga = _load_trace(ga_npz_path, "ga_")
    es = _load_trace(es_npz_path, "es_")

    if ga.pts is None and es.pts is None and es.pop_pts is None:
        print("No 2D latent arrays found to plot.")
        return None

    # unified bounds
    pts_for_bounds = [p for p in [ga.pts, es.pts, es.pop_pts, es.best_per_gen] if p is not None]
    XY = np.concatenate(pts_for_bounds, axis=0)
    xlim, ylim = _nice_bounds(XY)

    # collect samples for the soft heatmap
    bgP, bgV = [], []
    # GA path values
    if ga.pts is not None and ga.vals is not None and len(ga.pts) == len(ga.vals):
        bgP.append(ga.pts); bgV.append(np.asarray(ga.vals))
    # ES population values
    if es.pop_pts is not None and es.pop_vals is not None and len(es.pop_pts) == len(es.pop_vals):
        bgP.append(es.pop_pts); bgV.append(np.asarray(es.pop_vals))
    # if nothing, background stays white
    have_field = len(bgP) > 0

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
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.5))
    ax.set_title("Latent search: GA and ES")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # soft heatmap background by splatting losses if available
    if have_field:
        P = np.concatenate(bgP, axis=0)
        V = orient(np.concatenate(bgV, axis=0))
        XX, YY, ZZ = _splat_background(P, V, xlim, ylim, n=background_resolution, enable_smoothing=background_smoothing)
        im = ax.pcolormesh(XX, YY, ZZ, shading="auto", cmap=cmap, norm=norm, zorder=0, alpha=0.7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_name)
    else:
        ax.set_facecolor("white")

    # ES population: color each generation differently for better visibility
    if es.pop_pts is not None and es.gen_idx is not None:
        # Define distinct colors for each generation (avoiding pink/red used by GA)
        generation_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot each generation with a different color
        unique_gens = np.unique(es.gen_idx)
        for gen in unique_gens:
            mask = es.gen_idx == gen
            gen_pts = es.pop_pts[mask]
            color = generation_colors[gen % len(generation_colors)]
            ax.scatter(gen_pts[:, 0], gen_pts[:, 1], s=24, alpha=0.7,
                       color=color, linewidths=0, zorder=1, 
                       label=f"ES gen {gen}" if gen < 3 else None)  # Only label first 3 gens to avoid clutter
    elif es.pop_pts is not None:
        # Fallback: if no generation info, use single color
        ax.scatter(es.pop_pts[:, 0], es.pop_pts[:, 1], s=24, alpha=0.7,
                   color="#ff7f0e", linewidths=0, zorder=1, label="ES population")

    # ES selected path (best per generation if present, otherwise es.pts)
    es_sel = es.best_per_gen if es.best_per_gen is not None else es.pts
    if es_sel is not None and len(es_sel) > 1:
        _plot_traj(ax, es_sel, color="#ff7f0e", label="ES selected", alpha=1.0)

    # GA path
    if ga.pts is not None and len(ga.pts) > 1:
        _plot_traj(ax, ga.pts, color="#e91e63", label="GA path", alpha=1.0)

    # Create comprehensive legend with all elements
    legend_elements = []
    
    # ES population generations (if available)
    if es.pop_pts is not None and es.gen_idx is not None:
        unique_gens = np.unique(es.gen_idx)
        generation_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, gen in enumerate(unique_gens[:3]):  # Show first 3 generations
            color = generation_colors[gen % len(generation_colors)]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                           markersize=10, alpha=0.7, label=f'ES gen {gen}'))
        if len(unique_gens) > 3:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', 
                                           markersize=10, alpha=0.7, label=f'ES gen {unique_gens[3:]}...'))
    else:
        # Fallback: single ES population color
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                                       markersize=10, alpha=0.7, label='ES population'))
    
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
    return png


def upload_to_wandb(project: str, entity: Optional[str], cfg: dict, ga_npz: str, es_npz: str, plot_path: Optional[str]) -> None:
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
    if plot_path and os.path.exists(plot_path):
        wandb.log({"trajectory_plot": wandb.Image(plot_path)})
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store and plot latent search trajectories (GA & ES). "
        "Both methods start from the same mean latent for fair comparison. "
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
    parser.add_argument("--background_resolution", type=int, default=240, help="Base resolution for background heatmap (higher = smoother)")
    parser.add_argument("--background_smoothing", action="store_true", help="Enable additional Gaussian smoothing for small-scale searches")
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

    # Plot
    plot_path = plot_and_save(ga_out, es_out, args.out_dir, 
                              background_resolution=args.background_resolution,
                              background_smoothing=args.background_smoothing)
    if plot_path:
        print(f"Saved plot to {plot_path}")

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
        "ga_return_code": ga_rc,
        "es_return_code": es_rc,
    }
    try:
        upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, plot_path)
    except Exception as e:
        print(f"Failed to upload to wandb: {e}")


if __name__ == "__main__":
    main()


