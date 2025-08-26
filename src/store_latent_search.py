#!/usr/bin/env python3
"""
Run gradient_ascent and evolutionary_search once on a single W&B model artifact and
store latent search trajectories to NPZ. Then plot both trajectories (2D latents) and
upload NPZs and plot to W&B.
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


def _splat_background(P: np.ndarray, V: np.ndarray, xlim, ylim, n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Gaussian splat, sigma picked from median nn distance
    xv = np.linspace(xlim[0], xlim[1], n)
    yv = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xv, yv)

    if len(P) == 1:
        sigma = 0.1 * max(xlim[1]-xlim[0], ylim[1]-ylim[0])
    else:
        d2 = ((P[None, :, :] - P[:, None, :]) ** 2).sum(-1)
        nn = np.sqrt(np.partition(d2 + np.eye(len(P))*1e12, 1, axis=1)[:, 1])
        sigma = float(np.median(nn) + 1e-9)

    Xg = XX[..., None] - P[None, None, :, 0]
    Yg = YY[..., None] - P[None, None, :, 1]
    W = np.exp(-0.5 * (Xg*Xg + Yg*Yg) / (sigma * sigma)) + 1e-12   # [n,n,N]
    num = (W * V[None, None, :]).sum(axis=-1)
    den = W.sum(axis=-1)
    Z = num / den
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


def plot_and_save(ga_npz_path: str, es_npz_path: str, out_dir: str, field_name: str = "loss") -> Optional[str]:
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
        XX, YY, ZZ = _splat_background(P, V, xlim, ylim, n=240)
        im = ax.pcolormesh(XX, YY, ZZ, shading="auto", cmap=cmap, norm=norm, zorder=0, alpha=0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(field_name)
    else:
        ax.set_facecolor("white")

    # ES population (alpha 0.7), colored by losses if we have them
    if es.pop_pts is not None:
        if es.pop_vals is not None and len(es.pop_vals) == len(es.pop_pts):
            ax.scatter(es.pop_pts[:, 0], es.pop_pts[:, 1],
                       c=orient(np.asarray(es.pop_vals)), cmap=cmap, norm=norm,
                       s=16, alpha=0.7, linewidths=0, zorder=1, label="ES population")
        else:
            ax.scatter(es.pop_pts[:, 0], es.pop_pts[:, 1], s=16, alpha=0.7,
                       color="#777777", linewidths=0, zorder=1, label="ES population")

    # ES selected path (best per generation if present, otherwise es.pts)
    es_sel = es.best_per_gen if es.best_per_gen is not None else es.pts
    if es_sel is not None and len(es_sel) > 1:
        _plot_traj(ax, es_sel, color="#ff7f0e", label="ES selected", alpha=1.0)

    # GA path
    if ga.pts is not None and len(ga.pts) > 1:
        _plot_traj(ax, ga.pts, color="#e91e63", label="GA path", alpha=1.0)

    ax.legend(loc="upper right", frameon=True, fontsize=9)
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
    parser = argparse.ArgumentParser(description="Store and plot latent search trajectories (GA & ES)")
    parser.add_argument("--wandb_artifact_path", required=True, type=str)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--ga_lr", type=float, default=0.5)
    parser.add_argument("--es_mutation_std", type=float, default=0.5)
    parser.add_argument("--use_subspace_mutation", action="store_true")
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--ga_step_length", type=float, default=0.5)
    parser.add_argument("--trust_region_radius", type=float, default=None)
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
    ga_steps = int(math.ceil(args.budget / 2))
    ga_out = os.path.join(args.out_dir, "ga_latents.npz")
    ga_cmd = [
        sys.executable, "src/evaluate_checkpoint.py",
        "-w", args.wandb_artifact_path,
        "-i", "gradient_ascent",
        "--num-steps", str(ga_steps),
        "--lr", str(args.ga_lr),
        "--no-wandb-run", "true",
        "--store-latents", ga_out,
    ] + src_args
    print("Running:", " ".join(ga_cmd))
    ga_rc = subprocess.run(ga_cmd, check=False).returncode
    print(f"GA return code: {ga_rc}")

    # Evolutionary Search config
    pop = max(3, min(32, int(round(math.sqrt(args.budget)))))
    gens = max(1, int(math.ceil(args.budget / pop)))
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
    ] + src_args
    if args.use_subspace_mutation:
        es_cmd += ["--use-subspace-mutation", "--subspace-dim", str(args.subspace_dim), "--ga-step-length", str(args.ga_step_length)]
        if args.trust_region_radius is not None:
            es_cmd += ["--trust-region-radius", str(args.trust_region_radius)]
    print("Running:", " ".join(es_cmd))
    es_rc = subprocess.run(es_cmd, check=False).returncode
    print(f"ES return code: {es_rc}")

    # Plot
    plot_path = plot_and_save(ga_out, es_out, args.out_dir)
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
        "ga_return_code": ga_rc,
        "es_return_code": es_rc,
    }
    try:
        upload_to_wandb(args.wandb_project, args.wandb_entity, cfg, ga_out, es_out, plot_path)
    except Exception as e:
        print(f"Failed to upload to wandb: {e}")


if __name__ == "__main__":
    main()


