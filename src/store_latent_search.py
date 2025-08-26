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
    # Prefer known keys
    preferred = [
        f"{prefix}latents",
        f"{prefix}all_latents",
        f"{prefix}best_latents_per_generation",
    ]
    for key in preferred:
        if key in npz:
            arr = np.array(npz[key])
            if arr.ndim >= 2 and arr.shape[-1] == 2:
                return arr.reshape(-1, 2)
    # Fallback: first array with last-dim=2
    for key in npz.files:
        if not key.startswith(prefix):
            continue
        arr = np.array(npz[key])
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            return arr.reshape(-1, 2)
    return None


def plot_and_save(ga_npz_path: str, es_npz_path: str, out_dir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting unavailable: {e}")
        return None

    ga_pts = None
    es_pts = None
    if os.path.exists(ga_npz_path):
        with np.load(ga_npz_path, allow_pickle=True) as f:
            ga_pts = try_extract_2d_points(f, "ga_")
    if os.path.exists(es_npz_path):
        with np.load(es_npz_path, allow_pickle=True) as f:
            es_pts = try_extract_2d_points(f, "es_")

    if ga_pts is None and es_pts is None:
        print("No 2D latent arrays found to plot.")
        return None

    plt.figure(figsize=(8, 6))
    if ga_pts is not None and len(ga_pts) > 0:
        c = np.linspace(0, 1, len(ga_pts))
        plt.scatter(ga_pts[:, 0], ga_pts[:, 1], c=c, cmap="viridis", s=14, alpha=0.7, label="GA path")
        plt.scatter(ga_pts[0, 0], ga_pts[0, 1], c="green", s=60, marker="o", edgecolors="black", linewidths=1.0, label="GA start")
        plt.scatter(ga_pts[-1, 0], ga_pts[-1, 1], c="red", s=60, marker="s", edgecolors="black", linewidths=1.0, label="GA end")
    if es_pts is not None and len(es_pts) > 0:
        c = np.linspace(0, 1, len(es_pts))
        plt.scatter(es_pts[:, 0], es_pts[:, 1], c=c, cmap="plasma", s=16, alpha=0.8, marker="^", label="ES path")
        plt.scatter(es_pts[0, 0], es_pts[0, 1], c="blue", s=60, marker="^", edgecolors="black", linewidths=1.0, label="ES start")
        plt.scatter(es_pts[-1, 0], es_pts[-1, 1], c="purple", s=60, marker="D", edgecolors="black", linewidths=1.0, label="ES end")

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent search trajectories (GA vs ES)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "search_trajectories.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


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


