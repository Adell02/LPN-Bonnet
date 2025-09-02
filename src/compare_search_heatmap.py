#!/usr/bin/env python3
"""Generate search heatmaps across checkpoints.

This script evaluates gradient ascent (GA) and evolutionary search (ES)
for multiple checkpoints from a Weights & Biases run. For each checkpoint,
we run a search with a fixed evaluation budget, extract intermediate losses
and accuracies, and build heatmaps showing how performance evolves across
training checkpoints and search budgets. Differential heatmaps highlight the
performance gap between ES and GA using `visualize_loss_difference_heatmap`.

The script keeps things simple: checkpoints are fetched via the W&B API,
`evaluate_checkpoint.py` is invoked in a subprocess for GA and ES, and the
resulting trajectory files (NPZ) are parsed to obtain per-budget metrics.
Heatmaps are uploaded to a single W&B run as checkpoints are processed.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import wandb
from matplotlib import pyplot as plt

from visualization import visualize_loss_difference_heatmap
from store_latent_search import _extract_vals  # reuse robust NPZ parsing helpers


# ----------------------------------------------------------------------------
# Helper data structures
# ----------------------------------------------------------------------------

@dataclass
class TrajectoryData:
    budgets: np.ndarray
    losses: np.ndarray
    accuracies: Optional[np.ndarray]


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def build_dataset_args(args: argparse.Namespace) -> List[str]:
    """Build dataset-related CLI arguments for evaluate_checkpoint.py."""
    cli: List[str] = []
    if args.json_challenges and args.json_solutions:
        cli += ["-jc", args.json_challenges, "-js", args.json_solutions]
    elif args.dataset_folder:
        cli += ["-d", args.dataset_folder]
        if args.dataset_length is not None:
            cli += ["--dataset-length", str(args.dataset_length)]
        if args.dataset_batch_size is not None:
            cli += ["--dataset-batch-size", str(args.dataset_batch_size)]
        cli += ["--dataset-use-hf", str(args.dataset_use_hf).lower()]
        cli += ["--dataset-seed", str(args.dataset_seed)]
    return cli


def get_all_checkpoints(run_name: str, project: str, entity: str) -> List[Dict]:
    """Return metadata for all checkpoint artifacts of a W&B run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_name}")
    checkpoints: List[Dict] = []
    for artifact in run.logged_artifacts():
        if "checkpoint" not in artifact.name.lower():
            continue
        step = None
        if "--checkpoint" in artifact.name:
            name_part = artifact.name.split("--checkpoint")[0]
            nums = [int(n) for n in name_part.split("-") if n.isdigit()]
            if nums:
                step = nums[-1]
        checkpoints.append({"artifact": artifact, "name": artifact.name, "step": step})
    checkpoints.sort(key=lambda x: x["step"] if x["step"] is not None else -1)
    return checkpoints


def _extract_accuracy(npz: np.lib.npyio.NpzFile, prefix: str) -> Optional[np.ndarray]:
    """Best-effort extraction of accuracy arrays from trajectory files."""
    for key in npz.files:
        if key.startswith(prefix) and "accuracy" in key:
            arr = np.array(npz[key]).reshape(-1)
            return arr
    return None


def run_method(
    artifact_path: str,
    method: str,
    method_kwargs: Dict[str, float],
    dataset_args: Iterable[str],
) -> TrajectoryData:
    """Invoke evaluate_checkpoint.py for a single method and return trajectory data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = os.path.join(tmpdir, f"{method}.npz")
        cmd = [
            sys.executable,
            "src/evaluate_checkpoint.py",
            "-w",
            artifact_path,
            "-i",
            method,
            "--no-wandb-run",
            "true",
            "--store-latents",
            traj_path,
            "--track-progress",
        ]
        cmd.extend(dataset_args)

        if method == "gradient_ascent":
            cmd += ["--num-steps", str(method_kwargs.get("num_steps", 100))]
            cmd += ["--lr", str(method_kwargs.get("lr", 0.5))]
        elif method == "evolutionary_search":
            cmd += ["--population-size", str(method_kwargs.get("population_size", 32))]
            cmd += ["--num-generations", str(method_kwargs.get("num_generations", 25))]
            cmd += ["--mutation-std", str(method_kwargs.get("mutation_std", 0.5))]
            if method_kwargs.get("mutation_decay") is not None:
                cmd += ["--mutation-decay", str(method_kwargs["mutation_decay"])]
            if method_kwargs.get("elite_size") is not None:
                cmd += ["--elite-size", str(method_kwargs["elite_size"])]
        else:
            raise ValueError(f"Unsupported method: {method}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{method} failed: {result.stderr}")

        npz = np.load(traj_path)
        budgets = np.array(npz.get(f"{method[:2]}_budget", np.arange(len(_extract_vals(npz, f"{method[:2]}_")))))
        losses = _extract_vals(npz, f"{method[:2]}_")
        accuracies = _extract_accuracy(npz, f"{method[:2]}_")
        return TrajectoryData(budgets=budgets, losses=losses, accuracies=accuracies)


# ----------------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------------

def simple_heatmap(data: np.ndarray, steps: np.ndarray, budgets: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=[steps[0], steps[-1], budgets[0], budgets[-1]],
    )
    ax.set_xlabel("Training Checkpoint")
    ax.set_ylabel("Search Budget")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GA and ES across checkpoints with heatmaps")
    parser.add_argument("--run_name", required=True, help="W&B run name")
    parser.add_argument("--project", default="LPN-ARC", help="W&B project name")
    parser.add_argument("--entity", default="ga624-imperial-college-london", help="W&B entity")
    parser.add_argument("--budget", type=int, default=500, help="Total evaluation budget")
    parser.add_argument("--json_challenges", type=str, default=None)
    parser.add_argument("--json_solutions", type=str, default=None)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--dataset_length", type=int, default=None)
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument("--dataset_use_hf", type=bool, default=True)
    parser.add_argument("--dataset_seed", type=int, default=0)
    args = parser.parse_args()

    dataset_args = build_dataset_args(args)
    checkpoints = get_all_checkpoints(args.run_name, args.project, args.entity)
    if not checkpoints:
        print("No checkpoints found.")
        return

    run = wandb.init(project=args.project, entity=args.entity, name=f"compare_search_heatmap::{args.run_name}")

    ga_losses_all: List[np.ndarray] = []
    es_losses_all: List[np.ndarray] = []
    ga_acc_all: List[np.ndarray] = []
    es_acc_all: List[np.ndarray] = []

    # Use common budget grid for interpolation
    common_budgets = np.linspace(0, args.budget, num=11)

    for idx, cp in enumerate(checkpoints, 1):
        artifact_path = f"{args.entity}/{args.project}/{cp['name']}"
        print(f"\nProcessing checkpoint {idx}/{len(checkpoints)}: {cp['name']}")

        ga_data = run_method(
            artifact_path,
            "gradient_ascent",
            {"num_steps": args.budget // 2, "lr": 0.5},
            dataset_args,
        )
        es_data = run_method(
            artifact_path,
            "evolutionary_search",
            {"population_size": 50, "num_generations": 10, "mutation_std": 0.5, "mutation_decay": 0.95},
            dataset_args,
        )

        ga_losses_interp = np.interp(common_budgets, ga_data.budgets, ga_data.losses)
        es_losses_interp = np.interp(common_budgets, es_data.budgets, es_data.losses)
        ga_losses_all.append(ga_losses_interp)
        es_losses_all.append(es_losses_interp)

        if ga_data.accuracies is not None:
            ga_acc_interp = np.interp(common_budgets[: len(ga_data.accuracies)], ga_data.budgets[: len(ga_data.accuracies)], ga_data.accuracies)
        else:
            ga_acc_interp = np.full_like(common_budgets, np.nan)
        if es_data.accuracies is not None:
            es_acc_interp = np.interp(common_budgets[: len(es_data.accuracies)], es_data.budgets[: len(es_data.accuracies)], es_data.accuracies)
        else:
            es_acc_interp = np.full_like(common_budgets, np.nan)
        ga_acc_all.append(ga_acc_interp)
        es_acc_all.append(es_acc_interp)

        steps = np.array([c["step"] or i for i, c in enumerate(checkpoints[: idx])])
        ga_losses_mat = np.stack(ga_losses_all)
        es_losses_mat = np.stack(es_losses_all)
        loss_diff = es_losses_mat - ga_losses_mat
        ga_acc_mat = np.stack(ga_acc_all)
        es_acc_mat = np.stack(es_acc_all)
        acc_diff = es_acc_mat - ga_acc_mat

        figs = {
            "ga_loss_heatmap": simple_heatmap(ga_losses_mat, steps, common_budgets, "GA Loss"),
            "es_loss_heatmap": simple_heatmap(es_losses_mat, steps, common_budgets, "ES Loss"),
            "loss_diff_heatmap": visualize_loss_difference_heatmap(steps, common_budgets, loss_diff.T, "GA", "ES"),
            "ga_acc_heatmap": simple_heatmap(ga_acc_mat, steps, common_budgets, "GA Accuracy"),
            "es_acc_heatmap": simple_heatmap(es_acc_mat, steps, common_budgets, "ES Accuracy"),
            "acc_diff_heatmap": visualize_loss_difference_heatmap(steps, common_budgets, acc_diff.T, "GA", "ES"),
        }

        wandb.log({name: wandb.Image(fig) for name, fig in figs.items()})
        for fig in figs.values():
            plt.close(fig)

    run.finish()


if __name__ == "__main__":
    main()