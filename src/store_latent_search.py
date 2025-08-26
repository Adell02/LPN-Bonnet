#!/usr/bin/env python3
"""
Run gradient_ascent and evolutionary_search once on a single W&B model artifact and
store latent search trajectories to a lightweight NPZ file per method.

Usage:
  python src/store_latent_search.py \
    --wandb_artifact_path ENTITY/PROJECT/ARTIFACT:VERSION \
    --budget 50 \
    --ga_lr 0.5 \
    --es_mutation_std 0.5 \
    --use_subspace_mutation \
    --subspace_dim 32 \
    --ga_step_length 0.5 \
    --out_dir results/latent_traces

Notes:
  - For GA, budget refers to compute budget where num_steps ≈ ceil(budget/2)
  - For ES, we map budget to population_size × num_generations with population ≈ sqrt(budget)
"""

import argparse
import math
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Store latent search trajectories for GA and ES")
    parser.add_argument("--wandb_artifact_path", required=True, type=str)
    parser.add_argument("--budget", required=True, type=int, help="Compute budget (shared across methods)")
    parser.add_argument("--ga_lr", type=float, default=0.5)
    parser.add_argument("--es_mutation_std", type=float, default=0.5)
    parser.add_argument("--use_subspace_mutation", action="store_true")
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--ga_step_length", type=float, default=0.5)
    parser.add_argument("--trust_region_radius", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default="results/latent_traces")
    # Optional: pick either JSON or dataset if needed later
    parser.add_argument("--json_challenges", type=str, default=None)
    parser.add_argument("--json_solutions", type=str, default=None)
    parser.add_argument("--dataset_folder", type=str, default=None)
    parser.add_argument("--dataset_length", type=int, default=None)
    parser.add_argument("--dataset_batch_size", type=int, default=None)
    parser.add_argument("--dataset_use_hf", type=str, default="true")
    parser.add_argument("--dataset_seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Shared source validation: prefer JSON if both provided
    src_args = []
    if args.json_challenges and args.json_solutions:
        src_args += ["-jc", args.json_challenges, "-js", args.json_solutions]
    elif args.dataset_folder:
        src_args += ["-d", args.dataset_folder]
        if args.dataset_length is not None:
            src_args += ["--dataset-length", str(args.dataset_length)]
        if args.dataset_batch_size is not None:
            src_args += ["--dataset-batch-size", str(args.dataset_batch_size)]
        # dataset_use_hf is a required boolean-like flag; always pass it
        src_args += ["--dataset-use-hf", args.dataset_use_hf]
        # dataset_seed has a valid default (0); always pass it
        src_args += ["--dataset-seed", str(args.dataset_seed)]

    # 1) Gradient Ascent: num_steps ≈ ceil(budget/2)
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
    subprocess.run(ga_cmd, check=False)

    # 2) Evolutionary Search: population ≈ max(3, min(32, round(sqrt(budget)))) ; gens = ceil(budget / population)
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
        es_cmd += [
            "--use-subspace-mutation",
            "--subspace-dim", str(args.subspace_dim),
            "--ga-step-length", str(args.ga_step_length),
        ]
        if args.trust_region_radius is not None:
            es_cmd += ["--trust-region-radius", str(args.trust_region_radius)]

    print("Running:", " ".join(es_cmd))
    subprocess.run(es_cmd, check=False)

    print(f"Done. Saved GA to {ga_out} and ES to {es_out}")


if __name__ == "__main__":
    main()


