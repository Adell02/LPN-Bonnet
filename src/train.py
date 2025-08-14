from functools import partial
import json
import logging
import math
import os
from typing import Callable, Optional
import time
import numpy as np

import chex
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from matplotlib import pyplot as plt
import optax
from flax.serialization import from_bytes, msgpack_serialize, to_state_dict
from flax.training.train_state import TrainState
import tqdm
from tqdm.auto import trange as tqdm_trange
import wandb
import hydra
import omegaconf

import sys
print(sys.path)

from models.lpn import LPN
from models.utils import DecoderTransformerConfig, EncoderTransformerConfig
from evaluator import Evaluator
from models.transformer import EncoderTransformer, DecoderTransformer
from visualization import (
    visualize_dataset_generation,
    visualize_heatmap,
    visualize_tsne,
    visualize_json_submission,
    visualize_latents_samples,
    visualize_optimization_comparison  # NEW
)
from data_utils import (
    load_datasets,
    shuffle_dataset_into_batches,
    data_augmentation_fn,
    make_leave_one_out,
    DATASETS_BASE_PATH,
)
from datasets.task_gen.dataloader import make_task_gen_dataloader, make_dataset


logging.getLogger().setLevel(logging.INFO)


class Trainer:
    def __init__(self, cfg: omegaconf.DictConfig, model: LPN, num_devices: Optional[int] = None) -> None:
        if num_devices is None:
            self.num_devices = jax.device_count()
        else:
            self.num_devices = min(num_devices, jax.device_count())
        logging.info(f"Number of devices: {self.num_devices}")
        self.devices = jax.local_devices()[: self.num_devices]
        self.model = model

        self.batch_size = cfg.training.batch_size
        self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
        if self.batch_size % (self.gradient_accumulation_steps * self.num_devices) != 0:
            raise ValueError(
                f"Effective batch size {self.batch_size} is not divisible by the number of devices {self.num_devices} "
                f"times the number of gradient accumulation steps {self.gradient_accumulation_steps}."
            )
        self.prior_kl_coeff = cfg.training.get("prior_kl_coeff")
        self.pairwise_kl_coeff = cfg.training.get("pairwise_kl_coeff")
        self.train_inference_mode = cfg.training.inference_mode
        self.train_inference_kwargs = cfg.training.get("inference_kwargs") or {}

        def train_one_step_accumulate(state, batch, key):
            grad_acc = self.gradient_accumulation_steps
            batches = tree_map(lambda x: x.reshape(grad_acc, x.shape[0] // grad_acc, *x.shape[1:]), batch)
            keys = jax.random.split(key, grad_acc)
            old_num_steps = state.step
            state, metrics = jax.lax.scan(lambda s, b_k: self.train_one_step(s, *b_k), state, (batches, keys))
            # Update the step count manually to account for gradient accumulation
            state = state.replace(step=old_num_steps + 1)
            return state, metrics

        self.pmap_train_steps = jax.pmap(
            lambda state, batches, keys: jax.lax.scan(
                lambda s, b_k: train_one_step_accumulate(s, *b_k),
                state,
                (batches, keys),
            ),
            axis_name="devices",
            devices=self.devices,
        )

        def eval_one_step(batch: tuple[chex.Array, chex.Array, chex.PRNGKey], state: TrainState) -> dict:
            pairs, grid_shapes, key = batch
            random_search_key, perturbation_key, latents_key, latents_init_key = jax.random.split(key, 4)
            _, metrics = state.apply_fn(
                {"params": state.params},
                pairs,
                grid_shapes,
                dropout_eval=True,
                prior_kl_coeff=self.prior_kl_coeff,
                pairwise_kl_coeff=self.pairwise_kl_coeff,
                mode=self.train_inference_mode,
                rngs={
                    "random_search": random_search_key,
                    "gradient_ascent_random_perturbation": perturbation_key,
                    "latents": latents_key,
                    "latents_init": latents_init_key,
                },
                **self.train_inference_kwargs,
            )
            return metrics

        self.pmap_eval_steps = jax.pmap(
            lambda state, *args: jax.lax.map(partial(eval_one_step, state=state), args),
            axis_name="devices",
            devices=self.devices,
        )

        def build_generate_output_to_be_pmapped(eval_inference_mode: str, eval_inference_mode_kwargs: dict):
            def generate_output_to_be_pmapped(
                params,
                leave_one_out_grids,
                leave_one_out_shapes,
                grids_inputs,
                grids_outputs,
                shapes_inputs,
                shapes_outputs,
                keys,
            ) -> tuple[chex.Array, chex.Array, dict, dict]:
                generated_grids_outputs, generated_shapes_outputs, generated_info_outputs = jax.lax.map(
                    lambda args: self.model.apply(
                        {"params": params},
                        *args,
                        dropout_eval=True,
                        mode=eval_inference_mode,
                        **eval_inference_mode_kwargs,
                        method=self.model.generate_output,
                    ),
                    (leave_one_out_grids, leave_one_out_shapes, grids_inputs, shapes_inputs, keys),
                )

                correct_shapes = jnp.all(generated_shapes_outputs == shapes_outputs, axis=-1)
                batch_ndims = len(grids_inputs.shape[:-2])

                row_arange_broadcast = jnp.arange(grids_inputs.shape[-2]).reshape(
                    (*batch_ndims * (1,), grids_inputs.shape[-2])
                )
                input_row_mask = row_arange_broadcast < shapes_outputs[..., :1]
                col_arange_broadcast = jnp.arange(grids_inputs.shape[-1]).reshape(
                    (*batch_ndims * (1,), grids_inputs.shape[-1])
                )
                input_col_mask = col_arange_broadcast < shapes_outputs[..., 1:]
                input_mask = input_row_mask[..., None] & input_col_mask[..., None, :]

                pixels_equal = jnp.where(
                    input_mask & correct_shapes[..., None, None],
                    (generated_grids_outputs == grids_outputs),
                    False,
                )
                pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / shapes_outputs.prod(axis=(-1))
                accuracy = pixels_equal.sum(axis=(-1, -2)) == shapes_outputs.prod(axis=(-1))

                metrics = {
                    "correct_shapes": jnp.mean(correct_shapes),
                    "pixel_correctness": jnp.mean(pixel_correctness),
                    "accuracy": jnp.mean(accuracy),
                }
                return generated_grids_outputs, generated_shapes_outputs, generated_info_outputs, metrics

            return generate_output_to_be_pmapped

        if cfg.training.train_datasets and cfg.training.task_generator:
            raise ValueError("Only one of 'train_datasets' and 'task_generator' can be specified.")
        if cfg.training.train_datasets:
            # Load train datasets
            self.task_generator = False
            train_datasets = cfg.training.train_datasets
            if isinstance(train_datasets, str):
                train_datasets = [train_datasets]
            try:
                train_dataset_grids, train_dataset_shapes = [], []
                for grids, shapes, _ in load_datasets(train_datasets, cfg.training.get("use_hf", True)):
                    assert grids.shape[0:1] == shapes.shape[0:1]
                    train_dataset_grids.append(grids)
                    train_dataset_shapes.append(shapes)
                self.train_dataset_grids = jnp.concat(train_dataset_grids, axis=0)
                self.train_dataset_shapes = jnp.concat(train_dataset_shapes, axis=0)
                self.init_grids = self.train_dataset_grids[:1]
                self.init_shapes = self.train_dataset_shapes[:1]
            except Exception as e:
                logging.error(f"Error loading training datasets: {e}")
                raise
            logging.info(f"Train dataset shape: {self.train_dataset_grids.shape}")
        if cfg.training.task_generator:
            logging.info("Using a task generator for training.")
            self.task_generator = True
            self.task_generator_kwargs = cfg.training.task_generator
            for arg in ["num_workers", "num_pairs", "class"]:
                assert arg in self.task_generator_kwargs, f"Task generator must have arg '{arg}'."
            num_pairs = self.task_generator_kwargs["num_pairs"]
            num_rows, num_cols = self.model.encoder.config.max_rows, self.model.encoder.config.max_cols
            self.init_grids = jnp.zeros((1, num_pairs, num_rows, num_cols, 2), jnp.uint8)
            self.init_shapes = jnp.ones((1, num_pairs, 2, 2), jnp.uint8)
        self.online_data_augmentation = cfg.training.online_data_augmentation

        # Load eval datasets
        self.eval_datasets = []
        for dict_ in cfg.eval.eval_datasets or []:
            for arg in ["folder"]:
                assert arg in dict_, f"Each eval dataset must have arg '{arg}'."
            folder, length, seed = dict_["folder"], dict_.get("length"), dict_.get("seed", 0)
            grids, shapes, _ = load_datasets([folder], dict_.get("use_hf", True))[0]
            if length is not None:
                key = jax.random.PRNGKey(seed)
                indices = jax.random.permutation(key, len(grids))[:length]
                grids, shapes = grids[indices], shapes[indices]
            batch_size = dict_.get("batch_size", len(grids))
            # Drop the last batch if it's not full
            num_batches = len(grids) // batch_size
            grids, shapes = grids[: num_batches * batch_size], shapes[: num_batches * batch_size]
            self.eval_datasets.append(
                {
                    "dataset_name": folder.rstrip().split("/")[-1],
                    "dataset_grids": grids,
                    "dataset_shapes": shapes,
                    "batch_size": batch_size,
                }
            )

        # Load test datasets
        self.test_datasets = []
        for i, dict_ in enumerate(cfg.eval.test_datasets or []):
            if dict_.get("generator", False):
                for arg in ["num_pairs", "length"]:
                    assert arg in dict_, f"Each test generator dataset must have arg '{arg}'."
                num_pairs, length = dict_["num_pairs"], dict_["length"]
                default_dataset_name = dict_["generator"]  # CHANGE: Use the actual generator class name
                task_generator_kwargs = dict_.get("task_generator_kwargs") or {}
                grids, shapes, program_ids = make_dataset(
                    length,
                    num_pairs,
                    num_workers=8,
                    task_generator_class=dict_["generator"],
                    online_data_augmentation=self.online_data_augmentation,
                    seed=dict_.get("seed", 0),
                    **task_generator_kwargs,
                )
            else:
                for arg in ["folder", "length"]:
                    assert arg in dict_, f"Each test dataset must have arg '{arg}'."
                folder, length = dict_["folder"], dict_["length"]
                default_dataset_name = folder.rstrip().split("/")[-1]
                grids, shapes, program_ids = load_datasets([folder], dict_.get("use_hf", True))[0]
            if length is not None:
                key = jax.random.PRNGKey(dict_.get("seed", 0))
                indices = jax.random.permutation(key, len(grids))[:length]
                grids, shapes, program_ids = grids[indices], shapes[indices], program_ids[indices]
            batch_size = dict_.get("batch_size", len(grids))
            # Drop the last batch if it's not full
            num_batches = len(grids) // batch_size
            grids, shapes, program_ids = (
                grids[: num_batches * batch_size],
                shapes[: num_batches * batch_size],
                program_ids[: num_batches * batch_size],
            )
            inference_mode = dict_.get("inference_mode", "mean")
            # CHANGE: Fix the test name construction
            if dict_.get("name"):
                # If explicit name is provided, use it directly with inference mode
                test_name = dict_["name"] + "_" + inference_mode
            else:
                # If no name provided, use default_dataset_name + inference_mode
                test_name = default_dataset_name + "_" + inference_mode
            
            # Remove the duplicate generator prefix if it exists
            if test_name.startswith("generator_generator"):
                test_name = test_name.replace("generator_generator", "generator", 1)

            inference_kwargs = dict_.get("inference_kwargs", {})
            self.test_datasets.append(
                {
                    "pmap_dataset_generate_output": jax.pmap(
                        build_generate_output_to_be_pmapped(inference_mode, inference_kwargs),
                        axis_name="devices",
                        devices=self.devices,
                        donate_argnums=(3, 5),  # donate grid_inputs and shapes_inputs
                    ),
                    "test_name": test_name,
                    "dataset_grids": grids,
                    "dataset_shapes": shapes,
                    "batch_size": batch_size,
                    "num_tasks_to_show": dict_.get("num_tasks_to_show", 5),
                    "program_ids": program_ids,
                }
            )

        # Load json datasets
        self.json_datasets = []
        for i, dict_ in enumerate(cfg.eval.json_datasets or []):
            for arg in ["challenges", "solutions"]:
                assert arg in dict_, f"Each json dataset must have arg '{arg}'."
            json_challenges_file = dict_["challenges"]
            json_solutions_file = dict_["solutions"]
            inference_mode = dict_.get("inference_mode", "mean")
            default_dataset_name = json_challenges_file.rstrip().split("/")[-1].split(".")[0]
            test_name = default_dataset_name + "_" + dict_.get("name", f"{inference_mode}_{i}")
            evaluator = Evaluator(
                self.model,
                inference_mode=inference_mode,
                inference_mode_kwargs=dict_.get("inference_kwargs", {}),
                devices=self.devices,
            )
            self.json_datasets.append(
                {
                    "test_name": test_name,
                    "json_challenges_file": os.path.join(DATASETS_BASE_PATH, json_challenges_file),
                    "json_solutions_file": os.path.join(DATASETS_BASE_PATH, json_solutions_file),
                    "evaluator": evaluator,
                    "only_n_tasks": dict_.get("only_n_tasks", None),
                    "num_tasks_to_show": dict_.get("num_tasks_to_show", 5),
                    "overfit_task": dict_.get("overfit_task", None),
                }
            )

    def init_train_state(
        self, key: chex.PRNGKey, learning_rate: float, linear_warmup_steps: int = 99
    ) -> TrainState:
        variables = self.model.init(
            key,
            self.init_grids,
            self.init_shapes,
            dropout_eval=False,
            prior_kl_coeff=0.0,  # dummy value for initialization
            pairwise_kl_coeff=0.0,  # dummy value for initialization
            mode=self.train_inference_mode,
            **self.train_inference_kwargs,
        )
        linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
            init_value=learning_rate / (linear_warmup_steps + 1),
            peak_value=learning_rate,
            warmup_steps=linear_warmup_steps,
            transition_steps=1,
            end_value=learning_rate,
            decay_rate=1.0,
        )
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler))
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=self.gradient_accumulation_steps)
        return TrainState.create(apply_fn=self.model.apply, tx=optimizer, params=variables["params"])

    def train_one_step(
        self, state: TrainState, batch: tuple[chex.Array, chex.Array], key: chex.PRNGKey
    ) -> tuple[TrainState, dict]:
        pairs, grid_shapes = batch
        grads, metrics = jax.grad(state.apply_fn, has_aux=True)(
            {"params": state.params},
            pairs,
            grid_shapes,
            dropout_eval=False,
            prior_kl_coeff=self.prior_kl_coeff,
            pairwise_kl_coeff=self.pairwise_kl_coeff,
            mode=self.train_inference_mode,
            rngs=key,
            **self.train_inference_kwargs,
        )
        grads = grads["params"]
        grads = jax.lax.pmean(grads, axis_name="devices")
        state = state.apply_gradients(grads=grads)
        metrics.update(grad_norm=optax.global_norm(grads))
        return state, metrics

    def train_n_steps(
        self, state: TrainState, batches: tuple[chex.Array, chex.Array], key: chex.PRNGKey
    ) -> tuple[TrainState, dict]:
        num_devices, num_steps = batches[0].shape[0:2]
        keys = jax.random.split(key, (num_devices, num_steps))
        state, metrics = self.pmap_train_steps(state, batches, keys)
        # Mean the metrics over the devices and the n mini-batches
        metrics = tree_map(jnp.mean, metrics)
        return state, metrics

    @partial(jax.jit, static_argnames=("self", "log_every_n_steps"), backend="cpu")
    def prepare_train_dataset_for_epoch(
        self, key: chex.PRNGKey, log_every_n_steps: int
    ) -> tuple[chex.Array, chex.Array]:
        """Shuffle the dataset and reshape it to
        (num_logs, log_every_n_steps, num_devices, batch_size_per_device, *).
        """
        shuffle_key, augmentation_key = jax.random.split(key)
        grids, shapes = shuffle_dataset_into_batches(
            self.train_dataset_grids, self.train_dataset_shapes, self.batch_size, shuffle_key
        )  # (L, B, *)
        num_logs = grids.shape[0] // log_every_n_steps
        grids = grids[: num_logs * log_every_n_steps]
        shapes = shapes[: num_logs * log_every_n_steps]

        if grids.shape[1] % self.num_devices != 0:
            raise ValueError(
                f"Train batch size {grids.shape[1]} is not divisible by the number of devices {self.num_devices}."
            )
        batch_size_per_device = grids.shape[1] // self.num_devices

        if self.online_data_augmentation:
            grids, shapes = data_augmentation_fn(grids, shapes, augmentation_key)
        grids = grids.reshape(
            num_logs, self.num_devices, log_every_n_steps, batch_size_per_device, *grids.shape[2:]
        )
        shapes = shapes.reshape(
            num_logs, self.num_devices, log_every_n_steps, batch_size_per_device, *shapes.shape[2:]
        )
        return grids, shapes

    def eval(
        self,
        state: TrainState,
        dataset_name: str,
        dataset_grids: chex.Array,
        dataset_shapes: chex.Array,
        key: chex.PRNGKey,
        batch_size: int,
    ) -> dict[str, chex.Array]:
        """
        Evaluate the model on the given dataset. Computes the metrics similar to the training loss.

        Args:
            state: The current training state.
            dataset_name: The name of the dataset for logging purposes.
            dataset_grids: The dataset grids. Shape (L, N, R, C, 2).
            dataset_shapes: The shapes of the grids (e.g. 30x30). Shape (L, N, 2, 2).
                Expects dataset shapes values to be in [1, max_rows] and [1, max_cols].
            key: The random key to use for any inference stochasticity during the evaluation.
            batch_size: The batch size to use to iterate over the dataset.

        Returns:
            A dictionary containing the metrics.
        """
        # Split the dataset onto devices.
        assert dataset_grids.shape[0] % self.num_devices == 0
        dataset_grids, dataset_shapes = tree_map(
            lambda x: x.reshape((self.num_devices, x.shape[0] // self.num_devices, *x.shape[1:])),
            (dataset_grids, dataset_shapes),
        )
        # Split the dataset into batches for each device.
        batch_size_per_device = batch_size // self.num_devices
        assert dataset_grids.shape[1] % batch_size_per_device == 0
        dataset_grids, dataset_shapes = tree_map(
            lambda x: x.reshape(
                (x.shape[0], x.shape[1] // batch_size_per_device, batch_size_per_device, *x.shape[2:])
            ),
            (dataset_grids, dataset_shapes),
        )
        keys = jax.random.split(key, (self.num_devices, dataset_grids.shape[1]))  # (num_devices, num_batches)
        metrics = self.pmap_eval_steps(state, dataset_grids, dataset_shapes, keys)
        # Mean the metrics over the devices and the batches
        metrics = tree_map(jnp.mean, metrics)
        # Add the dataset name to the metrics
        metrics = {f"eval/{dataset_name}/{k}": v.item() for k, v in metrics.items()}
        return metrics

    def test_dataset_submission(
        self,
        state: TrainState,
        pmap_dataset_generate_output: Callable,
        test_name: str,
        dataset_grids: chex.Array,
        dataset_shapes: chex.Array,
        program_ids: Optional[chex.Array],
        batch_size: int,
        key: chex.PRNGKey,
        num_tasks_to_show: int = 5,
    ) -> tuple[dict[str, float], Optional[plt.Figure], plt.Figure, Optional[plt.Figure], Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Generate the output grids for each task in the dataset by leaving one (input, output) pair out.
        Does this by masking each of the N (input, output) pairs for each task.

        Args:
            state: The current training state.
            pmap_dataset_generate_output: The pmap function to generate the output grids.
            test_name: The name of the test dataset for logging purposes.
            dataset_grids: The dataset grids. Shape (L, N, R, C, 2).
            dataset_shapes: The shapes of the grids (e.g. 30x30). Shape (L, N, 2, 2).
                Expects dataset shapes values to be in [1, max_rows] and [1, max_cols].
            program_ids: The program ids for each task. Shape (L,).
            batch_size: The batch size to use to iterate over the dataset.
            key: The random key to use for any inference stochasticity during generation.
            num_tasks_to_show: The number of tasks to visualize (default to 5). 0 means no visualization.

        Returns:
            - A dictionary containing the metrics.
            - A figure containing the visualization of the generated grids.
            - A figure containing the visualization of the pixel accuracy heatmap.
            - A figure containing the visualization of the latents (T-SNE).
        """
        leave_one_out_grids = jax.jit(partial(make_leave_one_out, axis=-4), backend="cpu")(dataset_grids)
        leave_one_out_shapes = jax.jit(partial(make_leave_one_out, axis=-3), backend="cpu")(dataset_shapes)

        # Split the dataset onto devices.
        assert dataset_grids.shape[0] % self.num_devices == 0
        leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes = tree_map(
            lambda x: x.reshape((self.num_devices, x.shape[0] // self.num_devices, *x.shape[1:])),
            (leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes),
        )
        # Split the dataset into batches for each device.
        batch_size_per_device = batch_size // self.num_devices
        assert dataset_grids.shape[1] % batch_size_per_device == 0
        leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes = tree_map(
            lambda x: x.reshape(
                (x.shape[0], x.shape[1] // batch_size_per_device, batch_size_per_device, *x.shape[2:])
            ),
            (leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes),
        )
        grids_inputs, grids_outputs = dataset_grids[..., 0], dataset_grids[..., 1]
        shapes_inputs, shapes_outputs = dataset_shapes[..., 0], dataset_shapes[..., 1]
        keys = jax.random.split(key, (self.num_devices, dataset_grids.shape[1]))  # (num_devices, num_batches)
        generated_grids, generated_shapes, generated_info, metrics = pmap_dataset_generate_output(
            state.params,
            leave_one_out_grids,
            leave_one_out_shapes,
            grids_inputs,
            grids_outputs,
            shapes_inputs,
            shapes_outputs,
            keys,
        )

        # Extract search trajectory information
        search_trajectories = generated_info.get("search_trajectories", {})

        # Create search progress visualization
        if search_trajectories and ("gradient_ascent" in test_name or "random_search" in test_name):
            fig_search_progress = self.visualize_search_progress(search_trajectories, test_name)
        else:
            fig_search_progress = None

        program_context = generated_info["context"]
        pairs_per_problem = program_context.shape[-2]

        # Reshape to one large batch dim
        program_context = program_context.reshape(-1, program_context.shape[-1])

        # Aggregate the metrics over the devices
        metrics = tree_map(lambda x: x.mean(axis=0), metrics)
        # Add the dataset name to the metrics
        metrics = {f"test/{test_name}/{k}": v.item() for k, v in metrics.items()}

        # Concatenate the grids and shapes onto the batch dimension and the device dimension
        dataset_grids, dataset_shapes, generated_grids, generated_shapes = tree_map(
            lambda x: x.reshape((-1, *x.shape[3:])),
            (dataset_grids, dataset_shapes, generated_grids, generated_shapes),
        )

        # Create a mask based on the true shapes
        max_rows, max_cols = self.model.decoder.config.max_rows, self.model.decoder.config.max_cols
        grid_row_mask = jnp.arange(max_rows) < dataset_shapes[..., 0, 1:]
        grid_col_mask = jnp.arange(max_cols) < dataset_shapes[..., 1, 1:]
        grid_pad_mask = grid_row_mask[..., None] & grid_col_mask[..., None, :]

        # Extract the average accuracy for each pixel across batch and num_problems dimensions
        pixel_correct_binary = (generated_grids == dataset_grids[..., 1]) * grid_pad_mask
        pixel_accuracy = pixel_correct_binary.sum(axis=(0, 1)) / (grid_pad_mask.sum(axis=(0, 1)) + 1e-5)

        # Create heatmap of pixel accuracy and pixel frequency
        fig_heatmap = visualize_heatmap(
            pixel_accuracy, (grid_pad_mask.sum(axis=(0, 1)) / grid_pad_mask.sum())
        )

        if num_tasks_to_show:
            fig_grids = visualize_dataset_generation(
                dataset_grids, dataset_shapes, generated_grids, generated_shapes, num_tasks_to_show
            )
        else:
            fig_grids = None

        if program_ids is not None:
            repeated_program_ids = jnp.repeat(program_ids, pairs_per_problem)
            fig_latents = visualize_tsne(program_context, repeated_program_ids)
            
            # FIXED: Use original program_ids and actual latents_samples from generated_info
            if 'latents_samples' in generated_info:
                fig_latents_samples = visualize_latents_samples(
                    dataset_grids, 
                    dataset_shapes, 
                    program_ids,  # Use original program_ids (not repeated)
                    generated_info['latents_samples'],  # Use actual latents samples
                    num_tasks=min(num_tasks_to_show, 5),
                    num_samples_per_task=3
                )
            else:
                fig_latents_samples = None
        else:
            fig_latents = None
            fig_latents_samples = None

        # NEW: Extract and log optimization trajectories
        optimization_metrics = {}
        
        if 'optimization_trajectory' in generated_info:
            traj = generated_info['optimization_trajectory']
            
            # Log each optimization step
            for step, (accuracy, loss) in enumerate(zip(traj['step_accuracies'], traj['step_losses'])):
                optimization_metrics[f"optimization/{test_name}/step_{step:03d}/accuracy"] = accuracy
                optimization_metrics[f"optimization/{test_name}/step_{step:03d}/loss"] = loss
                
                if step < len(traj['step_improvements']):
                    optimization_metrics[f"optimization/{test_name}/step_{step:03d}/improvement"] = traj['step_improvements'][step]
            
            # Log summary metrics
            optimization_metrics[f"optimization/{test_name}/total_improvement"] = (
                traj['step_accuracies'][-1] - traj['step_accuracies'][0]
            )
            optimization_metrics[f"optimization/{test_name}/convergence_step"] = self._find_convergence_step(traj['step_accuracies'])
            optimization_metrics[f"optimization/{test_name}/final_accuracy"] = traj['step_accuracies'][-1]
        
        # NEW: Track random search progress
        if 'search_trajectory' in generated_info:
            search_traj = generated_info['search_trajectory']
            
            # Log each sample's performance
            for sample_idx, (accuracy, loss) in enumerate(zip(search_traj['sample_accuracies'], search_traj['sample_losses'])):
                optimization_metrics[f"search/{test_name}/sample_{sample_idx:03d}/accuracy"] = accuracy
                optimization_metrics[f"search/{test_name}/sample_{sample_idx:03d}/loss"] = loss
                optimization_metrics[f"search/{test_name}/sample_{sample_idx:03d}/improvement"] = search_traj['sample_improvements'][sample_idx]
            
            # Log best-so-far progression
            for sample_idx, best_acc in enumerate(search_traj['best_accuracy_progression']):
                optimization_metrics[f"search/{test_name}/best_so_far_{sample_idx:03d}"] = best_acc
            
            # Log summary statistics
            optimization_metrics[f"search/{test_name}/final_best_accuracy"] = max(search_traj['sample_accuracies'])
            optimization_metrics[f"search/{test_name}/mean_sample_accuracy"] = sum(search_traj['sample_accuracies']) / len(search_traj['sample_accuracies'])
            optimization_metrics[f"search/{test_name}/std_sample_accuracy"] = jnp.std(jnp.array(search_traj['sample_accuracies']))
            mean_accuracy = optimization_metrics[f"search/{test_name}/mean_sample_accuracy"]
            if hasattr(mean_accuracy, 'item'):
                mean_accuracy = mean_accuracy.item()
            optimization_metrics[f"search/{test_name}/samples_above_mean"] = sum(1 for acc in search_traj['sample_accuracies'] if acc > mean_accuracy)
            optimization_metrics[f"search/{test_name}/improvement_over_first"] = search_traj['sample_accuracies'][-1] - search_traj['sample_accuracies'][0]
            optimization_metrics[f"search/{test_name}/best_sample_index"] = search_traj['sample_accuracies'].index(max(search_traj['sample_accuracies']))
        
        # Merge with existing metrics
        metrics.update(optimization_metrics)
        
        # NEW: Store optimization data for comparison plots (FIXED structure)
        if not hasattr(self, '_optimization_data'):
            self._optimization_data = {}
        
        current_step = state.step[0].item() if hasattr(state.step, 'item') else state.step
        
        # Extract optimization trajectories for comparison
        if 'optimization_trajectory' in generated_info:
            traj = generated_info['optimization_trajectory']
            method_name = test_name.replace('_gradient_ascent', '').replace('_random_search', '')
            
            # Initialize storage structure (dict-of-steps)
            store = self._optimization_data.setdefault(method_name, {
                'gradient_ascent': {},
                'random_search': {}
            })
            
            if 'gradient_ascent' in test_name:
                store['gradient_ascent'][current_step] = {
                    'budgets': np.arange(len(traj['step_accuracies'])),
                    'accs': np.asarray(traj['step_accuracies'], dtype=float),
                }
        
        if 'search_trajectory' in generated_info:
            search_traj = generated_info['search_trajectory']
            method_name = test_name.replace('_gradient_ascent', '').replace('_random_search', '')
            
            # Initialize storage structure (dict-of-steps)
            store = self._optimization_data.setdefault(method_name, {
                'gradient_ascent': {},
                'random_search': {}
            })
            
            if 'random_search' in test_name:
                store['random_search'][current_step] = {
                    'budgets': np.arange(len(search_traj['best_accuracy_progression'])),
                    'accs': np.asarray(search_traj['best_accuracy_progression'], dtype=float),
                }
        
        return metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress

    def _find_convergence_step(self, accuracies):
        """Find the step where accuracy improvement becomes < 1%"""
        if len(accuracies) < 2:
            return len(accuracies)
        
        for i in range(1, len(accuracies)):
            improvement = accuracies[i] - accuracies[i-1]
            if improvement < 0.01:  # Less than 1% improvement
                return i
        return len(accuracies)

    def visualize_search_progress(self, trajectories, test_name):
        """Enhanced search progress visualization for both gradient ascent and random search"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        if "gradient_ascent" in test_name and 'optimization_trajectory' in trajectories:
            traj = trajectories['optimization_trajectory']
            
            # Plot 1: Accuracy over optimization steps
            axes[0, 0].plot(range(len(traj['step_accuracies'])), traj['step_accuracies'], 'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_title('Accuracy During Gradient Ascent')
            axes[0, 0].set_xlabel('Optimization Step')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Loss over optimization steps
            axes[0, 1].plot(range(len(traj['step_losses'])), traj['step_losses'], 'r-o', linewidth=2, markersize=4)
            axes[0, 1].set_title('Loss During Gradient Ascent')
            axes[0, 1].set_xlabel('Optimization Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Improvement over initial
            axes[0, 2].plot(range(len(traj['step_improvements'])), traj['step_improvements'], 'g-o', linewidth=2, markersize=4)
            axes[0, 2].set_title('Improvement Over Initial')
            axes[0, 2].set_xlabel('Optimization Step')
            axes[0, 2].set_ylabel('Accuracy Improvement')
            axes[0, 2].grid(True, alpha=0.3)
            
        elif "random_search" in test_name and 'search_trajectory' in trajectories:
            search_traj = trajectories['search_trajectory']
            
            # Plot 1: Best accuracy progression
            axes[0, 0].plot(range(len(search_traj['best_accuracy_progression'])), search_traj['best_accuracy_progression'], 'b-o', linewidth=2, markersize=3)
            axes[0, 0].set_title('Best Accuracy Found So Far')
            axes[0, 0].set_xlabel('Sample Number')
            axes[0, 0].set_ylabel('Best Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Sample accuracy distribution
            axes[0, 1].hist(search_traj['sample_accuracies'], bins=min(20, len(search_traj['sample_accuracies'])//2), alpha=0.7, color='orange')
            axes[0, 1].axvline(sum(search_traj['sample_accuracies'])/len(search_traj['sample_accuracies']), color='red', linestyle='--', label='Mean')
            axes[0, 1].axvline(max(search_traj['sample_accuracies']), color='green', linestyle='--', label='Best')
            axes[0, 1].set_title('Distribution of Sample Accuracies')
            axes[0, 1].set_xlabel('Accuracy')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].legend()
            
            # Plot 3: Individual sample accuracies
            axes[0, 2].scatter(range(len(search_traj['sample_accuracies'])), search_traj['sample_accuracies'], alpha=0.6, s=20)
            axes[0, 2].plot(range(len(search_traj['sample_accuracies'])), search_traj['sample_accuracies'], alpha=0.3)
            best_idx = search_traj['sample_accuracies'].index(max(search_traj['sample_accuracies']))
            axes[0, 2].scatter([best_idx], [search_traj['sample_accuracies'][best_idx]], color='red', s=100, marker='*', label='Best Sample')
            axes[0, 2].set_title('Individual Sample Accuracies')
            axes[0, 2].set_xlabel('Sample Number')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Bottom row: Comparative analysis (if both methods are available)
        # For now, just hide unused subplots
        for i in range(3):
            axes[1, i].set_visible(False)
        
        plt.suptitle(f'Search Progress Analysis: {test_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    @classmethod
    def test_json_submission(
        cls,
        state: TrainState,
        evaluator: Evaluator,
        json_challenges_file: str,
        json_solutions_file: str,
        test_name: str,
        key: chex.PRNGKey,
        only_n_tasks: Optional[int] = None,
        overfit_task: Optional[str] = None,
        num_tasks_to_show: int = 5,
        progress_bar: bool = False,
    ) -> tuple[dict[str, float], Optional[plt.Figure]]:
        with open(json_challenges_file, "r") as f:
            challenges = json.load(f)
        train = "training" in json_challenges_file
        generations = evaluator.json_submission(
            challenges, state.params, only_n_tasks, overfit_task, progress_bar, key, train=train
        )
        with open(json_solutions_file, "r") as f:
            solutions = json.load(f)
        metrics = evaluator.evaluate_generations(generations, solutions)
        metrics = {f"test/{test_name}/{k}": v for k, v in metrics.items()}

        if num_tasks_to_show:
            fig_grids = visualize_json_submission(challenges, generations, solutions, num_tasks_to_show)
        else:
            fig_grids = None

        return metrics, fig_grids

    def train_epoch(
        self,
        state: TrainState,
        key: chex.PRNGKey,
        trange: tqdm.tqdm,
        total_num_steps: int,
        log_every_n_steps: int,
        eval_every_n_logs: Optional[int] = None,
        save_checkpoint_every_n_logs: Optional[int] = None,
    ) -> TrainState:
        key, dataset_key = jax.random.split(key)
        if self.task_generator:
            task_generator_kwargs = dict(self.task_generator_kwargs)
            num_workers = task_generator_kwargs.pop("num_workers")
            task_generator_class = task_generator_kwargs.pop("class")
            num_pairs = task_generator_kwargs.pop("num_pairs")
            dataloader = make_task_gen_dataloader(
                batch_size=self.batch_size,
                log_every_n_steps=log_every_n_steps,
                num_workers=num_workers,
                task_generator_class=task_generator_class,
                num_pairs=num_pairs,
                num_devices=self.num_devices,
                online_data_augmentation=self.online_data_augmentation,
                **task_generator_kwargs,
            )
        else:
            # dataset shapes (num_logs, num_devices, log_every_n_steps, batch_size_per_device, *)
            grids, shapes = self.prepare_train_dataset_for_epoch(dataset_key, log_every_n_steps)
            dataloader = zip(grids, shapes)
        dataloading_time = time.time()
        for batches in dataloader:
            wandb.log({"timing/dataloading_time": time.time() - dataloading_time})
            # Training
            key, train_key = jax.random.split(key)
            start = time.time()
            state, metrics = self.train_n_steps(state, batches, train_key)
            end = time.time()
            trange.update(log_every_n_steps)
            self.num_steps += log_every_n_steps
            self.num_logs += 1
            throughput = log_every_n_steps * self.batch_size / (end - start)
            metrics.update(
                {"timing/train_time": end - start, "timing/train_num_samples_per_second": throughput}
            )
            wandb.log(metrics, step=self.num_steps)

            # Save checkpoint
            if save_checkpoint_every_n_logs and self.num_logs % save_checkpoint_every_n_logs == 0:
                # Save a checkpoint, after getting the state from the first device
                self.save_checkpoint("state.msgpack", tree_map(lambda x: x[0], state))

            # Evaluation
            if eval_every_n_logs and self.num_logs % eval_every_n_logs == 0:
                key, eval_key, test_key, json_key = jax.random.split(key, 4)

                # Eval datasets
                for dataset_dict in self.eval_datasets:
                    start = time.time()
                    eval_metrics = self.eval(state, key=eval_key, **dataset_dict)
                    eval_metrics[f"timing/eval_{dataset_dict['dataset_name']}"] = time.time() - start
                    wandb.log(eval_metrics, step=self.num_steps)

                # Dataset test with enhanced search progress tracking
                for dataset_dict in self.test_datasets:
                    start = time.time()
                    test_metrics, fig_grids, fig_heatmap, fig_latents, fig_latents_samples, fig_search_progress = self.test_dataset_submission(
                        state, key=test_key, **dataset_dict
                    )
                    test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                    
                    # Upload all figures including search progress
                    for fig, name in [
                        (fig_grids, "generation"),
                        (fig_heatmap, "pixel_accuracy"),
                        (fig_latents, "latents"),
                        (fig_latents_samples, "latents_samples"),
                        (fig_search_progress, "search_progress"),  # NEW
                    ]:
                        if fig is not None:
                            test_metrics[f"test/{dataset_dict['test_name']}/{name}"] = wandb.Image(fig)
                    
                    wandb.log(test_metrics, step=self.num_steps)
                    plt.close('all')  # Close all figures to prevent memory leaks

                # Json test
                for json_file_dict in self.json_datasets:
                    start = time.time()
                    test_metrics, fig_grids = self.test_json_submission(state, key=json_key, **json_file_dict)
                    json_test_name = json_file_dict["test_name"]
                    test_metrics[f"timing/test_{json_test_name}"] = time.time() - start
                    if fig_grids is not None:
                        test_metrics[f"test/{json_test_name}/generation"] = wandb.Image(fig_grids)
                    wandb.log(test_metrics, step=self.num_steps)

                # NEW: Generate optimization comparison plot
                comparison_fig = self._generate_optimization_comparison()
                if comparison_fig is not None:
                    comparison_metrics = {
                        "optimization_comparison/methods": wandb.Image(comparison_fig)
                    }
                    wandb.log(comparison_metrics, step=self.num_steps)
                    plt.close(comparison_fig)

            # Exit if the total number of steps is reached
            if self.num_steps >= total_num_steps:
                break

            dataloading_time = time.time()

        return state

    def _generate_optimization_comparison(self) -> Optional[plt.Figure]:
        """Generate comparison plot between optimization methods if data is available"""
        print(f"Starting optimization comparison with data: {self._optimization_data.keys()}")
        
        if not hasattr(self, '_optimization_data') or len(self._optimization_data) == 0:
            return None
        
        # Check if we have both gradient ascent and random search data
        method_names = list(self._optimization_data.keys())

        print(f"Available optimization methods: {method_names}")
        if len(method_names) == 0:
            return None
        
        # For now, take the first method that has both optimization types
        for method_name in method_names:
            data = self._optimization_data[method_name]
            
            grad_data = data.get('gradient_ascent', {})
            search_data = data.get('random_search', {})
            
            # Check if we have sufficient data for both methods
            print(f"Gradient Ascent Data points: {len(grad_data.keys())}, "
                         f"Random Search Data points: {len(search_data.keys())}")
            if (len(grad_data.keys()) > 0 and 
                len(search_data.keys()) > 0):
                
                # Use common steps
                common_steps = np.intersect1d(list(grad_data.keys()), list(search_data.keys()))
                if len(common_steps) < 2:
                    logging.warning(f"Skipping optimization comparison plot: common_steps={common_steps}")
                    continue
                
                # Get budget ranges
                max_grad_budget = max(v['budgets'].max() for k, v in grad_data.items() if k in common_steps)
                max_search_budget = max(v['budgets'].max() for k, v in search_data.items() if k in common_steps)
                
                print(f"Max Grad Budget: {max_grad_budget}, Max Search Budget: {max_search_budget}")
                
                # Use common budget range
                max_budget = min(max_grad_budget, max_search_budget)
                if max_budget < 2:
                    print(f"Skipping optimization comparison plot: max_budget={max_budget} < 2")
                    continue
                
                budgets = np.arange(max_budget + 1)
                print(f"Preparing accuracy grids for budgets: {budgets}, steps: {common_steps}")

                # Create accuracy grids with NaN for missing cells
                acc_grad = np.full((len(budgets), len(common_steps)), np.nan)
                acc_search = np.full((len(budgets), len(common_steps)), np.nan)
                
                # Fill gradient ascent data
                for j, step in enumerate(common_steps):
                    if step not in grad_data:
                        print(f"Warning: Step {step} not found in gradient ascent data")
                        continue
                    
                    step_data = grad_data[step]
                    if 'budgets' not in step_data or 'accs' not in step_data:
                        print(f"Warning: Step {step} missing required data keys: {list(step_data.keys())}")
                        continue
                    
                    b_ga, a_ga = step_data['budgets'], step_data['accs']
                    print(f"Gradient Ascent - step {step}: budgets={b_ga}, accs={a_ga}")
                    
                    # Validate data shapes
                    if len(b_ga) != len(a_ga):
                        print(f"Warning: Mismatch in gradient ascent data for step {step}: {len(b_ga)} budgets vs {len(a_ga)} accuracies")
                        continue
                    
                    # Validate budget indices are within bounds
                    if np.any(b_ga >= len(budgets)):
                        print(f"Warning: Budget indices {b_ga} exceed budget range {len(budgets)} for step {step}")
                        continue
                    
                    acc_grad[b_ga, j] = a_ga
                
                # Fill random search data  
                for j, step in enumerate(common_steps):
                    if step not in search_data:
                        print(f"Warning: Step {step} not found in random search data")
                        continue
                    
                    step_data = search_data[step]
                    if 'budgets' not in step_data or 'accs' not in step_data:
                        print(f"Warning: Step {step} missing required data keys: {list(step_data.keys())}")
                        continue
                    
                    b_rs, a_rs = step_data['budgets'], step_data['accs']
                    print(f"Random Search - step {step}: budgets={b_rs}, accs={a_rs}")
                    
                    # Validate data shapes
                    if len(b_rs) != len(a_rs):
                        print(f"Warning: Mismatch in random search data for step {step}: {len(b_rs)} budgets vs {len(a_rs)} accuracies")
                        continue
                    
                    # Validate budget indices are within bounds
                    if np.any(b_rs >= len(budgets)):
                        print(f"Warning: Budget indices {b_rs} exceed budget range {len(budgets)} for step {step}")
                        continue
                    
                    acc_search[b_rs, j] = a_rs
                
                # Debug: Show final accuracy grids
                print(f"Final accuracy grids:")
                print(f"Gradient Ascent shape: {acc_grad.shape}, NaN count: {np.isnan(acc_grad).sum()}")
                print(f"Random Search shape: {acc_search.shape}, NaN count: {np.isnan(acc_search).sum()}")
                
                # Validate that we have enough data to create a meaningful plot
                valid_grad_data = np.sum(~np.isnan(acc_grad))
                valid_search_data = np.sum(~np.isnan(acc_search))
                
                if valid_grad_data < 2 or valid_search_data < 2:
                    print(f"Warning: Insufficient valid data for plot. Gradient Ascent: {valid_grad_data}, Random Search: {valid_search_data}")
                    return None
                
                # Ensure we have at least 2 steps with data
                steps_with_grad_data = np.sum(np.any(~np.isnan(acc_grad), axis=0))
                steps_with_search_data = np.sum(np.any(~np.isnan(acc_search), axis=0))
                
                if steps_with_grad_data < 2 or steps_with_search_data < 2:
                    print(f"Warning: Insufficient steps with data. Gradient Ascent: {steps_with_grad_data}, Random Search: {steps_with_search_data}")
                    return None
                
                print(f"Generating optimization comparison plot with {valid_grad_data} gradient ascent points and {valid_search_data} random search points")
                print(f"Steps with data - Gradient Ascent: {steps_with_grad_data}, Random Search: {steps_with_search_data}")
                
                # Final validation of array shapes
                print(f"Final array shapes - acc_grad: {acc_grad.shape}, acc_search: {acc_search.shape}")
                print(f"Steps: {common_steps.shape}, Budgets: {budgets.shape}")
                
                # Ensure arrays have the expected shapes for visualization
                if acc_grad.shape != (len(budgets), len(common_steps)) or acc_search.shape != (len(budgets), len(common_steps)):
                    print(f"Warning: Array shape mismatch. Expected: ({len(budgets)}, {len(common_steps)}), Got: acc_grad: {acc_grad.shape}, acc_search: {acc_search.shape}")
                    return None
                
                # Generate comparison plot
                return visualize_optimization_comparison(
                    steps=common_steps,
                    budgets=budgets,
                    acc_A=acc_grad,
                    acc_B=acc_search,
                    method_A_name="Gradient Ascent",
                    method_B_name="Random Search"
                )
        return None

    def train(
        self,
        state: TrainState,
        cfg: omegaconf.DictConfig,
        key: chex.PRNGKey,
        progress_bar: bool = True,
        start_num_steps: int = 0,
    ) -> TrainState:
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        num_params_encoder = sum(x.size for x in jax.tree_util.tree_leaves(state.params["encoder"]))
        num_params_decoder = sum(x.size for x in jax.tree_util.tree_leaves(state.params["decoder"]))
        total_num_steps: int = cfg.training.total_num_steps
        log_every_n_steps: int = cfg.training.log_every_n_steps
        eval_every_n_logs: Optional[int] = cfg.training.eval_every_n_logs
        save_checkpoint_every_n_logs: Optional[int] = cfg.training.save_checkpoint_every_n_logs
        self.num_steps, self.num_logs = start_num_steps, 0
        logging.info("Starting training...")
        logging.info(f"Number of total parameters: {num_params:,}")
        logging.info(f"Number of encoder parameters: {num_params_encoder:,}")
        logging.info(f"Number of decoder parameters: {num_params_decoder:,}")
        logging.info(f"Running on devices: {self.devices}.")
        logging.info(f"Total number of gradient steps: {total_num_steps:,}.")
        if not self.task_generator:
            num_logs_per_epoch = self.train_dataset_grids.shape[0] // (log_every_n_steps * self.batch_size)
            if num_logs_per_epoch == 0:
                raise ValueError(
                    "The number of logs per epoch is 0 because the dataset size is "
                    f"{self.train_dataset_grids.shape[0]} < {self.batch_size=} * {log_every_n_steps=}."
                )
            num_steps_per_epoch = num_logs_per_epoch * log_every_n_steps
            num_epochs = math.ceil(total_num_steps / num_steps_per_epoch)
            logging.info(f"Number of epochs: {num_epochs:,}.")
            logging.info(f"Number of logs per epoch: {num_logs_per_epoch:,}.")
            logging.info(f"Number of gradient steps per epoch: {num_steps_per_epoch:,}.")
            logging.info(f"Total number of logs: {num_logs_per_epoch * num_epochs:,}.")
        else:
            num_epochs = 1
            logging.info(f"Total number of logs: {total_num_steps // log_every_n_steps:,}.")
        logging.info(f"Logging every {log_every_n_steps:,} gradient steps.")
        if eval_every_n_logs:
            steps_between_evals = eval_every_n_logs * log_every_n_steps
            logging.info(f"Total number of evaluations: {total_num_steps // steps_between_evals:,}.")
            logging.info(f"Evaluating every {steps_between_evals:,} gradient steps.")
        else:
            logging.info("Not evaluating during training.")
        if save_checkpoint_every_n_logs:
            steps_between_checkpoints = save_checkpoint_every_n_logs * log_every_n_steps
            logging.info(f"Total number of checkpoints: {total_num_steps // steps_between_checkpoints:,}.")
            logging.info(f"Saving a checkpoint every {steps_between_checkpoints:,} gradient steps.")
        else:
            logging.info("Not saving checkpoints during training.")
        trange = tqdm_trange(total_num_steps, disable=not progress_bar)
        try:
            for _ in range(num_epochs):
                key, epoch_key = jax.random.split(key)
                state = self.train_epoch(
                    state,
                    epoch_key,
                    trange,
                    total_num_steps,
                    log_every_n_steps,
                    eval_every_n_logs,
                    save_checkpoint_every_n_logs,
                )
        except KeyboardInterrupt:
            logging.info("Interrupted training.")
        return state

    def save_checkpoint(self, ckpt_path: str, state: TrainState) -> None:
        """Assume the state is not replicated on devices."""
        with open(ckpt_path, "wb") as outfile:
            outfile.write(msgpack_serialize(to_state_dict(state)))
        run_name = self.make_safe_run_name(wandb.run.name)
        artifact = wandb.Artifact(f"{run_name}--checkpoint", type="model", metadata=dict(wandb.run.config))
        artifact.add_file(ckpt_path)
        num_steps = state.step.item()
        wandb.run.log_artifact(artifact, name="checkpoint", aliases=["latest", f"num_steps_{num_steps}"])

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, state: TrainState) -> TrainState:
        artifact = wandb.use_artifact(checkpoint_path)
        artifact_dir = artifact.download()
        with open(os.path.join(artifact_dir, "state.msgpack"), "rb") as data_file:
            byte_data = data_file.read()
        state = from_bytes(state, byte_data)
        # Get the number of steps from the checkpoint alias
        start_num_steps = int([x for x in artifact.aliases if x.startswith("num_steps")][0].split("_")[-1])
        assert state.step == start_num_steps
        return state

    @classmethod
    def make_safe_run_name(cls, run_name: str) -> str:
        return (
            run_name.replace(",", ".")
            .replace(":", "")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("+", "_")
            .replace("=", "_")
        )


def instantiate_config_for_mpt(
    transformer_cfg: omegaconf.DictConfig,
) -> DecoderTransformerConfig | EncoderTransformerConfig:
    """Override the TransformerLayer config to account for mixed-precision training (bfloat16 data type)."""
    config = hydra.utils.instantiate(
        transformer_cfg,
        transformer_layer=hydra.utils.instantiate(transformer_cfg.transformer_layer, dtype=jnp.bfloat16),
    )
    return config


@hydra.main(config_path="configs", version_base=None, config_name="task_gen")
def run(cfg: omegaconf.DictConfig):
    logging.info("All devices available: {}".format(jax.devices()))
    if cfg.training.get("mixed_precision", False):
        encoder = EncoderTransformer(instantiate_config_for_mpt(cfg.encoder_transformer))
        decoder = DecoderTransformer(instantiate_config_for_mpt(cfg.decoder_transformer))
    else:
        encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
        decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))
    lpn = LPN(encoder=encoder, decoder=decoder)
    wandb.init(
        entity="ga624-imperial-college-london",
        project="LPN-ARC",
        settings=wandb.Settings(console="redirect"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    trainer = Trainer(cfg=cfg, model=lpn)
    init_key, train_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    init_state = trainer.init_train_state(init_key, cfg.training.learning_rate)
    if cfg.training.get("resume_from_checkpoint", False):
        checkpoint_path = cfg.training.resume_from_checkpoint
        logging.info(f"Resuming from checkpoint: {checkpoint_path}...")
        init_state = trainer.load_checkpoint(checkpoint_path, init_state)
    init_state = jax.device_put_replicated(init_state, trainer.devices)
    final_state = trainer.train(
        state=init_state,
        cfg=cfg,
        key=train_key,
        progress_bar=True,
        start_num_steps=init_state.step[0].item(),
    )
    # Save the final checkpoint, after getting the state from the first device
    trainer.save_checkpoint("state.msgpack", tree_map(lambda x: x[0], final_state))


if __name__ == "__main__":
    run()
