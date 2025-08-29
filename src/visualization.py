import io
import random

import chex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
import jax.numpy as jnp
import jax
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image

# Define color map
arc_cmap = ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
arc_norm = Normalize(vmin=0, vmax=9)


def display_grid(ax: plt.Axes, grid: chex.Array, grid_shape: chex.Array) -> None:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    rows = min(rows, grid.shape[0])
    cols = min(cols, grid.shape[1])

    ax.imshow(
        grid[:rows, :cols],
        cmap=arc_cmap,
        interpolation="nearest",
        aspect="equal",
        norm=arc_norm,
        origin="lower",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Add shape information below the grid
    ax.text(0.5, -0.02, f"{rows}x{cols}", transform=ax.transAxes, ha="center", va="top")


def ax_to_pil(ax):
    fig = ax.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory
    buf.close()
    return pil_image


def display_function_examples(grids, shapes, num_pairs=None, seed=None) -> tuple:
    if seed is not None:
        random.seed(seed)
    if num_pairs is None:
        num_pairs = grids.shape[1]
    num_pairs = min(max(num_pairs, 1), grids.shape[1])  # Ensure num_pairs between 1 and number of pairs

    b = random.randint(0, grids.shape[0] - 1)  # Choose a random batch
    _, axs = plt.subplots(2, num_pairs, figsize=(5 * num_pairs, 2 * num_pairs))

    for i in range(num_pairs):
        input_grid = grids[b, i, :, :, 0]
        output_grid = grids[b, i, :, :, 1]
        input_grid_shape = shapes[b, i, :, 0]
        output_grid_shape = shapes[b, i, :, 1]

        # Display input grid
        display_grid(axs[0, i], input_grid, input_grid_shape)

        # Display output grid
        display_grid(axs[1, i], output_grid, output_grid_shape)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    return grids[b], shapes[b]  # Return the input grids and shapes for the batch


def visualize_json_submission(
    challenges: dict[str, list], generations: dict[str, list], solutions: dict[str, list], num_tasks: int = 5
) -> plt.Figure:
    keys = list(generations.keys())[:num_tasks]
    num_tasks = len(keys)
    # Find the maximum number of (train + test) pairs throughout all tasks
    max_num_pairs = max(
        len(challenge["train"]) + len(challenge["test"])
        for task_id, challenge in challenges.items()
        if task_id in generations
    )
    height_ratios = num_tasks * [1, 1, 1, 0.2]  # Input, True Output, Predicted Output, Line
    fig, axs = plt.subplots(
        4 * num_tasks - 1,
        max_num_pairs,
        figsize=(3 * max_num_pairs, 9.6 * num_tasks - 0.6),
        height_ratios=height_ratios[:-1],
        dpi=min(100, int(100 / (num_tasks / 60))),  # Limit the total number of pixels
    )

    for task_index, key in enumerate(keys):
        challenge_list, solution_list, generation_list = challenges[key], solutions[key], generations[key]
        num_test_grids = len(challenge_list["test"])
        for test_index, (challenge, solution, generation) in enumerate(
            zip(challenge_list["test"], solution_list, generation_list)
        ):
            input_grid = np.array(challenge["input"])
            output_grid = np.array(solution)
            prediction_grid = np.array(generation["attempt_1"])

            display_grid(axs[4 * task_index, test_index], input_grid, (30, 30))
            axs[4 * task_index, test_index].set_title("Input")

            display_grid(axs[4 * task_index + 1, test_index], output_grid, (30, 30))
            axs[4 * task_index + 1, test_index].set_title("Output")

            display_grid(axs[4 * task_index + 2, test_index], prediction_grid, (30, 30))
            axs[4 * task_index + 2, test_index].set_title("Prediction")

        for train_task_index, train_task in enumerate(challenge_list["train"]):
            input_grid = np.array(train_task["input"])
            output_grid = np.array(train_task["output"])

            display_grid(axs[4 * task_index, num_test_grids + train_task_index], input_grid, (30, 30))
            axs[4 * task_index, num_test_grids + train_task_index].set_title("Input")

            display_grid(axs[4 * task_index + 1, num_test_grids + train_task_index], output_grid, (30, 30))
            axs[4 * task_index + 1, num_test_grids + train_task_index].set_title("Output")

            axs[4 * task_index + 2, num_test_grids + train_task_index].axis("off")

        for col in range(num_test_grids + len(challenge_list["train"]), max_num_pairs):
            for row in range(4 * task_index, 4 * task_index + 3):
                axs[row, col].axis("off")

        # Draw a line to separate tasks
        if task_index < len(keys) - 1:
            for i in range(max_num_pairs):
                axs[4 * task_index + 3, i].axis("off")
                axs[4 * task_index + 3, i].axhline(0.5, color="black", linewidth=4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.25)  # Increased hspace for shape labels

    return fig


def visualize_dataset_generation(
    dataset: chex.Array,
    grid_shapes: chex.Array,
    predicted_grids: chex.Array,
    predicted_shapes: chex.Array,
    num_tasks: int,
) -> plt.Figure:
    num_pairs = dataset.shape[1]
    height_ratios = num_tasks * [1, 1, 1, 0.2]  # Input, True Output, Predicted Output, Line
    fig, axs = plt.subplots(
        4 * num_tasks - 1,
        num_pairs,
        figsize=(3 * num_pairs, 9.6 * num_tasks - 0.6),
        height_ratios=height_ratios[:-1],
    )

    for task_index in range(num_tasks):

        for i in range(num_pairs):
            input_grid = dataset[task_index, i, :, :, 0]
            output_grid = dataset[task_index, i, :, :, 1]
            prediction_grid = predicted_grids[task_index, i]

            input_shape = grid_shapes[task_index, i, :, 0]
            output_shape = grid_shapes[task_index, i, :, 1]
            prediction_shape = predicted_shapes[task_index, i]

            display_grid(axs[4 * task_index, i], input_grid, input_shape)
            axs[4 * task_index, i].set_title("Input")

            display_grid(axs[4 * task_index + 1, i], output_grid, output_shape)
            axs[4 * task_index + 1, i].set_title("Output")

            display_grid(axs[4 * task_index + 2, i], prediction_grid, prediction_shape)
            axs[4 * task_index + 2, i].set_title("Prediction")

        # Draw a line to separate tasks
        if task_index < num_tasks - 1:
            for i in range(num_pairs):
                axs[4 * task_index + 3, i].axis("off")
                axs[4 * task_index + 3, i].axhline(0.5, color="black", linewidth=4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.25)  # Increased hspace for shape labels

    return fig


def visualize_heatmap(data, proportion):
    # Ensure the inputs are JAX arrays
    data = jnp.asarray(data)
    proportion = jnp.asarray(proportion)

    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the original data
    im1 = ax1.imshow(data, cmap="coolwarm")
    # make the color bar in the range [0, 1]
    im1.set_clim(0, 1)
    ax1.set_title("Original Data")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(which="major", color="w", linestyle="-", linewidth=0.5)
    fig.colorbar(im1, ax=ax1, label="Value")

    # Plot the proportion data
    im2 = ax2.imshow(proportion, cmap="cool")
    ax2.set_title("Proportion in Dataset")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(which="major", color="w", linestyle="-", linewidth=0.5)
    fig.colorbar(im2, ax=ax2, label="Proportion")

    # Adjust layout and return the figure
    plt.tight_layout()
    return fig


def visualize_tsne(latents, program_ids, perplexity=2, max_iter=1000, random_state=42):
    """
    Create a t-SNE visualization of latent embeddings, colored by program IDs with distinct colors,
    a legend, and numbered points.

    Args:
    latents (jnp.array): Array of latent embeddings with shape (B, latent_embedding_size)
    program_ids (jnp.array or list): Integer values for each latent, used for coloring
    perplexity (int or float): Perplexity parameter for t-SNE (default: 30)
    max_iter (int): Number of iterations for t-SNE (default: 1000)
    random_state (int): Random state for reproducibility (default: 42)

    Returns:
    fig (matplotlib.figure.Figure): Figure object containing the t-SNE plot
    """
    # Convert JAX array to NumPy array
    latents_np = np.array(latents).astype(float)

    # Ensure perplexity is a scalar float
    perplexity = float(perplexity)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    try:
        if np.all(latents_np == latents_np[0]):
            # If all latents are the same, t-SNE will fail
            return None
        embeddings_2d = tsne.fit_transform(latents_np)
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        print(f"Shape of latents: {latents_np.shape}")
        print(f"Data type of latents: {latents_np.dtype}")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Get unique program IDs and assign a color to each
    unique_ids = np.unique(program_ids)
    num_colors = len(unique_ids)

    # Use custom color palette
    custom_colors = ['#FBB998', '#DB74DB', '#5361E5', '#4B9D61']
    # Cycle through custom colors if we have more program IDs than colors
    color_palette = [custom_colors[i % len(custom_colors)] for i in range(num_colors)]
    color_map = dict(zip(unique_ids, color_palette))

    # Plot each program ID with its assigned color and add numbering
    for id in unique_ids:
        mask = np.array(program_ids) == id
        points = embeddings_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], c=[color_map[id]], label=f"Program {id}", alpha=0.7, s=50)

        # Add numbering to each point
        for point in points:
            ax.annotate(str(id), point, xytext=(3, 3), textcoords="offset points", fontsize=8, alpha=0.8)

    ax.set_title("t-SNE Visualization of Latent Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()

    return fig


def visualize_tsne_sources(
    latents: jnp.ndarray,
    program_ids: jnp.ndarray,
    source_ids: jnp.ndarray,
    max_points: int = 2000,
    random_state: int = 42,
):
    """t-SNE with task color (arc_cmap) and source encoded as marker.

    Args:
        latents: [N, D]
        program_ids: [N]
        source_ids: [N] integers (0..E-1 encoders, E for PoE)
        max_points: cap points for memory
    Returns:
        fig
    """
    lat_np = np.asarray(latents, dtype=float)
    prog_np = np.asarray(program_ids)
    src_np = np.asarray(source_ids)
    N = lat_np.shape[0]
    if N == 0:
        return None
    if N > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, size=max_points, replace=False)
        lat_np = lat_np[idx]
        prog_np = prog_np[idx]
        src_np = src_np[idx]

    try:
        if np.all(lat_np == lat_np[0]):
            return None
        tsne = TSNE(n_components=2, perplexity=2, max_iter=1000, random_state=random_state)
        emb = tsne.fit_transform(lat_np)
    except Exception as e:
        print(f"Error during t-SNE (sources): {e}")
        return None

    # Create the plot - EXACTLY like train.py style
    fig, ax = plt.subplots(figsize=(15, 12))

    # Get unique program IDs (pattern types)
    unique_ids = np.unique(prog_np)
    
    # Get unique source IDs (encoders + context)
    unique_sources = sorted(list(np.unique(src_np)))

    # Define colors for patterns and shapes for sources
    pattern_colors = {
        1: '#FBB998',  # O-tetromino - Blue
        2: '#DB74DB',  # T-tetromino - Green  
        3: '#5361E5'   # L-tetromino - Red
    }
    
    source_markers = {
        0: 'o',    # Encoder 0 - Circle
        1: 's',    # Encoder 1 - Square
        2: '^',    # Encoder 2 - Triangle
        3: 'D'     # Context - Diamond
    }
    
    source_labels = {
        0: "Encoder 0",
        1: "Encoder 1", 
        2: "Encoder 2",
        3: "Context"
    }
    
    pattern_labels = {
        1: "O-tetromino",
        2: "T-tetromino", 
        3: "L-tetromino"
    }
    
    # Plot each source with different markers
    for src in unique_sources:
        m = src_np == src
        mk = source_markers.get(src, 'o')
        
        # Plot each program ID separately
        for prog_id in unique_ids:
            prog_mask = m & (prog_np == prog_id)
            if np.any(prog_mask):
                points = emb[prog_mask]
                
                # Use pattern color and source marker
                color = pattern_colors.get(prog_id, '#AAAAAA')
                ax.scatter(
                    points[:, 0], points[:, 1], 
                    c=[color], 
                    marker=mk, 
                    alpha=0.7, 
                    s=50,
                    edgecolors='none'
                )
                
                # Add labels on samples showing GROUP ID (sample index), not pattern type
                # We need to find the original sample indices for these points
                sample_indices = np.where(prog_mask)[0]
                for i, (point, sample_idx) in enumerate(zip(points, sample_indices)):
                    # Calculate the group ID based on the sample index
                    # Since we have 4 points per group (3 encoders + 1 context), 
                    # and we're plotting all sources together, we need to map back to original samples
                    group_id = sample_idx // 4  # Each group has 4 points
                    ax.annotate(str(group_id), point, xytext=(3, 3), textcoords="offset points", fontsize=8, alpha=0.8)

    # EXACTLY same title, labels, and style as train.py
    ax.set_title("t-SNE Visualization of Latent Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Build a clean legend with:
    # - Colors: Pattern 1/2/3
    # - Shapes: Encoder 0/1/2 and Context
    from matplotlib.lines import Line2D
    pattern_handles = []
    for pid in [1, 2, 3]:
        if pid in unique_ids:
            pattern_handles.append(
                Line2D([0], [0], marker='o', linestyle='None', color='none',
                       markerfacecolor=pattern_colors.get(pid, '#AAAAAA'), markeredgecolor='none', markersize=10,
                       label=f"Pattern {pid}")
            )
    # Shapes legend: show only marker glyphs (no text labels)
    shape_handles = []
    for src in unique_sources:
        marker = source_markers.get(src, 'o')
        shape_handles.append(
            Line2D([0], [0], marker=marker, linestyle='None', color='black',
                   markerfacecolor='white', markeredgecolor='black', markersize=10, label='')
        )
    # Combine: first colors with labels, then shapes without labels
    handles = pattern_handles + shape_handles
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0,
              title="Patterns (color) and Sources (shape)")

    # Adjust layout to prevent legend from being cut off - EXACTLY like train.py
    plt.tight_layout()

    return fig

def visualize_latents_samples(
    dataset_grids: chex.Array,
    dataset_shapes: chex.Array, 
    program_ids: chex.Array,
    latents_samples: chex.Array,
    num_tasks: int = 5,
    num_samples_per_task: int = 3
) -> plt.Figure:
    """
    Visualize latent samples showing input-output pairs like generation visualization.
    
    Args:
        dataset_grids: [num_tasks, num_pairs, max_height, max_width]
        dataset_shapes: [num_tasks, num_pairs, 2] 
        program_ids: [num_tasks]
        latents_samples: [num_tasks, num_samples, latent_dim]
        num_tasks: Number of tasks to show
        num_samples_per_task: Number of latent samples to show per task
    """
    import matplotlib.pyplot as plt
    
    num_tasks = min(num_tasks, dataset_grids.shape[0])
    num_samples_per_task = min(num_samples_per_task, latents_samples.shape[1])
    
    # Calculate figure dimensions
    # Each task shows: input pairs + output pairs for each sample
    num_pairs = dataset_grids.shape[1] // 2  # Assuming input/output pairs
    cols_per_task = num_pairs * 2  # input + output columns
    total_cols = cols_per_task * num_samples_per_task
    
    fig, axes = plt.subplots(
        num_tasks, total_cols, 
        figsize=(total_cols * 1.5, num_tasks * 1.5),
        squeeze=False
    )
    
    for task_idx in range(num_tasks):
        pattern_id = int(program_ids[task_idx]) if program_ids is not None else task_idx
        
        for sample_idx in range(num_samples_per_task):
            # Get the input-output pairs for this task
            task_grids = dataset_grids[task_idx]  # [num_pairs, H, W]
            task_shapes = dataset_shapes[task_idx]  # [num_pairs, 2]
            
            col_offset = sample_idx * cols_per_task
            
            # Show input-output pairs
            for pair_idx in range(num_pairs):
                input_idx = pair_idx * 2
                output_idx = pair_idx * 2 + 1
                
                # Input grid
                input_col = col_offset + pair_idx * 2
                display_grid(
                    axes[task_idx, input_col],
                    task_grids[input_idx],
                    task_shapes[input_idx]
                )
                
                # Output grid  
                output_col = col_offset + pair_idx * 2 + 1
                display_grid(
                    axes[task_idx, output_col], 
                    task_grids[output_idx],
                    task_shapes[output_idx]
                )
                
                # Add titles for first sample only
                if sample_idx == 0:
                    axes[task_idx, input_col].set_title(f"In {pair_idx+1}", fontsize=8)
                    axes[task_idx, output_col].set_title(f"Out {pair_idx+1}", fontsize=8)
        
        # Add row label with pattern number
        axes[task_idx, 0].set_ylabel(f"Pattern {pattern_id}", fontsize=10, fontweight='bold')
    
    # Add column headers for each sample
    for sample_idx in range(num_samples_per_task):
        col_offset = sample_idx * cols_per_task
        fig.text(
            (col_offset + cols_per_task/2) / total_cols, 0.95,
            f"Latent Sample {sample_idx + 1}",
            ha='center', fontsize=12, fontweight='bold'
        )
    
    plt.suptitle("Latent Samples - Input/Output Pairs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def visualize_optimization_comparison(
    steps: chex.Array,
    budgets: chex.Array, 
    acc_A: chex.Array,
    acc_B: chex.Array,
    method_A_name: str = "Method A",
    method_B_name: str = "Method B"
) -> plt.Figure:
    """
    Visualize comparison between two optimization methods across training steps and search budgets.
    
    Args:
        steps: 1D array of training steps [S]
        budgets: 1D array of search budgets [B] 
        acc_A: 2D array of accuracies for method A [B, S]
        acc_B: 2D array of accuracies for method B [B, S]
        method_A_name: Name of first method
        method_B_name: Name of second method
        
    Returns:
        Figure showing heatmap of accuracy differences with crossing contour
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # numpy
    steps   = np.asarray(steps)
    budgets = np.asarray(budgets)
    acc_A   = np.asarray(acc_A, dtype=float)
    acc_B   = np.asarray(acc_B, dtype=float)

    # diff heatmap data [B,S]
    diff = acc_A - acc_B
    diff_masked = np.ma.masked_invalid(diff)
    if diff_masked.count() > 0:
        vmax = float(np.nanmax(np.abs(diff_masked))) or 1.0
    else:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(12, 8))

    # heatmap
    # Handle single-point axes for sane extents
    x0, x1 = (steps[0] - 0.5, steps[0] + 0.5) if steps.size == 1 else (steps[0], steps[-1])
    y0, y1 = (budgets[0] - 0.5, budgets[0] + 0.5) if budgets.size == 1 else (budgets[0], budgets[-1])
    im = ax.imshow(
        diff_masked,
        extent=[x0, x1, y0, y1],
        origin='lower', aspect='auto',
        cmap='cool', vmin=-vmax, vmax=+vmax
    )

    # zero contour A==B, and make it show in legend
    X, Y = np.meshgrid(steps, budgets)
    try:
        cs = ax.contour(X, Y, diff, levels=[0.0], colors='black', linewidths=2.0, alpha=0.9)
        # label the first collection so legend picks it up
        if cs.collections:
            cs.collections[0].set_label('Equal accuracy (A = B)')
    except (ValueError, RuntimeError, TypeError):
        cs = None  # ignore if not possible

    # axes labels/title
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Search budget", fontsize=12)
    ax.set_title(f"Optimization strategies comparison\n({method_A_name} vs {method_B_name})", fontsize=14)

    # all ticks
    ax.set_xticks(steps)
    ax.set_yticks(budgets)
    if steps.size > 12:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')

    # layout helper
    divider = make_axes_locatable(ax)

    # colorbar axis
    cax = divider.append_axes("right", size="4%", pad=0.6)
    cbar = fig.colorbar(im, cax=cax)
    # colorbar title ABOVE, horizontal
    cbar.ax.set_title(f"Accuracy diff\n({method_A_name} − {method_B_name})",
                      fontsize=11, pad=10, rotation=0, loc='center')
    cbar.ax.tick_params(length=3, pad=3)

    # separate slim axis to the RIGHT of the colorbar for the explanatory texts (increase padding to avoid overlap)
    label_ax = divider.append_axes("right", size="12%", pad=0.8)
    label_ax.axis("off")
    # two-line labels, centered vertically near top and bottom of this axis
    label_ax.text(0.05, 0.95, f"{method_A_name}\nmore accurate",
                  ha="left", va="top", fontsize=9)
    label_ax.text(0.05, 0.05, f"{method_B_name}\nmore accurate",
                  ha="left", va="bottom", fontsize=9)

    # legend: include contour and optionally a proxy for heatmap
    handles = []
    labels  = []
    if cs and cs.collections:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='black', lw=2))
        labels.append('Equal accuracy')
    # If you want a legend entry for "A−B heatmap", add a proxy
    # handles.append(Line2D([0],[0], color='none')) ; labels.append('A−B heatmap')
    if handles:
        ax.legend(handles, labels, loc='upper left', frameon=True)

    fig.tight_layout()
    return fig


def visualize_struct_confidence_panel(
    sample_grids: chex.Array,
    sample_shapes: chex.Array,
    encoder_mus: list[chex.Array],
    encoder_logvars: list[chex.Array],
    poe_mu: chex.Array,
    poe_logvar: chex.Array,
    title: str = "Structured Confidence Panel",
    encoder_labels: list[str] | None = None,
) -> plt.Figure:
    """
    Panel with:
    - Top: the struct (input-output pairs) for one task
    - Bottom-left: histogram of latent means per encoder and PoE
    - Bottom-right: histogram of latent variances per encoder and PoE

    Args:
        sample_grids: [N, R, C, 2] for a single task (pairs)
        sample_shapes: [N, 2, 2] shapes for that task
        encoder_mus: list of [N, D] or [N, D] means per encoder (aggregated over pair dim if needed)
        encoder_logvars: list of [N, D] logvars per encoder (same shape as mus)
        poe_mu: [N, D] PoE mean
        poe_logvar: [N, D] PoE logvar
        encoder_labels: optional labels for legend order
    """
    import numpy as _np
    num_pairs = int(sample_grids.shape[0])

    fig = plt.figure(figsize=(16, 10))
    grid = plt.GridSpec(3, 3, hspace=0.6, wspace=0.3)

    # Top: struct visualization spanning all columns
    # Create inner layout manually without GridSpecFromSubplotSpec for compatibility
    top_axes = []
    for i in range(num_pairs):
        # Position subplots manually: two rows in the top section
        # Row heights: first two rows share ~2/3 of height
        ax_in = fig.add_subplot(3, num_pairs, i + 1)
        display_grid(ax_in, _np.array(sample_grids[i, :, :, 0]), _np.array(sample_shapes[i, :, 0]))
        if i == 0:
            ax_in.set_title("Input")
        ax_out = fig.add_subplot(3, num_pairs, num_pairs + i + 1)
        display_grid(ax_out, _np.array(sample_grids[i, :, :, 1]), _np.array(sample_shapes[i, :, 1]))
        if i == 0:
            ax_out.set_title("Output")

    # Bottom-left: histogram of means
    ax_means = fig.add_subplot(3, 3, 7)
    # Bottom-right: histogram of variances
    ax_vars = fig.add_subplot(3, 3, 8)

    # Colors for encoders + PoE (consistent and distinct)
    enc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    poe_color = '#d62728'

    if encoder_labels is None:
        encoder_labels = [f"Encoder {i}" for i in range(len(encoder_mus))]

    # Plot encoder distributions
    for idx, (mu, logvar) in enumerate(zip(encoder_mus, encoder_logvars)):
        mu_flat = _np.asarray(mu).reshape(-1)
        var_flat = _np.exp(_np.asarray(logvar).reshape(-1))
        color = enc_colors[idx % len(enc_colors)]
        ax_means.hist(mu_flat, bins=30, alpha=0.5, color=color, label=encoder_labels[idx], density=True)
        ax_vars.hist(var_flat, bins=30, alpha=0.5, color=color, label=encoder_labels[idx], density=True)

    # Plot PoE distributions
    poe_mu_flat = _np.asarray(poe_mu).reshape(-1)
    poe_var_flat = _np.exp(_np.asarray(poe_logvar).reshape(-1))
    ax_means.hist(poe_mu_flat, bins=30, histtype='step', linewidth=2.0, color=poe_color, label='PoE', density=True)
    ax_vars.hist(poe_var_flat, bins=30, histtype='step', linewidth=2.0, color=poe_color, label='PoE', density=True)

    ax_means.set_title("Latent Means")
    ax_vars.set_title("Latent Variances")
    ax_means.set_xlabel("mean")
    ax_vars.set_xlabel("variance")
    ax_means.set_ylabel("density")
    ax_vars.set_ylabel("density")
    ax_means.legend(frameon=True)
    ax_vars.legend(frameon=True)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

