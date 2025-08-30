import os
import numpy as np
from src.datasets.task_gen.dataloader import make_dataset


def generate_struct_pattern(pattern_id: int, L: int = 96, N: int = 2, seed: int = 0):
    grids, shapes, program_ids = make_dataset(
        length=L,
        num_pairs=N,
        num_workers=0,
        task_generator_class="STRUCT_PATTERN",
        online_data_augmentation=False,
        seed=seed,
        pattern=pattern_id,
    )
    out_dir = f"src/datasets/struct_pattern_{pattern_id}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "grids.npy"), grids)
    np.save(os.path.join(out_dir, "shapes.npy"), shapes)
    np.save(os.path.join(out_dir, "program_ids.npy"), program_ids)


if __name__ == "__main__":
    for pid in (1, 2, 3):
        generate_struct_pattern(pid)


