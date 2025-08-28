import numpy as np
from datasets.task_gen.dataloader import make_dataset


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
    np.save(f"src/datasets/struct_pattern_{pattern_id}/grids.npy", grids)
    np.save(f"src/datasets/struct_pattern_{pattern_id}/shapes.npy", shapes)
    np.save(f"src/datasets/struct_pattern_{pattern_id}/program_ids.npy", program_ids)


if __name__ == "__main__":
    for pid in (1, 2, 3):
        generate_struct_pattern(pid)


