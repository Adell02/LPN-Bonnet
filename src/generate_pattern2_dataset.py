import numpy as np
from datasets.task_gen.dataloader import make_dataset
L=96; N=2
grids, shapes, program_ids = make_dataset(
    length=L, num_pairs=N, num_workers=0,
    task_generator_class='PATTERN',
    pattern_size=2, num_rows=4, num_cols=4,
    online_data_augmentation=False, seed=0
)
np.save('src/datasets/pattern2d_eval/grids.npy', grids)
np.save('src/datasets/pattern2d_eval/shapes.npy', shapes)
np.save('src/datasets/pattern2d_eval/program_ids.npy', program_ids)