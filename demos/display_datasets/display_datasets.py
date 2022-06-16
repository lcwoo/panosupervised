# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import sys

from display.display_sample import display_sample
from vidar.utils.config import read_config
from vidar.utils.data import set_random_seed
from vidar.utils.setup import setup_datasets

set_random_seed(42)

cfg = read_config('demos/display_datasets/config.yaml')
datasets = setup_datasets(cfg.datasets, verbose=True, stack=False)

filename = cfg.datasets.dict[sys.argv[1]].save_path[0].rstrip('/')
filename = '/'.join(filename.split('/')[:-1]) + '.mp4'

display_sample(datasets[0][sys.argv[1]][0][0], video_filename=filename, flip=False)
