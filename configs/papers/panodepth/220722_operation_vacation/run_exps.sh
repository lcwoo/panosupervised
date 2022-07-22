#!/usr/bin/bash

python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_1.yaml
python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_2.yaml
python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_3.yaml
python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_4.yaml
python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_5.yaml
python3 scripts/run_ddp.py configs/papers/panodepth/220722_operation_vacation/train_ddad_6.yaml