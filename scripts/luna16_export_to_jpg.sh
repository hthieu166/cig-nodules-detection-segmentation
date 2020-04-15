#!/usr/bin/env bash
source "scripts/master_env.sh"
python ./src/tools/export_luna16_to_jpg.py \
    --dataset_cfg   "./configs/dataset_cfgs/luna16.yaml" \
    --out_dir       "../luna16_slices" \
