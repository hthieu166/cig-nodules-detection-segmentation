#!/usr/bin/env bash
source "scripts/master_env.sh"
exp_id="luna16_frcnn"
python main.py \
    --gpu_id $GPUID \
    --dataset_cfg "./configs/dataset_cfgs/luna16_slices.yaml" \
    --model_cfg   "./configs/model_cfgs/luna16_faster_r_cnn.yaml" \
    --train_cfg   "./configs/train_cfgs/${exp_id}.yaml" \
    --logdir      "logs/${exp_id}" \
    --log_fname   "logs/${exp_id}/stdout.log" \
    --is_training false \
    --train_mode  "from_scratch" \
    --pretrained_model "logs/${exp_id}/epoch_00009.model" \
    --output "outputs/${exp_id}/"
    -w $N_WORKERS
