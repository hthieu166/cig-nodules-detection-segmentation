# 20.04.16
#!/usr/bin/env bash
source "scripts/master_env.sh"
exp_id="luna16_frcnn_efficnet-b0"
# Train
# python main.py \
#     --gpu_id $GPUID \
#     --dataset_cfg "./configs/dataset_cfgs/luna16_slices.yaml" \
#     --model_cfg   "./configs/model_cfgs/${exp_id}.yaml" \
#     --train_cfg   "./configs/train_cfgs/${exp_id}.yaml" \
#     --logdir      "logs/${exp_id}" \
#     --log_fname   "logs/${exp_id}/stdout.log" \
#     --is_training true \
#     --train_mode  "resume" \
#     -w $N_WORKERS

# Eval
python main.py \
    --gpu_id $GPUID \
    --dataset_cfg "./configs/dataset_cfgs/luna16_slices.yaml" \
    --model_cfg   "./configs/model_cfgs/${exp_id}.yaml" \
    --train_cfg   "./configs/train_cfgs/${exp_id}.yaml" \
    --logdir      "logs/${exp_id}" \
    --log_fname   "logs/${exp_id}/stdout.log" \
    --is_training false \
    --train_mode  "from_scratch" \
    --output "./outputs/${exp_id}/" \
    --pretrained_model "logs/${exp_id}/epoch_00009.model" \
    -w $N_WORKERS

