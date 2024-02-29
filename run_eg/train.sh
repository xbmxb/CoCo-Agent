#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
EXPERIMENT_NAME='aitwfull_fullset_percept_goopre10_fixed'
deepspeed --include localhost:1,5,2,7 llava/train/train_mem.py \
    --run_name $EXPERIMENT_NAME \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path root_dir/checkpoints/llava-7b-$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path root_dir/scripts/aitw_data/fullset/fullset_8hist_cot_norm/llava_aitwfull_fullsetgoopre_8histlocation_cot_norm_truncted_fixed_train_QCM-LEA.json \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-pretrain-$MODEL_VERSION/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$EXPERIMENT_NAME-$MODEL_VERSION-finetune \
    --max_steps 40000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
