#!/usr/bin/env bash

export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT="32500"
#pt=pretrain ppt=post-pretrain ft=finetune

DATA_ROOT_PATH='/mnt2/video_pretrain_dataset'
DATA_CSV_PATH='/mnt2/video_pretrain_dataset/annotation'
OUTPUT='fewshot_video_report/ActionRecognition'
MODEL_NAME='umt'

# model 2
FINETUNE='checkpoint/umt/umt-L-16____K710_65W_epoch200.pth'
model='vit_large_patch16_224'
EMBEDDING_SIZE=1024
PATCH_SIZE=16
NUM_FRAMES=8
INPUT_SIZE=224
TUBELET_SIZE=1
BATCH_SIZE=192

for SEED in 1
do
    for DATASET in K400 K600 K700 SSV2
    do
        for NUM_SHOTS in 10
        do
            echo "SEED: $SEED"
            echo "DATASET: $DATASET"
            echo "NUM_SHOTS: $NUM_SHOTS"

            FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
                --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
                ac_export_feature_and_attentive_probe.py \
                --embedding_size ${EMBEDDING_SIZE} \
                --data_set ${DATASET} \
                --seed ${SEED} \
                --num_shots ${NUM_SHOTS} \
                --num_step 8 \
                --data_root_path ${DATA_ROOT_PATH} \
                --data_csv_path ${DATA_CSV_PATH} \
                --save_report ${OUTPUT} \
                --batch_size ${BATCH_SIZE} \
                --model_name ${MODEL_NAME} \
                --model ${model} \
                --finetune ${FINETUNE} \
                --num_frames ${NUM_FRAMES} \
                --input_size ${INPUT_SIZE} \
                --tubelet_size ${TUBELET_SIZE} \
                --patch_size ${PATCH_SIZE}
        done
    done
done