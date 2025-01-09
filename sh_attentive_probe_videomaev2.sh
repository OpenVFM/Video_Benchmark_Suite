#!/usr/bin/env bash

export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT="32500"
#pt=pretrain ppt=post-pretrain ft=finetune

DATA_ROOT_PATH='fewshot_video/ActionRecognition'
DATA_CSV_PATH='fewshot_video/ActionRecognition'
OUTPUT='fewshot_video_report3/ActionRecognition'
MODEL_NAME='videomae_v2'

# model 2
FINETUNE='checkpoint/videomae_v2/videomae-g-14____UnlabeledHybrid_1.34M_epoch1200.pth'
model='vit_giant_patch14_224'
EMBEDDING_SIZE=1408
PATCH_SIZE=14
NUM_FRAMES=8
INPUT_SIZE=224
TUBELET_SIZE=2
BATCH_SIZE=64

for SEED in 1
do
    for DATASET in K400 K600 K700 SSV2
    do
        for NUM_SHOTS in 30
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