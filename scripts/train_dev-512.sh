#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
EXP_NAME="default-config"

CKPTS_DIR="./storage/experiments/${EXP_NAME}/checkpoints"
DATA_ROOT=""
TEST_IMAGES_DIR=""
RESULTS_ROOT_DIR="./storage/experiments/${EXP_NAME}/test_set_results"

BATCH_SIZE=12
IMAGE_SIZE=512
TEST_IMAGE_SIZE=512
NUM_WORKERS=10
NUM_EPOCHS=200
SAVE_FREQ=10
INFERENCE_EPOCH_FREQ=10


python train.py --name "${EXP_NAME}" --checkpoints_dir "${CKPTS_DIR}" --dataroot "${DATA_ROOT}" \
                 --batchSize $BATCH_SIZE --loadSize $IMAGE_SIZE --label_nc 0  \
                 --resize_or_crop resize --nThreads $NUM_WORKERS --niter "${NUM_EPOCHS}"\
                 --netG global --no_instance --verbose --save_epoch_freq "${SAVE_FREQ}"\
                 --inference_epoch_freq $INFERENCE_EPOCH_FREQ \
                 --test_data_dir "${TEST_IMAGES_DIR}" \
                 --results_dir "${RESULTS_ROOT_DIR}" --img_size $TEST_IMAGE_SIZE --fp16