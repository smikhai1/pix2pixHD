#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
EXP_NAME="default-config-balanced_celeba-mask_blending"

CKPTS_DIR="./storage/experiments/default-config-balanced_celeba-mask_blending/checkpoints"
DATA_ROOT="/home/msidorenko/datasets/stylegan_synth_data/nn_beauty-balanced_celeba"
TEST_IMAGES_DIR="/home/msidorenko/datasets/test-set-v2/faceapp_set/face-crops"
RESULTS_ROOT_DIR="./storage/experiments/default-config-balanced_celeba-mask_blending/test_set_results"

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