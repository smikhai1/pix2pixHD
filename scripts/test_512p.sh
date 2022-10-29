#!/usr/bin/env bash

DATA_PATH=/datasets/aligned-cara/ffhq-test_set/
CKPTS_DIR=./checkpoints
EXP_NAME=aligned-cara-512-base_exp-fp16
RESULTS_DIR=./results/
IMG_SIZE=512
DEVICE=cuda


echo 160
for EPOCH in 50 70 90 110 120 130
do
  echo "Run inference for ${EPOCH}"
  python run_inference.py --data_path=$DATA_PATH \
                        --label_nc=0 --netG=global \
                        --no_instance --checkpoints_dir=$CKPTS_DIR --name=$EXP_NAME \
                        --which_epoch=${EPOCH} --results_dir="./${RESULTS_DIR}/${EXP_NAME}/${EPOCH}" \
                        --img_size=$IMG_SIZE --device=$DEVICE --interp_lib=pil --interp_type=bilin
done
