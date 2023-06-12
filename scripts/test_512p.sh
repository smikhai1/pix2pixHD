#!/usr/bin/env bash


CKPTS_DIR=/home/smikhai1/projects/pix2pixHD/storage/experiments/default_config/checkpoints
EXP_NAME="default_config"
IMG_SIZE=512
DEVICE=cuda


DATA_PATH=/home/smikhai1/datasets/nn-beauty-test/face-crops
RESULTS_DIR=/home/smikhai1/projects/pix2pixHD/storage/experiments/default_config/nn_beauty_test_1/45
for EPOCH in 45
do
  echo "Run inference for ${EPOCH}"
  python run_inference.py --data_path=$DATA_PATH \
                        --label_nc=0 --netG=global \
                        --no_instance --checkpoints_dir=$CKPTS_DIR --name=$EXP_NAME \
                        --which_epoch=${EPOCH} --results_dir="${RESULTS_DIR}/${EXP_NAME}/${EPOCH}" \
                        --img_size=$IMG_SIZE --device=$DEVICE --interp_lib=pil --interp_type=bilin
done
