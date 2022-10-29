#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python train.py --name vector_v1-base_exp-fp16 --verbose --batchSize=12 --loadSize=512 --label_nc=0 \
                --dataroot=/datasets/sg2-disitll-data/vector --resize_or_crop=resize --nThreads=10 \
                --netG=global --no_instance --fp16 #--tf_log
