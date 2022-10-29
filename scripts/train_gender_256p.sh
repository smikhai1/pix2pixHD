#!/usr/bin/env bash

python train.py --name women2men-default_p2p-256 --verbose --batchSize=2 --loadSize=256 --label_nc=0 \
                --dataroot=./dataset/cartoons_full_v2_clean/ --resize_or_crop=resize --nThreads=2 --tf_log \
                --netG=global --no_instance