#!/usr/bin/env bash

python train.py --name aligned-cara-1024-use_sn_in_D --verbose --batchSize 1 --loadSize 1024 --label_nc 0 \
                --dataroot=/datasets/aligned-cara/ --resize_or_crop resize --nThreads 4 \
                --netG=local --no_instance --ngf 32 --num_D 3 \
                --load_pretrain=checkpoints/aligned-cara-512-base_exp-fp16 --which_epoch 110 \
                --niter 50 --niter_decay 50 --niter_fix_global 10 --sn
