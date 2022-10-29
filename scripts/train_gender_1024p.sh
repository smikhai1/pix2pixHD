#!/usr/bin/env bash

python train.py --name men2women_1024 --verbose --batchSize 1 --loadSize 1024 --label_nc 0 \
                --dataroot ./dataset/cartoons_full_v2_clean/ --resize_or_crop resize --nThreads 2 --tf_log \
                --netG=local --no_instance --ngf 32 --num_D 3 --load_pretrain checkpoints/img2cartoons --which_epoch latest \
                --niter 50 --niter_decay 50 --niter_fix_global 10