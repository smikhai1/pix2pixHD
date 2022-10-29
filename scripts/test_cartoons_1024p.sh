#!/usr/bin/env bash
python run_inference.py --data_path=../UGATIT/dataset/beautyPeopleAnWe  --label_nc=0 --netG=local --ngf 32\
                        --no_instance --checkpoints_dir=./checkpoints --name=img2cartoons_1024 \
                        --which_epoch=70 --results_dir=./results/full_cartoons_v2_1024/70 \
                        --img_size=1024 --device=cuda --add_bckg --interp_lib=pil --interp_type=bilin \
                        --crop_type=v2