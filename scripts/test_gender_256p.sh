#!/usr/bin/env bash
python run_inference.py --data_path=/Users/mikhail/projects/work/surricat_vision/pix2pixHD/synth-gender-small-sample/women --label_nc=0 --netG=global \
                        --no_instance --checkpoints_dir=./checkpoints --name=women2men-default_p2p-256 \
                        --which_epoch=10 --results_dir=./results/w2m-default-256/80_train_3 \
                        --img_size=256 --device=cpu