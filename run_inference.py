"""
python run_inference.py --data_path=.. --label_nc=0 --netG=global \
                        --no_instance --checkpoints_dir=./checkpoints --name=women2men \
                        --which_epoch=190 --results_dir=./results/ \
                        --img_size=512 --device=cuda --add_bckg --interp_lib=pil --interp_type=bilin \
                        --crop_type=v2

"""
from glob import glob
import numpy as np
import os
from tqdm import tqdm

from util.util import preprocess_image, postprocess_image
from util.misc import save_image, load_image
from options.test_options import TestOptions
from models.models import create_model



def inference(opt):

    model = create_model(opt)
    model.to(opt.device)

    # making dir for the results
    imgs_result_dir = os.path.join(opt.results_dir, 'imgs')
    imgs_src_result_dir = os.path.join(opt.results_dir, 'src+res')
    os.makedirs(imgs_result_dir, exist_ok=True)
    os.makedirs(imgs_src_result_dir, exist_ok=True)


    # start transformation
    for img_path in tqdm(glob(os.path.join(opt.data_path, '*.*'))):
        img = load_image(img_path, to_rgb=False, size=opt.img_size)
        img_name = os.path.basename(img_path)
        img_proc = preprocess_image(img, device=opt.device)

        fake_img = model.inference(img_proc)
        fake_img = postprocess_image(fake_img)
        merged = np.concatenate((img, fake_img[..., ::-1]), axis=1)

        save_image(os.path.join(imgs_result_dir, img_name), fake_img, to_bgr=True)
        save_image(os.path.join(imgs_src_result_dir, img_name), merged, to_bgr=False)


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    inference(opt)