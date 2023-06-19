from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
from tqdm import tqdm

from util.util import preprocess_image, postprocess_image
from util.misc import save_image, load_image
from models.pix2pixHD_model import GlobalGenerator


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--ckpt_path', type=str)

    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--netG', type=str, default='global')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--n_downsample_global', type=int, default=4)
    parser.add_argument('--n_blocks_global', type=int, default=9)
    parser.add_argument('--n_local_enhancers', type=int, default=1)
    parser.add_argument('--n_blocks_local', type=int, default=3)
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--up_block_type', type=str, default='up_conv')

    args = parser.parse_args()

    return args


def inference(opt):

    model = GlobalGenerator(input_nc=3, output_nc=3, ngf=opt.ngf, netG=opt.netG,
                            n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                            n_local_enhancers=opt.n_local_enhancers, n_blocks_local=opt.n_blocks_local,
                            norm=opt.norm, device=opt.device, up_block_type=opt.up_block_type)
    model.load_ckpt(opt.ckpt_path)

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

        fake_img = model(img_proc)
        fake_img = postprocess_image(fake_img)
        merged = np.concatenate((img, fake_img[..., ::-1]), axis=1)

        save_image(os.path.join(imgs_result_dir, img_name), fake_img, to_bgr=True)
        save_image(os.path.join(imgs_src_result_dir, img_name), merged, to_bgr=False)


if __name__ == '__main__':
    args = parse_args()
    inference(args)
