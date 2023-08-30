from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
import os.path as osp
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import skimage.transform

from util.util import preprocess_image, postprocess_image
from util.misc import save_image, load_image, create_images_grid
from models.pix2pixHD_model import GlobalGenerator
from data.aligned_dataset import load_mask, get_bbox_from_mask, make_crop


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--masks_dir', type=str)
    parser.add_argument('--class_label', type=int)

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
    parser.add_argument('--predict_offset', action='store_true', default=False)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--last_conv_zeros', action='store_true', default=False)
    parser.add_argument('--num_rows_in_grid', type=int, default=5)

    args = parser.parse_args()

    return args


def create_vignette_mask(shape, size=100, power=4):
    vx = 1 - np.abs(np.linspace(-1, 1, size)[None]) ** power
    vig = vx.T @ vx
    vig = vig[..., None]
    vig = skimage.transform.resize(vig, shape)
    return vig


def inference(opt):

    model = GlobalGenerator(input_nc=3, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG,
                            n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                            n_local_enhancers=opt.n_local_enhancers, n_blocks_local=opt.n_blocks_local,
                            norm=opt.norm, device=opt.device, up_block_type=opt.up_block_type,
                            predict_offset=opt.predict_offset, last_conv_zeros=opt.last_conv_zeros)
    model.load_ckpt(opt.ckpt_path)

    # making dir for the results
    imgs_result_dir = os.path.join(opt.results_dir, 'imgs')
    imgs_src_result_dir = os.path.join(opt.results_dir, 'src+res')
    os.makedirs(imgs_result_dir, exist_ok=True)
    os.makedirs(imgs_src_result_dir, exist_ok=True)
    grid_path = osp.join(opt.results_dir, 'grid.jpg')

    # start transformation
    grid = []
    for img_name in tqdm(sorted(os.listdir(opt.data_path), key=lambda x: int(x.split('.')[0]))):
        if img_name.startswith('.'):
            continue
        img_path = osp.join(opt.data_path, img_name)
        mask_path = osp.join(opt.masks_dir, osp.splitext(img_name)[0] + '.png')

        img = load_image(img_path, to_rgb=False, size=opt.img_size)
        size = img.shape[0]
        mask, mask_arr = load_mask(mask_path, size, class_label=opt.class_label, return_mask_arr=True)
        bbox = get_bbox_from_mask(mask_arr, size // 4)

        img_crop = make_crop(img, bbox)
        if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
            continue

        img_crop_proc = preprocess_image(img_crop, device=opt.device)
        warp_field = model(img_crop_proc)
        warp_field = warp_field.permute(0, 2, 3, 1)  # to [B, H, W, 2]

        # we obtain the predicted image by warping the input
        #   with the predicted warp field
        fake_img_crop = F.grid_sample(img_crop_proc, warp_field, mode='bilinear', align_corners=True)
        fake_img_crop = postprocess_image(fake_img_crop)
        fake_img_crop = fake_img_crop / 255.0
        fake_img_crop = fake_img_crop[..., ::-1]
        vign_mask = create_vignette_mask(fake_img_crop.shape[:2], power=6)
        fake_img_crop = fake_img_crop * vign_mask + img_crop / 255.0 * (1-vign_mask)
        fake_img_crop = (255 * fake_img_crop).astype(np.uint8)

        res = img.copy()
        y_top, x_left, y_bottom, x_right = bbox
        res[y_top:y_bottom, x_left:x_right] = fake_img_crop

        merged = np.concatenate((img, res), axis=1)

        save_image(os.path.join(imgs_result_dir, img_name), res, to_bgr=True)
        save_image(os.path.join(imgs_src_result_dir, img_name), merged, to_bgr=False)
        grid.append(res)

    grid = create_images_grid(grid, rows=opt.num_rows_in_grid)
    cv2.imwrite(grid_path, grid)


if __name__ == '__main__':
    args = parse_args()
    inference(args)
