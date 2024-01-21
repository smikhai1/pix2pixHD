from argparse import ArgumentParser
from glob import glob
import os
import os.path as osp

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from util.util import preprocess_image, postprocess_image
from util.misc import save_image, load_image, create_images_grid
from models.pix2pixHD_model import GlobalGenerator
from face_parsing.model import BiSeNet

weight_dict = {'seg.pth': 'https://drive.google.com/file/d/1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62/view?usp=sharing'}


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
    parser.add_argument('--predict_offset', action='store_true', default=False)
    parser.add_argument('--num_rows_in_grid', type=int, default=7)

    args = parser.parse_args()

    return args


def array2tensor(img, device):
    img = torch.from_numpy(img) / 255.0
    img = img.permute(2, 0, 1)[None]
    img = img.to(device=device, dtype=torch.float32)
    return img


def download_weight(weight_path):
    if osp.isfile(weight_path):
        return
    gdown.download(weight_dict[osp.basename(weight_path)],
                   output=weight_path, fuzzy=True)


class BisenetInferencer(nn.Module):
    def __init__(self, segm_ckpt_fp, device, target_size=1024):
        super().__init__()

        self.segm_ckpt_fp = segm_ckpt_fp
        self.device = device
        self.target_size = target_size

        self.segm_model = None
        self._load_segmentation_model()

    def _load_segmentation_model(self):
        self.segm_model = BiSeNet(n_classes=16)

        if not osp.exists(self.segm_ckpt_fp):
            os.makedirs(osp.dirname(self.segm_ckpt_fp), exist_ok=True)
            download_weight(self.segm_ckpt_fp)
        self.segm_model.load_state_dict(torch.load(self.segm_ckpt_fp, map_location='cpu'))
        for param in self.segm_model.parameters():
            param.requires_grad = False
        self.segm_model.eval()
        self.segm_model.to(device=self.device)

    @torch.no_grad()
    def predict_segmentation(self, img):
        img = img.clamp(0.0, 1.0)
        face_seg_logits = self.segm_model(img)
        face_seg = torch.argmax(face_seg_logits, dim=1, keepdim=False).long()
        face_seg = face_seg[0].cpu().numpy().astype(np.uint8)
        h, w = face_seg.shape
        if h != self.target_size or w != self.target_size:
            face_seg = cv2.resize(face_seg, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_NEAREST)
        return face_seg

    def get_mask(self, img, face_parts):
        if isinstance(face_parts, int):
            face_parts = [face_parts]

        img = array2tensor(img, self.device)
        segm = self.predict_segmentation(img)

        # select and combine parts from the face segmentation
        mask = np.zeros_like(segm)
        for label in face_parts:
            mask = mask | (segm == label)
        if mask.ndim == 2:  # opencv can reduce channel dimension
            mask = mask[..., None]

        return mask


def get_bbox_from_mask(mask, size=None):
    h, w = mask.shape[:2]

    # get coordinates of pixels inside the mask
    ys, xs = np.where(mask.squeeze() > 0.5)
    coords = np.stack((ys, xs), axis=-1)
    if coords.shape[0] == 0:  # no-crop mode
        print('Take full image')
        return 0, 0, h - 1, w - 1

    # center of the mask == mean of mask pixesl coordintaes
    center = np.mean(coords, axis=0).astype(np.int32)
    yc, xc = center

    if size is None:
        size = max(np.max(ys) - np.min(ys), np.max(xs) - np.min(xs))

    if isinstance(size, int):
        size = (size, size)

    size_y, size_x = size
    y_top, x_left = yc - size_y // 2, xc - size_x // 2
    y_bottom, x_right = yc + size_y // 2, xc + size_x // 2

    # add sanity checks for bbox coordinates

    y_top, x_left = max(0, y_top), max(0, x_left)
    y_bottom, x_right = min(h - 1, y_bottom), min(w - 1, x_right)

    return y_top, x_left, y_bottom, x_right


def crop_image_with_bbox(img, bbox):
    y_top, x_left, y_bottom, x_right = bbox
    crop = img[y_top:y_bottom, x_left:x_right]
    if any(x == 0 for x in crop.shape):
        raise RuntimeError('Something went wrong when cropping, check shape: ', crop.shape)
    return crop


def paste_crop_into_image(crop, image, bbox):
    result = image.copy()
    y_top, x_left, y_bottom, x_right = bbox
    result[y_top:y_bottom, x_left:x_right] = crop
    return result


def smooth_mask(mask, sigma, dilation_rate):
    if dilation_rate is not None and dilation_rate > 0:
        kernel = np.ones((dilation_rate, dilation_rate), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma)
    return mask


def inference(opt):

    model = GlobalGenerator(input_nc=3, output_nc=3, ngf=opt.ngf, netG=opt.netG,
                            n_downsample_global=opt.n_downsample_global, n_blocks_global=opt.n_blocks_global,
                            n_local_enhancers=opt.n_local_enhancers, n_blocks_local=opt.n_blocks_local,
                            norm=opt.norm, device=opt.device, up_block_type=opt.up_block_type,
                            predict_offset=opt.predict_offset)
    model.load_ckpt(opt.ckpt_path)

    # making dir for the results
    results_dir = osp.join(opt.results_dir, 'imgs')
    concats_dir = osp.join(opt.results_dir, 'src+res')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(concats_dir, exist_ok=True)

    bisenet = BisenetInferencer(segm_ckpt_fp='./pretrained_models/seg.pth',
                                device=opt.device,
                                target_size=opt.img_size)

    grid = []
    # start transformation
    for img_path in tqdm(sorted(glob(os.path.join(opt.data_path, '*.*')))):
        # load image and crop
        img = load_image(img_path, to_rgb=True, size=opt.img_size)
        mask_bin = bisenet.get_mask(img, face_parts=[3, 4, 5])
        bbox = get_bbox_from_mask(mask_bin, size=(256, 512))  # <---- HARDCODED CROP SIZE
        input_crop = crop_image_with_bbox(img, bbox)

        # prepare cropped image for model and run it
        input_crop_t = preprocess_image(input_crop, device=opt.device, to_rgb=False)
        output_crop_t = model(input_crop_t)
        output_crop = postprocess_image(output_crop_t)

        # paste results back into original image
        result_with_border = paste_crop_into_image(output_crop, img, bbox)
        mask_smoothed = smooth_mask(mask_bin, sigma=7, dilation_rate=11)[..., None]
        result = mask_smoothed * result_with_border / 255.0 + (1.0 - mask_smoothed) * img / 255.0
        result = (255 * result).astype(np.uint8)

        # save results
        grid.append(result)

        img_name = os.path.basename(img_path)
        #mask_smoothed = (255 * mask_smoothed).astype(np.uint8)
        #mask_smoothed = np.tile(mask_smoothed, (1, 1, 3))

        concat = np.concatenate((img, result), axis=1)

        save_image(os.path.join(results_dir, img_name), result, to_bgr=True)
        save_image(os.path.join(concats_dir, img_name), concat, to_bgr=True)

    grid = create_images_grid(grid, rows=opt.num_rows_in_grid)
    grid_fp = osp.join(opt.results_dir, 'grid.jpg')
    save_image(grid_fp, grid, to_bgr=True)


if __name__ == '__main__':
    args = parse_args()
    inference(args)
