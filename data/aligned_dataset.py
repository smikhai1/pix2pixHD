import os.path

import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np

from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
import torchvision.transforms.functional as TF


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

        if opt.use_mask:
            self.masks_paths = sorted(make_dataset(opt.train_masks_dir))
            self.mask_size = opt.loadSize

        self.use_mask = opt.use_mask
        self.class_label = opt.class_label

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        ### load mask
        if self.use_mask:
            mask_path = self.masks_paths[index]
            mask, mask_arr = load_mask(mask_path, self.mask_size, class_label=self.class_label,
                                       return_mask_arr=True)
            input_dict['mask'] = mask

            # get square bbox around the mask region
            bbox = get_bbox_from_mask(mask_arr, size=self.opt.loadSize // 4)
            A_tensor_crop = make_crop(A_tensor, bbox)
            B_tensor_crop = make_crop(B_tensor, bbox)
            mask_crop = make_crop(mask, bbox)

            input_dict['label_crop'] = A_tensor_crop
            input_dict['image_crop'] = B_tensor_crop
            input_dict['mask_crop'] = mask_crop

        if False:
            debug_dir = './debug_imgs'
            os.makedirs(debug_dir, exist_ok=True)
            mask_arr = (mask_arr * 255).astype(np.uint8)
            mask_arr = np.tile(mask_arr, reps=(1, 1, 3))
            concat = [make_crop(np.array(A.resize((self.opt.loadSize, self.opt.loadSize))), bbox),
                      make_crop(np.array(B.resize((self.opt.loadSize, self.opt.loadSize))), bbox),
                      make_crop(mask_arr, bbox)
                      ]
            for i in range(len(concat)):
                if concat[i].shape[0] != self.mask_size or concat[i].shape[1] != self.mask_size:
                    concat[i] = cv2.resize(concat[i], (self.mask_size, self.mask_size))
            concat = np.concatenate(concat, axis=1)
            cv2.imwrite(os.path.join(debug_dir, f'{index}.jpg'), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'


def load_mask(mask_fp, size, class_label=10, return_mask_arr=False):
    mask = cv2.imread(mask_fp)
    if mask.ndim == 2:
        mask = mask[..., None]  # make it H x W x 1
    elif mask.shape[-1] > 1:
        mask = mask[..., [0]]

    mask = (mask == class_label).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)

    if mask.ndim == 2:  # opencv can reduce channel dimension
        mask = mask[..., None]

    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
    if return_mask_arr:
        return mask_tensor, mask
    return mask_tensor


def get_bbox_from_mask(mask, size):
    # get coordinates of pixels inside the mask
    ys, xs = np.where(mask.squeeze() > 0.5)
    coords = np.stack((ys, xs), axis=-1)

    # center of the mask == mean of mask pixesl coordintaes
    center = np.mean(coords, axis=0).astype(np.int32)
    yc, xc = center

    y_top, x_left = yc - size // 2, xc - size // 2
    y_bottom, x_right = yc + size // 2, xc + size // 2

    return y_top, x_left, y_bottom, x_right


def make_crop(input, bbox):
    if isinstance(input, np.ndarray):
        input_shape = input.shape[:2]
        bbox = clip_bbox(bbox, input_shape)
        y_top, x_left, y_bottom, x_right = bbox
        crop = input[y_top:y_bottom, x_left:x_right]
    elif isinstance(input, torch.Tensor):
        input_shape = input.shape[-2:]
        bbox = clip_bbox(bbox, input_shape)
        y_top, x_left, y_bottom, x_right = bbox
        crop = input[..., y_top:y_bottom, x_left:x_right]
    else:
        raise ValueError('Unsupported input type: ', type(input))
    return crop


def clip_bbox(bbox, shape):
    h, w = shape
    y_top, x_left, y_bottom, x_right = bbox

    y_top, x_left = max(0, y_top), max(0, x_left)
    y_bottom, x_right = min(h-1, y_bottom), min(w-1, x_right)

    return y_top, x_left, y_bottom, x_right


def to_tensor(arr, mean=None, std=None):
    if arr is None:
        return torch.empty(0)
    t = TF.to_tensor(arr)
    if mean is not None and std is not None:
        t = TF.normalize(t, mean=mean, std=std)
    return t


class AlignedDatasetWithAugs(AlignedDataset):
    def initialize(self, opt):
        super().initialize(opt)
        augs_config = OmegaConf.load(opt.augs_cfg_fp)

        self.src_tgt_augs = instantiate(augs_config['src_tgt_augs'])
        self.src_augs = instantiate(augs_config['src_augs'])

    def __getitem__(self, idx):
        src_path = self.A_paths[idx]
        src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)

        tgt_path = self.B_paths[idx]
        tgt_img = cv2.cvtColor(cv2.imread(tgt_path), cv2.COLOR_BGR2RGB)

        ### load mask
        if self.use_mask:
            mask_path = self.masks_paths[idx]
            _, mask = load_mask(mask_path, self.mask_size, class_label=self.class_label,
                                       return_mask_arr=True)

            # get square bbox around the mask region
            bbox = get_bbox_from_mask(mask, size=self.opt.loadSize // 4)
            src_img_crop = make_crop(src_img, bbox)
            tgt_img_crop = make_crop(tgt_img, bbox)
            mask_crop = make_crop(mask, bbox)

            augm_images = self.src_tgt_augs(image=src_img_crop,
                                            target_image=tgt_img_crop,
                                            mask=mask_crop)
            src_img, tgt_img = augm_images['image'], augm_images['target_image']
        else:
            augm_images = self.src_tgt_augs(image=src_img, target_image=tgt_img)
            src_img, tgt_img = augm_images['image'], augm_images['target_image']

        src_img_auged = self.src_augs(image=src_img)
        src_img = src_img_auged['image']

        if False:
            save_dir = './debugging'
            os.makedirs(save_dir, exist_ok=True)
            concat = np.concatenate((src_img, tgt_img), axis=1)
            cv2.imwrite(f'{save_dir}/{idx}.jpg', cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

        src_img = src_img.astype(np.float32) / 255.0
        tgt_img = tgt_img.astype(np.float32) / 255.0

        src_img = to_tensor(src_img, mean=0.5, std=0.5)
        tgt_img = to_tensor(tgt_img, mean=0.5, std=0.5)
        input_dict = {'label': src_img,
                      'inst': 0, 'image': tgt_img,
                      'feat': 0, 'path': src_path}

        return input_dict

    def name(self):
        return 'AlignedDatasetWithAugs'
