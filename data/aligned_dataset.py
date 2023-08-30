import os.path

import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import cv2
import numpy as np


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

        if False:
            debug_dir = './debug_imgs'
            os.makedirs(debug_dir, exist_ok=True)
            mask_arr = (mask_arr * 255).astype(np.uint8)
            mask_arr = np.tile(mask_arr, reps=(1, 1, 3))
            concat = [np.array(A), np.array(B), mask_arr]
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
    if mask.shape[0] != size or mask.shape[1] != size:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if mask.ndim == 2:  # opencv can reduce channel dimension
        mask = mask[..., None]

    mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)
    if return_mask_arr:
        return mask_tensor, mask
    return mask_tensor
