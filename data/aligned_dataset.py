import os.path
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

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'


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
