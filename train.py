from glob import glob
import time
import os
import os.path as osp

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import math
from tqdm import tqdm
from torch.cuda.amp import GradScaler

def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from data.aligned_dataset import load_mask
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util.util import preprocess_image, postprocess_image
from util.misc import save_image, load_image, create_images_grid


@torch.no_grad()
def run_inference(model, epoch, opt):
    save_dir = os.path.join(opt.results_dir, f'{epoch:0>5}')
    imgs_result_dir = os.path.join(save_dir, 'imgs')
    imgs_src_result_dir = os.path.join(save_dir, 'src+res')
    grids_dir = os.path.join(opt.results_dir, 'grids')

    os.makedirs(imgs_result_dir, exist_ok=True)
    os.makedirs(imgs_src_result_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    grid = []
    # img_name in tqdm(sorted(os.listdir(opt.test_data_dir))) -- use to preserve previous order
    for img_name in tqdm(sorted(os.listdir(opt.test_data_dir), key=lambda x: int(x.split('.')[0]))):
        if img_name.startswith('.'):
            continue
        img_path = os.path.join(opt.test_data_dir, img_name)
        try:
            img = load_image(img_path, to_rgb=False, size=opt.img_size)
        except Exception as ex:
            print(f'During loading image following exception was caught: {str(ex)}')
            continue

        img_proc = preprocess_image(img, device=opt.device)
        if opt.use_mask:
            mask_path = osp.join(opt.test_masks_dir, osp.splitext(img_name)[0] + '.png')
            size = img_proc.shape[-1]
            mask = load_mask(mask_path, size)[None]
            mask = mask.to(device=opt.device)
            img_proc = torch.cat((img_proc, mask), dim=1)

        if opt.fp16:
            img_proc = img_proc.to(dtype=torch.float16)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.fp16):
            fake_img = model.simple_inference(img_proc)
        fake_img = postprocess_image(fake_img)
        merged = np.concatenate((img, fake_img[..., ::-1]), axis=1)

        save_image(os.path.join(imgs_result_dir, img_name), fake_img, to_bgr=True)
        save_image(os.path.join(imgs_src_result_dir, img_name), merged, to_bgr=False)

        grid.append(fake_img.astype(np.uint8))

    grid = create_images_grid(grid, rows=opt.num_rows_in_grid)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(osp.join(grids_dir, f'grid-{epoch:>05}.jpg'), grid)


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
grad_scaler = GradScaler(enabled=opt.fp16)

if opt.fp16:
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    grad_scaler = GradScaler()
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    dataset_iter = iter(dataset)
    while True:
        try:
            data = next(dataset_iter)
        except StopIteration:
            break
        except OSError as ex:
            print(f'Some problems occurred when reading data from disk: {str(ex)}\n Continue training...')

        model.train()
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        mask = data['mask'] if opt.use_mask else None

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.fp16):
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']),
                                      infer=save_fake, mask=mask)

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + \
                     loss_dict.get('G_color_pres', 0.0) + loss_dict.get('G_color_stats', 0.0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            grad_scaler.scale(loss_G).backward()
        else:
            loss_G.backward()
        grad_scaler.step(optimizer_G)

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            grad_scaler.scale(loss_D).backward()
        else:
            loss_D.backward()
        grad_scaler.step(optimizer_D)
        grad_scaler.update()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

    ### run inference on the specified directory
    if opt.inference_epoch_freq > 0 and epoch % opt.inference_epoch_freq == 0:
        print('Inferencing at epoch ', epoch)
        model.eval()
        run_inference(model.module, epoch, opt)
        model.train()

