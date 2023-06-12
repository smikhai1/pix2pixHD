import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14714119, -0.28886916, 0.43601035],
                         [0.61497538, -0.51496512, -0.10001026]])
YUV_FROM_RGB_TENSOR = torch.from_numpy(YUV_FROM_RGB.T)


def rgb2yuv(rgb_tensor):
    global YUV_FROM_RGB_TENSOR

    YUV_FROM_RGB_TENSOR.to(rgb_tensor.device)
    YUV_FROM_RGB_TENSOR = YUV_FROM_RGB_TENSOR.type_as(rgb_tensor)

    yuv_tensor = rgb_tensor.permute(0, 2, 3, 1) @ YUV_FROM_RGB_TENSOR
    return yuv_tensor.permute(0, 3, 1, 2)


def get_random_inds(shape, samples_num):
    h_inds = np.random.randint(0, shape[-2], size=samples_num)
    w_inds = np.random.randint(0, shape[-1], size=samples_num)
    return h_inds, w_inds


def pairwise_distances_sq_l2(x, y, absolute_l2_clamp_range=(1e-5, 1e5)):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)

    y_t = torch.transpose(y, 0, 1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, *absolute_l2_clamp_range) / x.size(1)


def pairwise_distances_cos(x, y, eps=1e-16):
    x_norm = torch.sqrt((x ** 2).sum(1).reshape(-1, 1)) + eps
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y ** 2).sum(1).reshape(1, -1)) + eps

    dist = - torch.mm(x, y_t) / x_norm / y_norm + 1

    return dist


def compute_color_matching_loss(input, target, samples_num, yuv_channels):

    input = rgb2yuv(input)[:, yuv_channels]
    with torch.no_grad():
        target = rgb2yuv(target)[:, yuv_channels]

    inds_y_input, inds_x_input = get_random_inds(input.shape, samples_num)
    inds_y_target, inds_x_target = get_random_inds(target.shape, samples_num)

    input_grid = input[:, :,  inds_y_input, inds_x_input]
    target_grid = target[:, :, inds_y_target, inds_x_target]

    input_grid = input_grid.reshape(input_grid.shape[0], -1).transpose(1, 0)
    target_grid = target_grid.reshape(target_grid.shape[0], -1).transpose(1, 0)

    dists_mat = pairwise_distances_sq_l2(input_grid, target_grid) + pairwise_distances_cos(input_grid, target_grid)

    m1, _ = dists_mat.min(1)
    m2, _ = dists_mat.min(0)

    loss = torch.max(m1.mean(), m2.mean())

    return loss


def unbiased_cov(x):
    x = x.t()
    fact = 1.0 / (x.size(1) - 1)
    x = x - torch.mean(x, dim=1, keepdim=True)
    xt = x.t()
    return fact * x.matmul(xt).squeeze()


def extract_color_statistics(image, yuv_channels=(0, 1, 2), mask=None, sample_frac=1.0):
    image = rgb2yuv(image)
    image = image[:, yuv_channels]
    image = image.reshape(len(yuv_channels), -1).t()
    if mask is not None:
        mask = mask.reshape(-1) > 0.5
        image = image[mask]

    if sample_frac != 1:
        inds = np.random.choice(image.shape[0], size=int(image.shape[0] * sample_frac), replace=True)
        image = image[inds]

    colors_mean = image.mean(0)
    colors_std = image.std(0)
    colors_min = image.min(0)[0]
    colors_max = image.max(0)[0]

    return {
        'mean': colors_mean,
        'std': colors_std,
        'cov': unbiased_cov(image),
        'min': colors_min,
        'max': colors_max
    }


class StatsLoss(torch.nn.Module):
    def __init__(self, weights=None, yuv_channels=(1, 2)):
        super().__init__()
        self.weights = weights
        if self.weights is None:
            self.weights = {
                'mean': 1,
                'std': 1,
                'cov': 0,
                'min': 0,
                'max': 0,
            }
        self.yuv_channels = yuv_channels

    def forward(self, input_img, target_img):
        input_img = (input_img + 1.0) / 2.0
        target_img = (target_img + 1.0) / 2.0
        input_img_stats = extract_color_statistics(input_img, yuv_channels=self.yuv_channels)
        with torch.no_grad():
            target_img_stats = extract_color_statistics(target_img, yuv_channels=self.yuv_channels)

        loss = 0.0
        for key in input_img_stats:
            predicted_stat, ref_stat = input_img_stats[key], target_img_stats[key]
            loss = loss + F.mse_loss(predicted_stat, ref_stat) * self.weights.get(key, 1)

        return loss


class ColorMatchingLoss(nn.Module):
    def __init__(self, yuv_channels, num_samples):
        super().__init__()
        self.yuv_channels = yuv_channels
        self.num_samples = num_samples

    def forward(self, input_img, target_img):
        input_img = (input_img + 1.0) / 2.0
        target_img = (target_img + 1.0) / 2.0
        loss = compute_color_matching_loss(input_img, target_img, self.num_samples, self.yuv_channels)
        return loss
