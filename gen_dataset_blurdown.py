import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
import scipy.io as scio
from scipy.ndimage import measurements, interpolation
from pathlib import Path
import random

def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("mkdir:", dir)


def kernel2png(kernel):
    kernel = cv2.resize(kernel, dsize=(0, 0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
    kernel = np.clip(kernel, 0, np.max(kernel))
    kernel = kernel / np.sum(kernel)
    mi = np.min(kernel)
    ma = np.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = np.round(np.clip(kernel * 255., 0, 255))
    kernel_png = np.stack([kernel, kernel, kernel], axis=2).astype('uint8')
    return kernel_png


def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def plot_kernel(gt_k_np, savepath):
    plt.clf()
    f, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    # im = ax[0, 1].imshow(out_k_np, vmin=0, vmax=out_k_np.max())
    # plt.colorbar(im, ax=ax[0, 1])
    # ax[0, 0].set_title('GT')
    # ax[0, 1].set_title('PSNR: {:.2f}'.format(calculate_psnr(gt_k_np, out_k_np, True)))

    plt.savefig(savepath)


def add_gaussian_noise_numpy(input_numpy, level=5, range=255.):
    noise = np.random.randn(*input_numpy.shape) * range * 0.01 * level
    input_numpy = input_numpy.astype('float32')
    out = input_numpy + noise
    out = np.round(np.clip(out, 0, range)).astype('uint8')
    return out


def get_blur_kernel_gaussian(trian=True):
    if trian:
        gaussian_sigma = random.choice(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    else:
        gaussian_sigma = 1.2
    gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
    kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
    # kernel = cv2.resize(kernel, dsize=(13, 13), interpolation=cv2.INTER_CUBIC)
    # kernel = np.clip(kernel, 0, np.max(kernel))
    # kernel = kernel / np.sum(kernel)
    return kernel


def get_blur_kernel_realistic(scale=4, kernel_size=31, noise_level=0.25, need_ksize=13):
    def kernel_shift(_kernel):
        # Function for centering a kernel
        # There are two reasons for shifting the kernel:,
        # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know",
        #    the degradation process included shifting so we always assume center of mass is center of the kernel.",
        # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first",
        #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the",
        #    top left corner of the first pixel. that is why different shift size needed between od and even size.",
        # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:",
        # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.",

        # First calculate the current center of mass for the kernel",
        current_center_of_mass = measurements.center_of_mass(_kernel),

        # The second (\"+ 0.5 * ....\") is for applying condition 2 from the comments above",
        # wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))
        wanted_center_of_mass = np.array(_kernel.shape) / 2

        # Define the shift vector for the kernel shifting (x,y)",
        shift_vec = (wanted_center_of_mass - current_center_of_mass)[0]

        kernel_shift = interpolation.shift(_kernel, shift_vec)

        # Finally shift the kernel and return",
        return kernel_shift

    scale = np.array([scale, scale])
    avg_sf = np.mean(scale)  # this is calculated so that min_var and max_var will be more intutitive\n",
    min_var = 0.6 * avg_sf  # variance of the gaussian kernel will be sampled between min_var and max_var\n",
    max_var = 5 * avg_sf
    k_size = np.array([kernel_size, kernel_size])  # size of the kernel, should have room for the gaussian\n",

    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix",
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)

    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta\n",
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T,
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)\n",
    MU = k_size // 2 + 0.5 * (scale - k_size % 2)
    MU = MU[None, None, :, None]
    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel\n",
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    # shift the kernel so it will be centered\n",
    raw_kernel_centered = kernel_shift(raw_kernel)
    # Normalize the kernel and return\n",
    raw_kernel_centered[raw_kernel_centered < 0] = 0
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    kernel = cv2.resize(kernel, dsize=(need_ksize, need_ksize), interpolation=cv2.INTER_CUBIC)
    kernel = np.clip(kernel, 0, np.max(kernel))
    kernel = kernel / np.sum(kernel)

    return kernel


def get_lr_blur_down(img_gt, kernel, downsample=True):
    img_gt = np.array(img_gt).astype('float32')
    gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(gt_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

    if downsample:
        lrx4_blur_tensor = blur_tensor[:, :, ::4, ::4]
        lrx4_blur_tensor = lrx4_blur_tensor.clamp(0, 255).round()
        lrx4_blur = lrx4_blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8')
        return lrx4_blur
    else:
        blur_tensor = blur_tensor.clamp(0, 255).round()
        blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8')
        return blur



def get_lr_blur_down_noise(img_gt, kernel, set_noise, downsample=True, random=True):
    img_gt = np.array(img_gt).astype('float32')
    gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(gt_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

    if downsample:
        lrx4_blur_tensor = blur_tensor[:, :, ::4, ::4]
        B, C, H_lr, W_lr = lrx4_blur_tensor.size()
        noise_level = torch.rand(B, 1, 1, 1, 1).to(lrx4_blur_tensor.device) * set_noise if random else set_noise
        noise = torch.randn_like(lrx4_blur_tensor).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
        lrx4_blur_tensor.add_(noise)
        lrx4_blur_tensor = lrx4_blur_tensor.clamp(0, 255).round()
        lrx4_blur = lrx4_blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8')
        return lrx4_blur
    else:
        B, C, H_lr, W_lr = blur_tensor.size()
        noise_level = torch.rand(1, 1, 1, 1, 1).to(blur_tensor.device) * set_noise if random else set_noise
        noise = torch.randn_like(blur_tensor).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
        blur_tensor.add_(noise)
        blur_tensor = blur_tensor.clamp(0, 255).round()
        blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8')
        return blur


def gene_dataset_blurdown(HR_root, save_root, type='Gaussian', set_noise=5):
    save_root_kernel = os.path.join(save_root, 'kernel')
    save_root_LRx4 = os.path.join(save_root, 'LR_blurdown_x4')
    save_root_HR = os.path.join(save_root, 'HR')
    handle_dir(save_root)
    handle_dir(save_root_kernel)
    handle_dir(save_root_LRx4)
    handle_dir(save_root_HR)

    video_names = sorted(os.listdir(HR_root))
    for vn in video_names:
        handle_dir(os.path.join(save_root_kernel, vn))
        handle_dir(os.path.join(save_root_LRx4, vn))
        handle_dir(os.path.join(save_root_HR, vn))

        frame_names = sorted(os.listdir(os.path.join(HR_root, vn)))

        for fn in frame_names:
            if type == 'Gaussian':
                kernel = get_blur_kernel_gaussian(trian=False)
            elif type == 'Realistic':
                kernel = get_blur_kernel_realistic(scale=4, need_ksize=13)
            elif type == 'Realistic_noise':
                kernel = get_blur_kernel_realistic(scale=4, need_ksize=13)
            else:
                raise NotImplementedError
            HR_img = cv2.imread(os.path.join(HR_root, vn, fn))
            HR_img = cv2.cvtColor(HR_img,cv2.COLOR_BGR2RGB)
            LRx4 = get_lr_blur_down(HR_img, kernel, downsample=True)
            # LRx4 = get_lr_blur_down_noise(HR_img, kernel, set_noise, downsample=True, random=False)

            basename = fn.split(".")[0]
            HR_img = cv2.cvtColor(HR_img, cv2.COLOR_RGB2BGR)
            LRx4 = cv2.cvtColor(LRx4, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_root_HR, vn, "{}.png".format(basename)), HR_img)
            cv2.imwrite(os.path.join(save_root_LRx4, vn, "{}.png".format(basename)), LRx4)
            scio.savemat(os.path.join(save_root_kernel, vn, "{}.mat".format(basename)), {'kernel': kernel})
            # kernel = scio.loadmat(os.path.join(save_root_kernel, vn, "{}.mat".format(basename)), {'kernel': kernel})['kernel'][...,np.newaxis]
            kernel_path = os.path.join(save_root_kernel, vn, "{}.mat".format(basename))
            kernel_ = process_kernel(kernel[...,np.newaxis])
            kernel_vizpath = kernel_path.replace('kernel','kernel_viz').replace('mat','png')
            dirname = os.path.dirname(kernel_vizpath)
            if not os.path.exists(dirname):
                Path(dirname).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(kernel_vizpath, kernel_)

            print("{}-{} done!".format(vn, fn))


def process_kernel(kernel):
    mi = np.min(kernel)
    ma = np.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = np.concatenate([kernel, kernel, kernel], axis=2)
    # kernel = kernel.mul(255.).clamp(0, 255).round()
    kernel = (255.0 * kernel).clip(0, 255).round()
    return kernel


if __name__ == '__main__':
    gene_dataset_blurdown(
        HR_root='../dataset/UDM10/HR',  #  REDS4_BlurDown_Realistic_13
        save_root='../dataset/UDM10_BlurDown_Gaussian_13',
        type='Gaussian',  #  Realistic  Gaussian Realistic_noise
        set_noise=0
    )
