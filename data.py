import os
import cv2
import glob
import torch
import random
# import imageio
import math
import glob
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import measurements, interpolation
import utils_dist

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass



def get_dataset(config, type):
    if 'FMA' in config.save_dir:       
        if config.dataloader == 'REDS_Gaussian':
            print('REDS_Gaussian')
            data = FMA_REDS_Dataset_Gaussian(config, type=type)
        elif config.dataloader == 'REDS_Realistic':
            data = FMA_REDS_Dataset_Realistic(config, type=type)
        else:
            data = FMA_REDS_Dataset(config, type=type)
    elif 'KCA_Vimeo' in config.save_dir:  
        if config.dataloader == 'Vimeo_Gaussian':
            data = KCA_Vimeo_Dataset_Gaussian(config, type=type)
        elif config.dataloader == 'Vimeo_Realistic':
            data = KCA_Vimeo_Dataset_Relasitic(config, type=type)
        else:
            data = FMA_REDS_Dataset(config, type=type)
    elif 'KCA_REDS' in config.save_dir:  
        if config.dataloader == 'REDS_Gaussian':
            data = KCA_REDS_Dataset_Gaussian(config, type=type)
        elif config.dataloader == 'REDS_Realistic':
            data = KCA_REDS_Dataset_Relasitic(config, type=type)
        else:
            data = FMA_REDS_Dataset(config, type=type)

    # config.gpu  = torch.cuda.device_count()
    
    if type == 'train':
        data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, drop_last=False, shuffle=True, num_workers=int(config.nThreads), pin_memory=True)
        # data_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, drop_last=True, shuffle=True, num_workers=int(config.nThreads), pin_memory=True)
        # train_sampler = utils_dist.DistSampler( dataset=data, num_replicas=config.gpu,  rank=rank)
        # data_loader = utils_dist.create_dataloader(dataset=data, opts_dict=opts_dict, sampler=train_sampler, phase='train', seed=opts_dict['train']['random_seed'])
    elif type == 'val':
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    elif type == 'test':
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    else:
        raise NotImplementedError('not implemented for this mode: {}!'.format(type))

    return data_loader





class KCA_REDS_Dataset_Gaussian:
    def __init__(self, config, type):
        self.config = config
        self.type = type
        self.num_seq = self.config.num_seq
        self.kernel_size = self.config.ds_kernel_size
        if self.kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(self.kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((self.kernel_size//2, self.kernel_size//2-1, self.kernel_size//2, self.kernel_size//2-1))

        bath_path = None
        # val_bath_path = None
        if type == 'train':
            bath_path = os.path.join(config.dataset_path, 'train_sharp')  # train_sharp_
        if type == 'val':
            # bath_path = os.path.join('./dataset/REDS4_BlurDown_Gaussian', 'HR')
            bath_path = os.path.join('./dataset/UDM10_BlurDown_Gaussian_13', 'HR')
        if type == 'test':
            bath_path = os.path.join('./dataset/REDS4_BlurDown_Gaussian', 'HR')

        self.seq_path = self.get_seq_path(bath_path)
        self.num_data = len(self.seq_path)

        print(f'num {type} dataset: {self.num_data}')

    def __getitem__(self, idx):
        # input GT
        hr_sharp_path = self.seq_path[idx]
        hr_sharp_seq = [cv2.imread(path) for path in hr_sharp_path]
        hr_sharp_seq = np.stack(hr_sharp_seq, axis=0)
        # print('filename', hr_sharp_seq.shape)

        if self.type == 'train':  #   or self.type == 'val'
            lr_sharp_path = [os.path.normpath(self.insert_dir(path.replace('sharp', 'sharp_bicubic'),'X4')) for path in hr_sharp_path]
            lr_sharp_seq = [cv2.imread(path) for path in lr_sharp_path] 
            lr_sharp_seq = np.stack(lr_sharp_seq, axis=0)

            # LR bicubic with blur seq

        if self.type == 'train':
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq, kernel = self.get_random_blur_patch(hr_sharp_seq, lr_sharp_seq)
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq = self.augment(lr_blur_seq, hr_sharp_seq, lr_sharp_seq)
            # kernel_tensor = torch.from_numpy(kernel).unsqueeze(0) # .permute(1,0,2,3)
            # kernel_tensor_pad = kernel_tensor
            #  _, _, ksize, _ = kernel_tensor.size()
            # print('kernel_tensor',kernel_tensor.shape, lr_blur_seq.shape)

            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq) # , kernel_tensor_pad

        if self.type == 'val':
            lr_blur_path = [os.path.normpath(path.replace('HR', 'LR_blurdown_x4')) for path in hr_sharp_path]
            lr_blur_seq = [cv2.imread(path) for path in lr_blur_path] 
            lr_blur_seq = np.stack(lr_blur_seq, axis=0)
            lr_sharp_path = [os.path.normpath(path.replace('HR', 'LR_sharp')) for path in hr_sharp_path]
            lr_sharp_seq = [cv2.imread(path) for path in lr_sharp_path] 
            lr_sharp_seq = np.stack(lr_sharp_seq, axis=0)

            # import scipy.io as sio
            # kernel_path = [os.path.normpath(path.replace('HR', 'kernel').replace('png', 'mat')) for path in hr_sharp_path]
            # kernel_seq = [sio.loadmat(path)['kernel'] for path in kernel_path] 
            # kernel_seq = np.stack(kernel_seq, axis=0)[...,np.newaxis]
            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq) # , self.np2tensor(kernel_seq)

        if self.type == 'test':

            lr_blur_path = [os.path.normpath(self.insert_dir(path.replace('sharp', 'blur_bicubic'),'X4')) for path in hr_sharp_path]
            lr_blur_seq = [cv2.imread(path) for path in lr_blur_path] 
            lr_blur_seq = np.stack(lr_blur_seq, axis=0)
            filename = lr_blur_path[self.num_seq // 2]
            print('filename', lr_blur_seq.shape)
            return self.np2tensor(lr_blur_seq), filename


    def get_random_blur_patch(self, hr_sharp_seq, lr_sharp_seq):
        ih, iw, c = lr_sharp_seq[0].shape
        tp = self.config.patch_size
        ip = tp // self.config.scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        (tx, ty) = (self.config.scale * ix, self.config.scale * iy)

        hr_sharp_seq = hr_sharp_seq[:, ty:ty + tp, tx:tx + tp, :]
        lr_sharp_seq = lr_sharp_seq[:, iy:iy + ip, ix:ix + ip, :]
        gts_list = [hr_sharp_seq[i, :, :, :] for i in range(self.num_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        gts_list = [gts_concat[:, :, i * self.config.in_channels:(i + 1) * self.config.in_channels] for i in range(self.num_seq)]

        # [one kernel, one frame]
        lr_blur_seq_list = []
        kernel_list = []
        for i in range(len(gts_list)):
            # kernel = self.random_isotropic_gaussian_kernel()
            kernel, kernel_siz = self.get_blur_kernel_gaussian()
            # kernel = torch.from_numpy(self.get_blur_kernel_gaussian()).unsqueeze(0).float()
            # print('kernel',kernel.shape)
            kernel_list.append(kernel_siz)
            lr_blur_seq_list.append(self.get_lr_blur_down(gts_list[i], kernel, self.config.scale))  # blur + downsample

        lr_blur_seq = np.stack(lr_blur_seq_list, axis=0)
        kernel_seq = np.stack(kernel_list, axis=0)

        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, kernel_seq


    def augment(self, lr_blur_seq, hr_sharp_seq, lr_sharp_seq):
        # random horizontal flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, :, ::-1, :]
            hr_sharp_seq = hr_sharp_seq[:, :, ::-1, :]
            lr_sharp_seq = lr_sharp_seq[:, :, ::-1, :]

        # random vertical flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, ::-1, :, :]
            hr_sharp_seq = hr_sharp_seq[:, ::-1, :, :]
            lr_sharp_seq = lr_sharp_seq[:, ::-1, :, :]


        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq

    
    def cal_smooth(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        smooth = np.mean(dst)
        return smooth

    def matlab_style_gauss2D(self, shape=(5, 5), sigma=0.5):
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

    def get_blur_kernel(self, trian=True):
        assert trian, "valuation should not use online data"
        gaussian_sigma = random.choice(
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
        gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 5)  + 1)
        kernel = self.matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
        return kernel

    def isotropic_gaussian_kernel(self, batch, kernel_size, sigma):
        ax = torch.arange(kernel_size).float() - kernel_size//2
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

        return kernel / kernel.sum([1,2], keepdim=True)
    
    def random_isotropic_gaussian_kernel(self, batch=1, kernel_size=13, sig_min=0.2, sig_max=4.0):
        x = torch.rand(batch) * (sig_max - sig_min) + sig_min
        kernel = self.isotropic_gaussian_kernel(batch, kernel_size, x)
        return kernel
    
    def add_gaussian_noise_numpy(self,input_numpy, level=5, range=255.):
        noise = np.random.randn(*input_numpy.shape) * range * 0.01 * level
        input_numpy = input_numpy.astype('float32')
        out = input_numpy + noise
        out = np.round(np.clip(out, 0, range)).astype('uint8')
        return out


    def get_blur_kernel_gaussian(self, need_ksize=13):
        gaussian_sigma = random.choice(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
        kernel = self.matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
        kernel_siz = cv2.resize(kernel, dsize=(need_ksize, need_ksize), interpolation=cv2.INTER_CUBIC)
        kernel_siz = np.clip(kernel_siz, 0, np.max(kernel_siz))
        kernel_siz = kernel_siz / np.sum(kernel_siz)
        return kernel, kernel_siz


    def get_blur_kernel_realistic(self,scale=4, kernel_size=31, noise_level=0.25, need_ksize=13):
        def kernel_shift(self,_kernel):
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
        # kernel = np.clip(kernel, 0, np.max(kernel))
        # kernel = kernel / np.sum(kernel)

        return kernel
    
    
    
    def get_lr_blur_down(self, img_gt, kernel, scale):
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
        blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0)

        blurdown = blur[::scale, ::scale, :]
        
        # add noise
        # if self.noise == True:
        #     _, C, H_lr, W_lr = blurdown.size()
        #     noise_level = torch.rand(B, 1, 1, 1, 1).to(blurdown.device) * self.noise if random else self.noise
        #     noise = torch.randn_like(blurdown).view(-1, N, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
        #     blurdown.add_(noise)

        return blurdown
    
    
    def np2tensor(self, x):
        # x shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (0, 3, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype('float64')).mul_(1.0)
        # normalization [0,1]
        x = x / 255.0

        return x

    def flow2tensor(self, flow):
        # flow shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (0, 3, 1, 2)
        flow = torch.Tensor(flow.transpose(ts).astype('float64')).mul_(1.0)

        return flow

    def get_seq_path(self, bath_path):
        seq_list = []
        # dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        print('dir_list',bath_path)
        dir_list = glob.glob(os.path.join(bath_path, '*'))
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.png')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def get_seq_mat_path(self, bath_path):
        seq_list = []
        # dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        # print('dir_list',dir_list)
        dir_list = glob.glob(os.path.join(bath_path, '*'))
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.mat')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def insert_dir(self, bath_path, dir):
        parts = os.path.split(bath_path)
        directory = parts[0]
        base_name = parts[1]
        
        folders = directory.split(os.path.sep)
        position = 4
        folders = folders[:position] + [dir] + folders[position:] + [base_name]
           
        new_path = os.path.join(*folders)
        return new_path

    def __len__(self):
        return self.num_data




class KCA_REDS_Dataset_Relasitic:
    def __init__(self, config, type):
        self.config = config
        self.type = type
        self.num_seq = self.config.num_seq
        self.kernel_size = self.config.ds_kernel_size
        if self.kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(self.kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((self.kernel_size//2, self.kernel_size//2-1, self.kernel_size//2, self.kernel_size//2-1))

        bath_path = None
        # val_bath_path = None
        if type == 'train':
            bath_path = os.path.join(config.dataset_path, 'train_sharp_')
        if type == 'val':
            bath_path = os.path.join('./dataset/REDS4_BlurDown_Realistic_13', 'HR')  #  REDS4_BlurDown_Realistic_13 REDS4_BlurDown_Realistic_13_noise_15
        if type == 'test':
            bath_path = os.path.join('./dataset/REDS4_BlurDown_Realistic_13', 'HR')
        
        self.seq_path = self.get_seq_path(bath_path)
        self.num_data = len(self.seq_path)

        print(f'num {type} dataset: {self.num_data}')

    def __getitem__(self, idx):
        # input GT
        hr_sharp_path = self.seq_path[idx]
        hr_sharp_seq = [cv2.imread(path) for path in hr_sharp_path]
        hr_sharp_seq = np.stack(hr_sharp_seq, axis=0)

        if self.type == 'train':  #   or self.type == 'val'
            lr_sharp_path = [os.path.normpath(self.insert_dir(path.replace('sharp', 'sharp_bicubic'),'X4')) for path in hr_sharp_path]
            lr_sharp_seq = [cv2.imread(path) for path in lr_sharp_path] 
            lr_sharp_seq = np.stack(lr_sharp_seq, axis=0)

            # LR bicubic with blur seq

        if self.type == 'train':
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq, kernel = self.get_random_blur_patch(hr_sharp_seq, lr_sharp_seq)
            lr_blur_seq, hr_sharp_seq, lr_sharp_seq = self.augment(lr_blur_seq, hr_sharp_seq, lr_sharp_seq)
            # print('kernel',lr_blur_seq.shape, kernel.shape)
            # kernel_tensor = torch.from_numpy(kernel).unsqueeze(0) # permute(1,0,2,3)
            # kernel_tensor_pad = kernel_tensor
            # _, _, ksize, _ = kernel_tensor.size()
            # print('kernel_tensor_pad',kernel_tensor_pad.shape, lr_sharp_seq.shape)

            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq) # , kernel_tensor_pad

        if self.type == 'val':
            lr_blur_path = [os.path.normpath(path.replace('HR', 'LR_blurdown_x4')) for path in hr_sharp_path]
            lr_blur_seq = [cv2.imread(path) for path in lr_blur_path] 
            lr_blur_seq = np.stack(lr_blur_seq, axis=0)
            lr_sharp_path = [os.path.normpath(path.replace('HR', 'LR_sharp')) for path in hr_sharp_path]
            lr_sharp_seq = [cv2.imread(path) for path in lr_sharp_path] 
            lr_sharp_seq = np.stack(lr_sharp_seq, axis=0)

            # import scipy.io as sio
            # kernel_path = [os.path.normpath(path.replace('HR', 'kernel').replace('png', 'mat')) for path in hr_sharp_path]
            # kernel_seq = [sio.loadmat(path)['kernel'] for path in kernel_path] 
            # kernel_seq = np.stack(kernel_seq, axis=0)[...,np.newaxis]
            
            return self.np2tensor(lr_blur_seq), self.np2tensor(hr_sharp_seq), self.np2tensor(lr_sharp_seq) # , self.np2tensor(kernel_seq)

        if self.type == 'test':

            lr_blur_path = [os.path.normpath(self.insert_dir(path.replace('sharp', 'blur_bicubic'),'X4')) for path in hr_sharp_path]
            lr_blur_seq = [cv2.imread(path) for path in lr_blur_path] 
            lr_blur_seq = np.stack(lr_blur_seq, axis=0)
            filename = lr_blur_path[self.num_seq // 2]
            return self.np2tensor(lr_blur_seq), filename


    def get_random_blur_patch(self, hr_sharp_seq, lr_sharp_seq):
        ih, iw, c = lr_sharp_seq[0].shape
        tp = self.config.patch_size
        ip = tp // self.config.scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        (tx, ty) = (self.config.scale * ix, self.config.scale * iy)

        hr_sharp_seq = hr_sharp_seq[:, ty:ty + tp, tx:tx + tp, :]
        lr_sharp_seq = lr_sharp_seq[:, iy:iy + ip, ix:ix + ip, :]
        gts_list = [hr_sharp_seq[i, :, :, :] for i in range(self.num_seq)]
        # print('gts_list', hr_sharp_seq.shape, len(gts_list))
        gts_concat = np.concatenate(gts_list, axis=2)
        gts_list = [gts_concat[:, :, i * self.config.in_channels:(i + 1) * self.config.in_channels] for i in range(self.num_seq)]

        # [one kernel, one frame]
        lr_blur_seq_list = []
        kernel_list = []
        for i in range(len(gts_list)):
            # kernel = self.random_isotropic_gaussian_kernel()
            kernel = torch.from_numpy(self.get_blur_kernel_realistic(scale=4, need_ksize=13)).float()
            # print('kernel',kernel.shape, len(gts_list[i]))
            kernel_list.append(kernel)
            # lr_blur_seq_list.append(self.get_lr_blur_down(gts_list[i], kernel, self.config.scale))  # blur + downsample
            lr_blur_seq_list.append(self.get_lr_blur_down_noise(gts_list[i], kernel, 25.0, self.config.scale))   # blur + downsample

        lr_blur_seq = np.stack(lr_blur_seq_list, axis=0)
        kernel_seq = np.stack(kernel_list, axis=0)

        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq, kernel_seq


    def augment(self, lr_blur_seq, hr_sharp_seq, lr_sharp_seq):
        # random horizontal flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, :, ::-1, :]
            hr_sharp_seq = hr_sharp_seq[:, :, ::-1, :]
            lr_sharp_seq = lr_sharp_seq[:, :, ::-1, :]

        # random vertical flip
        if random.random() < 0.5:
            lr_blur_seq = lr_blur_seq[:, ::-1, :, :]
            hr_sharp_seq = hr_sharp_seq[:, ::-1, :, :]
            lr_sharp_seq = lr_sharp_seq[:, ::-1, :, :]


        return lr_blur_seq, hr_sharp_seq, lr_sharp_seq

    
    def cal_smooth(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        smooth = np.mean(dst)
        return smooth

    def matlab_style_gauss2D(self, shape=(5, 5), sigma=0.5):
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

    def get_blur_kernel(self, trian=True):
        assert trian, "valuation should not use online data"
        # gaussian_sigma = random.choice(
        #     [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        # gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
        gaussian_sigma = random.choice(
            [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
        gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 5)  + 1)
        kernel = self.matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
        return kernel

    def isotropic_gaussian_kernel(self, batch, kernel_size, sigma):
        ax = torch.arange(kernel_size).float() - kernel_size//2
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

        return kernel / kernel.sum([1,2], keepdim=True)
    
    def random_isotropic_gaussian_kernel(self, batch=1, kernel_size=13, sig_min=0.2, sig_max=4.0):
        x = torch.rand(batch) * (sig_max - sig_min) + sig_min
        kernel = self.isotropic_gaussian_kernel(batch, kernel_size, x)
        return kernel
    
    
    def add_gaussian_noise_numpy(self,input_numpy, level=5, range=255.):
        noise = np.random.randn(*input_numpy.shape) * range * 0.01 * level
        input_numpy = input_numpy.astype('float32')
        out = input_numpy + noise
        out = np.round(np.clip(out, 0, range)).astype('uint8')
        return out


    def get_blur_kernel_gaussian(self,trian=True):
        if trian:
            gaussian_sigma = random.choice(
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        else:
            gaussian_sigma = 2.0
        gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
        kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
        return kernel

    def kernel_shift(self,_kernel):
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
    
    def get_blur_kernel_realistic(self,scale=4, kernel_size=31, noise_level=0.25, need_ksize=13):
        

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
        raw_kernel_centered = self.kernel_shift(raw_kernel)
        # Normalize the kernel and return\n",
        raw_kernel_centered[raw_kernel_centered < 0] = 0
        kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

        kernel = cv2.resize(kernel, dsize=(need_ksize, need_ksize), interpolation=cv2.INTER_CUBIC)
        kernel = np.clip(kernel, 0, np.max(kernel))
        kernel = kernel / np.sum(kernel)

        return kernel
    
    
    
    
    # def get_lr_blur_down(self, img_gt, kernel, scale):
    #     img_gt = np.array(img_gt).astype('float32')
    #     gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()
    #     B, C, H, W = gt_tensor.size()
    #     input_pad = self.pad(gt_tensor)
    #     H_p, W_p = input_pad.size()[-2:]

    #     if len(kernel.size()) == 2:
    #         input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
    #         kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))
    #         blur_tensor = F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
    #     else:
    #         input_CBHW = input_pad.view((1, C * B, H_p, W_p))
    #         kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
    #         kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))
    #         blur_tensor = F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))

    #     blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0)
    #     blurdown = blur[::scale, ::scale, :]

    #     return blurdown


    def get_lr_blur_down(self, img_gt, kernel, scale):
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
        blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0)

        blurdown = blur[::scale, ::scale, :]

        return blurdown

    def get_lr_blur_down_noise(self, img_gt, kernel, set_noise, scale):
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
        blurdown = blur_tensor[0][:, ::scale, ::scale]
        # print('blurdown',blurdown.shape)
        Channel, H_lr, W_lr = blurdown.shape

        random = True
        noise_level = torch.rand(1, 1, 1).to(blurdown.device) * set_noise if random else set_noise
        noise = torch.randn_like(blurdown).view(Channel, H_lr, W_lr).mul_(noise_level).view(Channel, H_lr, W_lr)
        blurdown.add_(noise)
        blurdown = blurdown.detach().numpy().transpose(1, 2, 0)

        return blurdown
    
    
    def np2tensor(self, x):
        # x shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (0, 3, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype('float64')).mul_(1.0)
        # normalization [0,1]
        x = x / 255.0

        return x

    def flow2tensor(self, flow):
        # flow shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = (0, 3, 1, 2)
        flow = torch.Tensor(flow.transpose(ts).astype('float64')).mul_(1.0)

        return flow

    def get_seq_path(self, bath_path):
        seq_list = []
        # dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        print('dir_list',bath_path)
        dir_list = glob.glob(os.path.join(bath_path, '*'))
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.png')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def get_seq_mat_path(self, bath_path):
        seq_list = []
        # dir_list = glob.glob(os.path.join(bath_path, '*/*/*/*'))
        # print('dir_list',dir_list)
        dir_list = glob.glob(os.path.join(bath_path, '*'))
        for dir in dir_list:
            frame_list = sorted(glob.glob(os.path.join(dir, '*.mat')))
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def insert_dir(self, bath_path, dir):
        parts = os.path.split(bath_path)
        directory = parts[0]
        base_name = parts[1]
        
        folders = directory.split(os.path.sep)
        position = 4
        folders = folders[:position] + [dir] + folders[position:] + [base_name]
           
        new_path = os.path.join(*folders)
        return new_path

    def __len__(self):
        return self.num_data




class Custom_Dataset:
    def __init__(self, config, path):
        self.path = path
        # self.num_seq = config.num_seq
        self.config = config
        self.num_seq = self.config.num_seq

        # bath_path = os.path.join(self.path)
        bath_path = os.path.join(self.path,'LR_blurdown_x4')

        self.seq_path = self.get_seq_path(bath_path)
        # print('self.seq_path',self.seq_path)
        # self.seq_path = self.seq_path.sort()
        self.num_data = len(self.seq_path)

        print(f'num custom dataset: {self.num_data}')

    def __getitem__(self, idx):
        # input
        lr_blur_path = self.seq_path[idx]
        # print('self.seq_path',lr_blur_path)
        lr_blur_seq = [cv2.imread(path) for path in lr_blur_path]
        lr_blur_seq = np.stack(lr_blur_seq, axis=0)
        # print('lr_blur_seq',lr_blur_seq.shape)

        filename = lr_blur_path[self.num_seq // 2]
        # print('filename',len(filename))
        return self.np2tensor(lr_blur_seq), filename

    def np2tensor(self, x):
        # x shape: [T, H, W, C]

        # reshape to [C, T, H, W]
        ts = ( 0, 3, 1, 2)
        x = torch.Tensor(x.transpose(ts).astype('float64')).mul_(1.0)
        # normalization [0,1]
        x = x / 255.0

        return x

    def get_seq_path(self, bath_path):
        seq_list = []
        
        sub_dir = [d for d in os.listdir(bath_path)]
        # print('sub_dir',sub_dir)
        for i_dir in range(len(sub_dir)):
            frame_list = sorted(glob.glob(os.path.join(bath_path, sub_dir[i_dir], '*.png')))
            # print('frame_list',frame_list)
            start = (self.num_seq - 1) // 2
            end = len(frame_list) - (self.num_seq - 1) // 2
            for i in range(start, end):
                frame_seq = []
                for seq_num in range(self.num_seq):
                    frame_seq.append(frame_list[i + seq_num - start])
                seq_list.append(frame_seq)
        return seq_list

    def __len__(self):
        return self.num_data

