import os
import sys
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

# def CharbonnierLoss(x, y, mean_res=False):
#     if x.shape != y.shape:
#         print("!!!")
#         print(x.shape, y.shape)
#     eps = 1e-4
#     diff = x - y
#     if mean_res:
#         batch_num = x.shape[0]
#         diff = diff.view(batch_num, -1).mean(1, keepdim=True)
#     loss = torch.sum(torch.sqrt(diff * diff + eps))

#     return loss


# ==========
# Loss & Metrics
# ==========

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps) + self.eps
        loss = torch.mean(error)
        return loss



class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows, frames):
        # pred_flows: [b t-1 2 h w,b t-1 2 h w] flows_forwards,flows_backwards
        #  gt_flows: same as pred_flows
        #  frames: b,t,3,h,w
        loss = 0
        warp_loss = 0
        h, w = pred_flows[0].shape[-2:]
        # masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:,:-1,...]
        frames1 = frames[:,1:,...]
        current_frames = [frames1, frames0]
        next_frames = [frames0, frames1]
        for i in range(len(pred_flows)):
            # print(pred_flows[i].shape)
            combined_flow = pred_flows[i]
            l1_loss = self.l1_criterion(pred_flows[i], gt_flows[i])
            # l1_loss += self.l1_criterion(pred_flows[i] * (1-masks[i]), gt_flows[i] * (1-masks[i])) / torch.mean((1-masks[i]))

            smooth_loss = smoothness_loss(combined_flow.reshape(-1,2,h,w))
            smooth_loss2 = second_order_loss(combined_flow.reshape(-1,2,h,w))
            
            warp_loss_i = ternary_loss(combined_flow.reshape(-1,2,h,w), gt_flows[i].reshape(-1,2,h,w), current_frames[i].reshape(-1,3,h,w), next_frames[i].reshape(-1,3,h,w)) 

            loss += l1_loss + smooth_loss + smooth_loss2

            warp_loss += warp_loss_i
            
        return loss, warp_loss



class FlowSimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows):
        # pred_flows: b t-1 2 h w
        loss = 0
        h, w = pred_flows[0].shape[-2:]
        h_orig, w_orig = gt_flows[0].shape[-2:]
        pred_flows = [f.view(-1, 2, h, w) for f in pred_flows]
        gt_flows = [f.view(-1, 2, h_orig, w_orig) for f in gt_flows]

        ds_factor = 1.0*h/h_orig
        gt_flows = [F.interpolate(f, scale_factor=ds_factor, mode='area') * ds_factor for f in gt_flows]
        for i in range(len(pred_flows)):
            loss += self.l1_criterion(pred_flows[i], gt_flows[i])

        return loss




# ==========
# Scheduler
# ==========


import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=[0],
                 restart_weights=[1],
                 last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]





def write(log, str):
    sys.stdout.flush()
    log.write(str + '\n')
    log.flush()


def denorm(x):

    
    x = x.cpu().detach().numpy()
    x = x.clip(0, 1) * 255.0
    x = np.round(x).astype(np.uint8)

    return x


# def Y_PSNR(img1, img2, border=0):
#     # img1 and img2 have range [0, 255]

#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')

#     diff = (img1 - img2).data.div(255)

#     shave = border
#     if diff.size(1) > 1:
#         convert = diff.new(1, 3, 1, 1)
#         convert[0, 0, 0, 0] = 65.738
#         convert[0, 1, 0, 0] = 129.057
#         convert[0, 2, 0, 0] = 25.064
#         diff.mul_(convert).div_(256)
#         diff = diff.sum(dim=1, keepdim=True)

#     valid = diff[:, :, shave:-shave, shave:-shave]
#     mse = valid.pow(2).mean()

#     return -10 * math.log10(mse)



def Y_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.squeeze()
    img2 = img2.squeeze()
    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    # print('img1_y 1 ', img1.shape)
    img1 = to_y_channel(img1)
    img2 = to_y_channel(img2)
    # img1 = bgr2ycbcr(img1, y_only=True)
    # img2 = bgr2ycbcr(img2, y_only=True)
    # print('img1_y 2 ', img1_y.shape)
    
    
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))




def RGB_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]

    img1 = img1.squeeze()
    img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



def gray_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]

    img1 = img1.squeeze()
    img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))




def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.detach().cpu().numpy()
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    img_y = img * 255.
    img_y = torch.from_numpy(img_y)
    return img_y


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.
    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    # img = img.detach().cpu().numpy()
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.134], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    # out_img = torch.from_numpy(out_img)
    return out_img



def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)




# --------------------------------------------
# SSIM
# --------------------------------------------
def RGB_SSIM(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.permute((1, 2, 0))
    img2 = img2.permute(1, 2, 0)
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')




def Y_SSIM(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.squeeze()
    img2 = img2.squeeze()
    img1 = img1.permute((1, 2, 0))
    img2 = img2.permute(1, 2, 0)

    img1 = to_y_channel(img1)
    img2 = to_y_channel(img2)
    
    # img1 = bgr2ycbcr(img1, y_only=True)
    # img2 = bgr2ycbcr(img2, y_only=True)

    
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def SSIM(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.permute((1, 2, 0))
    img2 = img2.permute(1, 2, 0)
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey):
    target_OF = cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    output_OF = cv2.calcOpticalFlowFarneback(pre_output_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    target_OF, ofy, ofx = crop_8x8(target_OF)
    output_OF, ofy, ofx = crop_8x8(output_OF)

    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm

    return OF_diff.mean()


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y + h, x:x + w]
    return crop_img, y, x


class Report():
    def __init__(self, save_dir, type, stage):
        filename = os.path.join(save_dir, f'stage{stage}_{type}_log.txt')

        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if os.path.exists(filename):
            self.logFile = open(filename, 'a')
        else:
            self.logFile = open(filename, 'w')

    def write(self, str):
        print(str)
        write(self.logFile, str)

    def __del__(self):
        self.logFile.close()


class Train_Report():
    def __init__(self):
        self.restoration_loss = []
        self.recon_loss = []
        self.hr_warping_loss = []
        self.lr_warping_loss = []
        self.flow_loss = []
        self.D_TA_loss = []
        self.R_TA_loss = []
        self.total_loss = []
        self.psnr = []
        self.recon_psnr = []
        self.num_examples = 0

    def update(self, batch_size, restoration_loss, recon_loss, hr_warping_loss, lr_warping_loss, flow_loss, D_TA_loss, R_TA_loss, total_loss):
        self.num_examples += batch_size
        self.restoration_loss.append(restoration_loss * batch_size)
        self.recon_loss.append(recon_loss * batch_size)
        self.hr_warping_loss.append(hr_warping_loss * batch_size)
        self.lr_warping_loss.append(lr_warping_loss * batch_size)
        self.flow_loss.append(flow_loss * batch_size)
        self.D_TA_loss.append(D_TA_loss * batch_size)
        self.R_TA_loss.append(R_TA_loss * batch_size)
        self.total_loss.append(total_loss * batch_size)

    def update_restoration_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    def update_recon_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.recon_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    def compute_mean(self):
        self.restoration_loss = np.sum(self.restoration_loss) / self.num_examples
        self.recon_loss = np.sum(self.recon_loss) / self.num_examples
        self.hr_warping_loss = np.sum(self.hr_warping_loss) / self.num_examples
        self.lr_warping_loss = np.sum(self.lr_warping_loss) / self.num_examples
        self.flow_loss = np.sum(self.flow_loss) / self.num_examples
        self.D_TA_loss = np.sum(self.D_TA_loss) / self.num_examples
        self.R_TA_loss = np.sum(self.R_TA_loss) / self.num_examples
        self.total_loss = np.sum(self.total_loss) / self.num_examples

    def result_str(self, lr_D, lr_R, period_time):
        self.compute_mean()
        if lr_R is None:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\n'
            str += f'D_TA Loss: {self.D_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\tlearning rate: {lr_D:.7f}\tTime: {period_time:.4f}'
        else:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Restoration Loss: {self.restoration_loss:.6f}\tLR Warping Loss: {self.lr_warping_loss:.6f}\tR_TA Loss: {self.R_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\n'
            str += f'learning rate (D): {lr_D:.7f}\tlearning rate (R): {lr_R:.7f}\tTime: {period_time:.4f}'
        return str

    def val_result_str(self, period_time):
        self.compute_mean()
        print('self.num_examples',self.num_examples)
        self.psnr = np.sum(self.psnr) / self.num_examples
        self.recon_psnr = np.sum(self.recon_psnr) / self.num_examples

        if self.psnr == 0:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Total Loss: {self.total_loss:.6f}\tTime: {period_time:.4f}\n'
            str += f'Recon LR blur PSNR: {self.recon_psnr:.5f}\n'
        else:
            str = f'Recon Loss: {self.recon_loss:.6f}\tHR Warping Loss: {self.hr_warping_loss:.6f}\tFlow Loss: {self.flow_loss:.8f}\tD_TA Loss: {self.D_TA_loss:.6f}\n'
            str += f'Restoration Loss: {self.restoration_loss:.6f}\tLR Warping Loss: {self.lr_warping_loss:.6f}\tR_TA Loss: {self.R_TA_loss:.6f}\tTotal Loss: {self.total_loss:.6f}\tTime: {period_time:.4f}\n'
            str += f'Recon LR blur PSNR-Y: {self.recon_psnr:.3f}\t HR PSNR-Y: {self.psnr:.3f}\n'

        return str



class Train_KCAReport():
    def __init__(self):
        self.restoration_loss = []
        self.corrected_loss = []
        self.total_loss = []
        self.psnr = []
        self.lr_psnr = []
        self.lr_corr_psnr = []
        self.num_examples = 0

    def update(self, batch_size, restoration_loss, corrected_loss, total_loss):
        self.num_examples += batch_size
        self.restoration_loss.append(restoration_loss * batch_size)
        self.corrected_loss.append(corrected_loss * batch_size)
        self.total_loss.append(total_loss * batch_size)

    def update_restoration_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    def update_correlation_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.lr_corr_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    def update_LR_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.lr_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    def update_recon_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.recon_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))
    

    def update_multi_kernel_metric(self, output, y):
        # print('output 0',output.shape,y.shape)
        output = output.squeeze(0)
        y = y.squeeze(0)
        C, T, H, W = output.shape
        ave_list = []
        for i in range(T):
            output1 = denorm(output[:,i,:,:])
            y1 = denorm(y[:,i,:,:])
            # print('output 0',output1.shape,y1.shape)
            ave_list.append(gray_PSNR(torch.tensor(output1), torch.tensor(y1)))
            output_save = self.process_kernel(torch.from_numpy(output1))
            y_save = self.process_kernel(torch.from_numpy(y1))

        self.kernel_psnr = np.sum(ave_list) / T

    def compute_mean(self):
        self.restoration_loss = np.sum(self.restoration_loss) / self.num_examples
        self.corrected_loss = np.sum(self.corrected_loss) / self.num_examples
        self.total_loss = np.sum(self.total_loss) / self.num_examples

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
    
    def result_str(self, lr_R, period_time):
        self.compute_mean()
        if lr_R is None:
            str = f'Total Loss: {self.total_loss:.6f}; learning rate: {lr_D:.7f}; Time: {period_time:.4f}'
        else:
            str = f'Restoration Loss: {self.restoration_loss:.6f}; Corrected Loss: {self.corrected_loss:.6f}; Total Loss: {self.total_loss:.6f}; learning rate (R):{lr_R:.6f}'
        return str

    def val_result_str(self, period_time):
        self.compute_mean()
        print('self.num_examples',self.num_examples)
        self.psnr = np.sum(self.psnr) / self.num_examples
        # self.lr_corr_psnr = np.sum(self.lr_corr_psnr) / self.num_examples

        if self.psnr == 0:
            str = f'Total Loss: {self.total_loss:.6f} Time: {period_time:.4f}'
        else:
            # str = f'Restoration Loss: {self.restoration_loss:.6f} Corrected Loss: {self.corrected_loss:.8f}; Total Loss: {self.total_loss:.6f}\n'
            str = f'HR PSNR-Y: {self.psnr:.3f} Time: {period_time:.4f}\n'

        return str





class Train_INRReport():
    def __init__(self):
        self.total_loss = []
        self.psnr = []
        # self.recon_psnr = []
        self.num_examples = 0

    def update(self, batch_size, total_loss):
        self.num_examples += batch_size
        self.total_loss.append(total_loss * batch_size)

    def update_restoration_metric(self, output, y):
        output = denorm(output)
        y = denorm(y)
        self.psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    # def update_correlation_metric(self, output, y):
    #     output = denorm(output)
    #     y = denorm(y)
    #     self.lr_corr_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    # def update_LR_metric(self, output, y):
    #     output = denorm(output)
    #     y = denorm(y)
    #     self.lr_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))

    # def update_recon_metric(self, output, y):
    #     output = denorm(output)
    #     y = denorm(y)
    #     self.recon_psnr.append(Y_PSNR(torch.tensor(output), torch.tensor(y)))
    
    # def update_kernel_metric(self, output, y):
    #     output = denorm(output)
    #     y = denorm(y)
    #     self.kernel_psnr.append(gray_PSNR(torch.tensor(output), torch.tensor(y)))


    # def update_multi_kernel_metric(self, output, y):
    #     output = output.squeeze(0)
    #     y = y.squeeze(0)
    #     C, T, H, W = output.shape
    #     ave_list = []
    #     for i in range(T):
    #         output1 = denorm(output[:,i,:,:])
    #         y1 = denorm(y[:,i,:,:])
    #         # print('output 0',output1.shape,y1.shape)
    #         ave_list.append(gray_PSNR(torch.tensor(output1), torch.tensor(y1)))
    #         output_save = self.process_kernel(torch.from_numpy(output1))
    #         y_save = self.process_kernel(torch.from_numpy(y1))

    #     self.kernel_psnr = np.sum(ave_list) / T

    def compute_mean(self):
        self.total_loss = np.sum(self.total_loss) / self.num_examples

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
    
    def result_str(self, lr_R, period_time):
        self.compute_mean()
        str = f'Total Loss: {self.total_loss:.6f}; learning rate (R):{lr_R:.6f}'
        return str

    def val_result_str(self, period_time):
        self.compute_mean()
        print('self.num_examples',self.num_examples)
        self.psnr = np.sum(self.psnr) / self.num_examples
        # self.recon_psnr = np.sum(self.recon_psnr) / self.num_examples
        # self.lr_psnr = np.sum(self.lr_psnr) / self.num_examples
        # self.lr_corr_psnr = np.sum(self.lr_corr_psnr) / self.num_examples
        # self.kernel_psnr = np.sum(self.kernel_psnr) / self.num_examples

        if self.psnr == 0:
            str = f'Total Loss: {self.total_loss:.6f} Time: {period_time:.4f}'
        else:
            str = f'Total Loss: {self.total_loss:.6f} HR PSNR-Y: {self.psnr:.3f}\n'

        return str







class TestReport():
    def __init__(self, epoch, base_dir):
        self.base_dir = base_dir
        self.epoch = epoch
        self.log_file_path = os.path.join(base_dir, 'avg_psnr_ssim_tof.txt')

        if os.path.exists(self.log_file_path):
            self.total_rgb_psnr_logFile = open(self.log_file_path, 'a')
        else:
            self.total_rgb_psnr_logFile = open(self.log_file_path, 'w')

        # self.total_rgb_psnr_logFile = open(os.path.join(base_dir, 'avg_psnr_ssim_tof.txt'), 'w')
        # self.total_rgb_ssim_logFile = open(os.path.join(base_dir, 'avg_rgb_ssim.txt'), 'w')
        # self.total_y_psnr_logFile = open(os.path.join(base_dir, 'avg_y_psnr.txt'), 'w')
        # self.total_y_ssim_logFile = open(os.path.join(base_dir, 'avg_y_ssim.txt'), 'w')
        # self.total_tOF_logFile = open(os.path.join(base_dir, 'avg_tOF.txt'), 'w')

        # self.total_rgb_psnr = []
        # self.total_rgb_ssim = []
        self.total_y_psnr = []
        self.total_y_ssim = []
        self.total_tOF = []

        # self.scene_rgb_psnr_logFile = None
        # self.scene_rgb_ssim_logFile = None
        # self.scene_y_psnr_logFile = None
        # self.scene_y_ssim_logFile = None
        # self.scene_tOF_logFile = None

        # self.scene_rgb_psnr = None
        # self.scene_rgb_ssim = None
        self.scene_y_psnr = None
        self.scene_y_ssim = None
        self.scene_tOF = None

        self.pre_gt_grey = None
        self.pre_output_grey = None

    def scene_init(self, scene_name):
        # self.scene_rgb_psnr_logFile = open(os.path.join(self.base_dir, scene_name + '_rgb_psnr.txt'), 'w')  # scene_name, 
        # self.scene_rgb_ssim_logFile = open(os.path.join(self.base_dir, scene_name + '_rgb_ssim.txt'), 'w')  #  scene_name, 
        # self.scene_y_psnr_logFile = open(os.path.join(self.base_dir, scene_name + '_y_psnr.txt'), 'w')  #  scene_name, 
        # self.scene_y_ssim_logFile = open(os.path.join(self.base_dir, scene_name + '_y_ssim.txt'), 'w')  #  scene_name, 
        # self.scene_tOF_logFile = open(os.path.join(self.base_dir, scene_name + '_tOF.txt'), 'w')  #  scene_name,

        # self.scene_rgb_psnr = []
        # self.scene_rgb_ssim = []
        self.scene_y_psnr = []
        self.scene_y_ssim = []
        self.scene_tOF = []

    def scene_del(self, scene_name):
        # write(self.scene_rgb_psnr_logFile, f'[Epoch:{self.epoch}] average RGB PSNR\t{np.mean(self.scene_rgb_psnr)}')
        # write(self.scene_rgb_ssim_logFile, f'[Epoch:{self.epoch}] average RGB SSIM\t{np.mean(self.scene_rgb_ssim)}')
        # write(self.scene_y_psnr_logFile, f'[Epoch:{self.epoch}] average Y PSNR\t{np.mean(self.scene_y_psnr)}')
        # write(self.scene_y_ssim_logFile, f'[Epoch:{self.epoch}] average Y SSIM\t{np.mean(self.scene_y_ssim)}')
        # write(self.scene_tOF_logFile, f'[Epoch:{self.epoch}] average tOF\t{np.mean(self.scene_tOF)}')

        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] [{scene_name}] average RGB PSNR: {np.mean(self.scene_rgb_psnr)}')
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] [{scene_name}] average RGB SSIM: {np.mean(self.scene_rgb_ssim)}')
        msg = f'[Epoch:{self.epoch}] [{scene_name}] average Y-PSNR/Y-SSIM/tOF: %.4f / %.4f/ %.4f' % (np.mean(self.scene_y_psnr),np.mean(self.scene_y_ssim),np.mean(self.scene_tOF))
        write(self.total_rgb_psnr_logFile, msg )
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] [{scene_name}] average Y SSIM: {np.mean(self.scene_y_ssim)}')
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] [{scene_name}] average tOF: {np.mean(self.scene_tOF)}')

        # self.scene_rgb_psnr_logFile.close()
        # self.scene_rgb_ssim_logFile.close()
        # self.scene_y_psnr_logFile.close()
        # self.scene_y_ssim_logFile.close()
        # self.scene_tOF_logFile.close()

        # self.scene_rgb_psnr_logFile = None
        # self.scene_rgb_ssim_logFile = None
        # self.scene_y_psnr_logFile = None
        # self.scene_y_ssim_logFile = None
        # self.scene_tOF_logFile = None

        # self.scene_rgb_psnr = None
        # self.scene_rgb_ssim = None
        self.scene_y_psnr = None
        self.scene_y_ssim = None
        self.scene_tOF = None

        self.pre_gt_grey = None
        self.pre_output_grey = None

    def update_metric(self, gt, output, filename):
        gt_grey = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        output_grey = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        ts = (2, 0, 1)
        gt = torch.Tensor(gt.transpose(ts).astype(float)).mul_(1.0)
        output = torch.Tensor(output.transpose(ts).astype(float)).mul_(1.0)

        gt = gt.unsqueeze(dim=0)
        output = output.unsqueeze(dim=0)

        # rgb_psnr = RGB_PSNR(output, gt, border=4) # border=4
        # rgb_ssim = RGB_SSIM(output, gt, border=4)
        y_psnr = Y_PSNR(output, gt, border=4)
        y_ssim = Y_SSIM(output, gt, border=4)

        # self.scene_rgb_psnr.append(rgb_psnr)
        # self.scene_rgb_ssim.append(rgb_ssim)
        self.scene_y_psnr.append(y_psnr)
        self.scene_y_ssim.append(y_ssim)

        # self.total_rgb_psnr.append(rgb_psnr)
        # self.total_rgb_ssim.append(rgb_ssim)
        self.total_y_psnr.append(y_psnr)
        self.total_y_ssim.append(y_ssim)

        # write(self.scene_rgb_psnr_logFile, f'{filename}\t{rgb_psnr}')
        # write(self.scene_rgb_ssim_logFile, f'{filename}\t{rgb_ssim}')
        # write(self.scene_y_psnr_logFile, f'{filename}\t{y_psnr}')
        # write(self.scene_y_ssim_logFile, f'{filename}\t{y_ssim}')

        if self.pre_gt_grey is not None:
            tOF = get_tOF(self.pre_gt_grey, gt_grey, self.pre_output_grey, output_grey)
            self.scene_tOF.append(tOF)
            self.total_tOF.append(tOF)
            # write(self.scene_tOF_logFile, f'{filename}\t{tOF}')

        self.pre_gt_grey = gt_grey
        self.pre_output_grey = output_grey

    def __del__(self):
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] total average RGB PSNR: {np.mean(self.total_rgb_psnr)}')
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] total average RGB SSIM: {np.mean(self.total_rgb_ssim)}')
        msg = f'[Epoch:{self.epoch}] total average Y-PSNR/Y-SSIM/tOF: %.4f/%.4f/%.4f' % (np.mean(self.total_y_psnr), np.mean(self.total_y_ssim),np.mean(self.total_tOF))
        # write(self.total_rgb_psnr_logFile, msg)
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] total average Y-PSNR/Y-SSIM/tOF: %.4f / %.4f/ %.4f' % ({np.mean(self.total_y_psnr)},{np.mean(self.total_y_ssim)},{np.mean(self.total_tOF)}))
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] total average Y SSIM: {np.mean(self.total_y_ssim)}')
        # write(self.total_rgb_psnr_logFile, f'[Epoch:{self.epoch}] total average tOF: {np.mean(self.total_tOF)}')

        # msg = f'[Epoch:{self.epoch}] total average RGB-PSNR/RGB-SSIM/Y-PSNR/Y-SSIM/tOF: %.4f/%.4f/%.4f/%.4f/%.4f' % (np.mean(self.total_rgb_psnr), np.mean(self.total_rgb_ssim), np.mean(self.total_y_psnr), np.mean(self.total_y_ssim),np.mean(self.total_tOF))

        print(msg)
        write(self.total_rgb_psnr_logFile, f'{msg}')

        self.total_rgb_psnr_logFile.close()
        # self.total_rgb_ssim_logFile.close()
        # self.total_y_psnr_logFile.close()
        # self.total_y_ssim_logFile.close()
        # self.total_tOF_logFile.close()


        
class SaveManager():
    def __init__(self, config):
        self.config = config

    def save_batch_images(self, src, batch_size, step):
        num = 5 if batch_size > 5 else batch_size
        dir = self.config.log_dir
        filename = os.path.join(dir, f'{step:08d}.png')
        scale = self.config.scale

        c, h, w = src[-1][0].shape
        log_img = np.zeros((c, h * num, w * len(src)), dtype=np.uint8)
        for i in range(num):
            for j in range(len(src)):
                tmp = denorm(src[j][i])
                if tmp.shape[1] < h:
                    tmp = np.transpose(tmp, (1, 2, 0))
                    tmp = cv2.resize(tmp, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    tmp = np.transpose(tmp, (2, 0, 1))
                log_img[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = tmp

        self.save_image(log_img, filename)

    def save_image(self, src, filename):
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        # #print('src',src.shape)
        t, c, h, w = src.shape
        src = np.transpose(src[t//2, :, :, :], (1, 2, 0))
        # src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, src)
