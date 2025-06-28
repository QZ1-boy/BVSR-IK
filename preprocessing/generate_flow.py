import os
import cv2
import glob
import torch
import argparse
import numpy as np

from raft import RAFT_bk
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]   # 
    rad = np.sqrt(np.square(u) + np.square(v))
    # print("u v",u.shape, v.shape)
    a = np.arctan2(-v, -u)/np.pi
    # print("a",a.shape)
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        # ch_idx =  2-i
        flow_image[:,:,ch_idx] = np.floor(255 * col)  # 255
        # flow_image[:,:,ch_idx] = np.floor(col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    # 归一化
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def viz_flow(flo, savepath):
    # print("flo oo",flo.shape) 
    b, c, higth, width = flo.shape
    # flo_tmp = torch.zeros([b,c,higth*4,width*4])  
    # img = img[0].permute(1,2,0).cpu().numpy()
    flow_0 = flo[0].permute(1,2,0).detach().cpu().numpy()
    flow_0 = flow_to_image(flow_0)
    flow_0_img = Image.fromarray(flow_0)  # flo[:, :, [2,1,0]]
    flow_0_img.save(savepath)


def read_img(filename):
    img = cv2.imread(filename)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1).cuda()
    img = img.unsqueeze(0)
    return img


def write_flow(flow, filename):
    flow = flow.permute(0, 2, 3, 1)
    flow = flow.squeeze(0).cpu().detach().numpy()

    path = os.path.dirname(filename)
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    np.save(filename, flow)
    return


def check_img_size(x, window_size):
    _, _, h, w = x.size()
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), "constant", 0)
    return x


def generate_flow(dir_path):
    dist_list = [1] # for FMA-Net w/ T=3  T=5 2

    parser = argparse.ArgumentParser()
    # --model ./pretrained/raft-sintel.pth --mixed_precision
    parser.add_argument('--model', default='/share3/home/zqiang/FMA-Net-main/preprocessing/pretrained/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = RAFT_bk()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.cuda()
    model.eval()

    dir_path = sorted(glob.glob(os.path.join(dir_path, '*')))
    # print('dir_path',dir_path)

    for dir_name in dir_path:
        img_list = sorted(glob.glob(os.path.join(dir_name, '*.png')))
        for dist in dist_list:
            for idx in tqdm(range(len(img_list))):
                if idx < dist or idx + dist > len(img_list) - 1:
                    continue
                img0 = read_img(img_list[idx - dist])
                img1 = read_img(img_list[idx])
                img2 = read_img(img_list[idx + dist])

                _, _, h, w = img0.shape
                img0 = check_img_size(img0, window_size=16)
                img1 = check_img_size(img1, window_size=16)
                img2 = check_img_size(img2, window_size=16)

                flow1_0 = model(img1, img0)[-1]
                flow1_2 = model(img1, img2)[-1]

                flow1_0 = flow1_0[:, :, :h, :w]
                flow1_2 = flow1_2[:, :, :h, :w]

                img0_name = os.path.basename(img_list[idx - dist]).split('.')[0]
                img1_name = os.path.basename(img_list[idx]).split('.')[0]
                img2_name = os.path.basename(img_list[idx + dist]).split('.')[0]

                # flow1_0_name = os.path.join(dir_name.replace('val_sharp_bicubic', 'val_flow_bicubic'), f'{img1_name}_{img0_name}.npy')
                # flow1_2_name = os.path.join(dir_name.replace('val_sharp_bicubic', 'val_flow_bicubic'), f'{img1_name}_{img2_name}.npy')
                flow1_0_img_name = os.path.join(dir_name.replace('LR_blurdown_x4', 'LR_blurdown_x4_flow'), f'{img1_name}_{img0_name}.png')
                flow1_2_img_name = os.path.join(dir_name.replace('LR_blurdown_x4', 'LR_blurdown_x4_flow'), f'{img1_name}_{img2_name}.png')

                # write_flow(flow1_0, flow1_0_name)
                # write_flow(flow1_2, flow1_2_name)
                viz_flow(flow1_0, flow1_0_img_name)
                viz_flow(flow1_2, flow1_2_img_name)





if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # generate_flow('/home/zhuqiang/FMA-Net-main/dataset/REDS4/train_sharp_bicubic/X4') # train_sharp_bicubic train_flow_bicubic
    # generate_flow('/share3/home/zqiang/FMA-Net-main/dataset/REDS4_BlurDown_Realistic_13/HR')

    # generate_flow('/share3/home/zqiang/FMA-Net-main/results/KCA_REDS_Realistic/Vid4_Realistic_bk/lr_corr')
    generate_flow('/share3/home/zqiang/FMA-Net-main/dataset/Vid4_BlurDown_Realistic_13/LR_blurdown_x4')