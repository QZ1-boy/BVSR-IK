import os
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
import numbers
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output

def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          padding_mode='reflect'))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



def plus_list(A, B):
    outs=[]
    for a, b in zip(A, B):
        outs.append(a+b)
    return outs

def forward_list(zs, f):
    outs=[]
    for z in zs:
        outs.append(f(z))
    return outs


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)

class Abs(nn.Module):
    def forward(self,x):
        return torch.abs(x)

class Exp(nn.Module):
    def __init__(self, w=0.5):
        super(Exp, self).__init__()
        self.w=w
    def forward(self, x):
        return torch.exp(self.w*x)

class Shift(nn.Module):
    def forward(self, x):
        x = x - x.min(dim=-2, keepdim=True).min(dim=-3, keepdim=True)
        return x


def shape2coordinate(shape=(3,3), device='cuda', normalize_range=(0.,1.)):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min, max=normalize_range
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

    return cord


class shapemotion2coordinate(nn.Module):
    def __init__(self, device='cuda', normalize_range=(0.,1.)):
        super().__init__()
        self.normalize_range = normalize_range
    def forward(self, shape, motion):
        h, w=shape
        x=torch.arange(0, h, device='cuda')
        y=torch.arange(0, w, device='cuda')
        x, y=torch.meshgrid(x, y)
        re_motion = F.interpolate(motion, (h, w))
        x_motion = re_motion[0, 0, :,:].squeeze(0)
        y_motion = re_motion[0, 1, :,:].squeeze(0)
        x = x  + x_motion
        y = y  + y_motion

        min, max=self.normalize_range
        x=x/(h-1)*(max-min)+min
        y=y/(w-1)*(max-min)+min
        cord=torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

        return cord


def shape2polar_coordinate(shape=(3,3), device='cuda'):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min=-1
    max=1
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=x+1j*y


    r=torch.abs(cord)/np.sqrt(2)
    theta=torch.angle(cord)
    theta_code=torch.cat([(torch.cos(theta).unsqueeze(-1)+1)/2, (torch.sin(theta).unsqueeze(-1)+1)/2], dim=-1)

    cord=torch.cat([r.unsqueeze(-1), theta_code], dim=-1)

    return cord


class KernelINR_Cord(nn.Module):
    def __init__(self, hidden_dim=64, w=1.):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(2, hidden_dim),
            # nn.ReLU(),
            Sine(w),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            Sine(w),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, cord):
        k=self.layers(cord).squeeze(-1)
        return k


class SEBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(SEBlock, self).__init__()

        layers = [Recurr_TransBlock(out_channel, num_heads=1) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out1, out2 = self.layers(x)
        return out1, out2 


class SDBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(SDBlock, self).__init__()

        layers = [Recurr_TransBlock(channel, num_heads=1) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Recurr_Attention(nn.Module):
    # Restormer (CVPR 2022) transposed-attnetion block
    # original source code: https://github.com/swz30/Restormer
    def __init__(self, dim, num_heads=4, bias= False):
        super(Recurr_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_qkv = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.to_qkv_h = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, h_0):
        b, c, h, w = x.shape
        q, k, v = torch.add(self.to_qkv(x), self.to_qkv_h(h_0)).chunk(3, dim = 1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class Recurr_TransBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor = 0.5, LayerNorm_type='WithBias', bias = False):
        super(Recurr_TransBlock, self).__init__()
        self.recurr_attn = Recurr_Attention(dim, num_heads, bias)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        Fw, F_h0 = x[0], x[1]
        F_h1 = self.recurr_attn(Fw, F_h0)
        F_o1 = Fw + F_h1 
        F_out = F_o1 + self.ffn1(F_o1)

        return F_out, F_h1 + F_h0


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class UNet_Spatial(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(UNet_Spatial, self).__init__()      
        
        self.Encoder = nn.ModuleList([
            SEBlock(base_ch, num_res),
            SEBlock(base_ch, num_res),
            SEBlock(base_ch, num_res),
        ])

        self.Decoder = nn.ModuleList([
            SDBlock(base_ch, num_res),
            SDBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch* 2, base_ch* 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=1, relu=True, stride=1)
        ])

        self.up1=BasicConv(base_ch * 2, base_ch  * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2=BasicConv(base_ch * 2, base_ch  * 2, kernel_size=4, relu=True, stride=2, transpose=True)

    def forward(self, x1, x2):
        '''Feature Extract 0'''
        x_1, x_2 = self.feat_extract[0](torch.cat([x1, x2], dim=1)).chunk(2, dim=1)
        res1, hidden1 = self.Encoder[0]([x_1, x_2])

        '''Down Sample 1'''
        z, hidden1_1 = self.feat_extract[1](torch.cat([res1, hidden1], dim=1)).chunk(2, dim=1)
        res2, hidden2 = self.Encoder[1]([z, hidden1_1])

        '''Down Sample 2'''
        z, hidden2_1 = self.feat_extract[2](torch.cat([res2, hidden2], dim=1)).chunk(2, dim=1)
        res3, hidden3 = self.Encoder[2]([z, hidden2_1])

        # deepz=res3

        '''Up Sample 2'''
        # z=self.up1(res3)
        z, hidden3_1 =self.up1(torch.cat([z, hidden3], dim=1)).chunk(2, dim=1)
        z, hidden3_1 = self.feat_extract[3](torch.cat([z + res2, hidden3_1], dim=1)).chunk(2, dim=1) # torch.cat([z, res2], dim=1))
        z, hidden3_2 = self.Decoder[0]([z, hidden3_1+hidden2])

        '''Up Sample 1'''
        # z=self.up2(z)
        z, hidden3_3=self.up2( torch.cat([z, hidden3_2], dim=1)).chunk(2, dim=1)
        z, hidden3_4 = self.feat_extract[4](torch.cat([z + res1, hidden3_3], dim=1)).chunk(2, dim=1)  # torch.cat([z, res1], dim=1)
        z, hidden3_5 = self.Decoder[1]([z, hidden3_4+hidden1])
        out = []
        out.append(z)

        return out 



class UNet_Tempal(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(UNet_Tempal, self).__init__()      
        
        self.Encoder = nn.ModuleList([
            SEBlock(base_ch, num_res),
            SEBlock(base_ch, num_res),
            SEBlock(base_ch, num_res),
        ])

        self.Decoder = nn.ModuleList([
            SDBlock(base_ch, num_res),
            SDBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch* 2, base_ch* 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 2, kernel_size=1, relu=True, stride=1)
        ])

        self.up1=BasicConv(base_ch * 2, base_ch  * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2=BasicConv(base_ch * 2, base_ch  * 2, kernel_size=4, relu=True, stride=2, transpose=True)

    def forward(self, x1, x2):
        '''Feature Extract 0'''
        x_1, x_2 = self.feat_extract[0](torch.cat([x1, x2], dim=1)).chunk(2, dim=1)
        res1, hidden1 = self.Encoder[0]([x_1, x_2])

        '''Down Sample 1'''
        z, hidden1_1 = self.feat_extract[1](torch.cat([res1, hidden1], dim=1)).chunk(2, dim=1)
        res2, hidden2 = self.Encoder[1]([z, hidden1_1])

        '''Down Sample 2'''
        z, hidden2_1 = self.feat_extract[2](torch.cat([res2, hidden2], dim=1)).chunk(2, dim=1)
        res3, hidden3 = self.Encoder[2]([z, hidden2_1])

        '''Up Sample 2'''
        # z=self.up1(res3)
        z, hidden3_1 =self.up1(torch.cat([z, hidden3], dim=1)).chunk(2, dim=1)
        z, hidden3_1 = self.feat_extract[3](torch.cat([z + res2, hidden3_1], dim=1)).chunk(2, dim=1) # torch.cat([z, res2], dim=1))
        z, hidden3_2 = self.Decoder[0]([z, hidden3_1+hidden2])

        '''Up Sample 1'''
        z, hidden3_3=self.up2( torch.cat([z, hidden3_2], dim=1)).chunk(2, dim=1)
        z, hidden3_4 = self.feat_extract[4](torch.cat([z + res1, hidden3_3], dim=1)).chunk(2, dim=1)  # torch.cat([z, res1], dim=1)
        z, hidden3_5 = self.Decoder[1]([z, hidden3_4+hidden1])
        out = []
        out.append(z)

        return out  # z # , deepz



class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        # self.init_weights()


    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x



class SPCNet(nn.Module):
    def __init__(self,in_ch= 3,base_ch=32):
        super(SPCNet, self).__init__()
        in_ch= 3
        base_ch=16
        num_res_unet=1
        max_kernel_size = 13
        basis_num = 8  ## MLP num
        learnable_freq = True
        w_max = 16
        w_min = 2

        self.feat_ext = ResidualBlocksWithInputConv(in_ch, base_ch, 5)
        group_num=max_kernel_size//2
        self.group_num=group_num
        self.basis_num=basis_num
        self.unet_basis_select = UNet_Spatial(in_ch=base_ch, base_ch=base_ch, num_res=num_res_unet)
        self.unet_scale_select = UNet_Spatial(in_ch=base_ch, base_ch=base_ch, num_res=num_res_unet)

        self.scale_select = nn.Conv2d(base_ch, group_num+1, kernel_size=3, padding=1)
        self.basis_select = nn.Conv2d(base_ch, basis_num, kernel_size=3, padding=1)  # 1d

        self.inr_conv = SizeGroupINRConvCord(max_kernel_size=max_kernel_size, num_ch=in_ch,
                                              basis_num=basis_num, w_max=w_max, w_min=w_min,
                                              learnable_freq=learnable_freq)

        self.sum = nn.Conv2d((group_num+1)*basis_num*3, 3, kernel_size=1)

    def forward(self, x, x2):

        xs=[x]

        unet_outs = []
        for xi in xs:
            unet_outs.append(self.feat_ext(xi)) # [0]
            unet_outs.append(self.feat_ext(x2))

        unet_basis_select=self.unet_basis_select(unet_outs[0], unet_outs[1])  # 
        unet_scale_select=self.unet_scale_select(unet_outs[0], unet_outs[1])  # 

        scale_selects=[]
        for i, select in enumerate(unet_scale_select):
            select=self.scale_select(select)
            select = torch.softmax(select, dim=1)
            scale_selects.append(select)

        basis_selects=[]
        for i, beta in enumerate(unet_basis_select):
            b, _, h, w = beta.shape
            beta = self.basis_select(beta) # b, m / (group_num+1)*m, h, w
            basis_selects.append(beta)

        inr_convs=[]  ##### 
        for xi in xs:
            inr_convs.append(self.inr_conv(xi)) # b, group_num+1, 3*m, h, w

        ### inr_conv (INR); scale_selects (mu weight); basis_selects (omga weight)
        multiplies=[]
        for inr_conv, scale_selec, basis_select in zip(inr_convs, scale_selects, basis_selects):
            multiplies.append((inr_conv * scale_selec.unsqueeze(2) * basis_select.repeat(1,3,1,1).unsqueeze(1)).flatten(1, 2))  # 1d

        outs=[]
        for multiply, xi in zip(multiplies, xs):
            outs.append(self.sum(multiply)+xi)


        return outs, unet_outs[-1]


class SizeGroupINRConvCord(nn.Module):
    def __init__(self, max_kernel_size=17, num_ch=3, basis_num=5, w_max=7., w_min=1., w_list=None, learnable_freq=False):
        super(SizeGroupINRConvCord, self).__init__()
        if w_list is None:
            w_list=[w_min+(w_max-w_min)/(basis_num-1)*i for i in range(basis_num)]
            if learnable_freq:
                newwlist=[torch.nn.Parameter(torch.scalar_tensor(w_list[i], dtype=torch.float32)) for i in range(basis_num)]
                w_list=newwlist
        assert len(w_list)==basis_num
        self.w_list=w_list
        self.num_ch=num_ch
        self.kernelINR_list=nn.ModuleList(KernelINR_Cord(hidden_dim=360, w=w_list[i]) for i in range(basis_num))

        self.basis_num=basis_num
        self.max_kernel_size=max_kernel_size
        self.kernel_sizes=[(2*(i+1)+1, 2*(i+1)+1) for i in range(max_kernel_size//2)]

        self.padding=max_kernel_size//2
        self.group_num=len(self.kernel_sizes)

        masks=[] # [1x1, 3x3, ..., 15x15, ...]

        cords=[] # [1x1xc, 3x3xc, ..., 15x15xc, ...]

        empty=torch.zeros(self.basis_num, 1, 1, 1, device='cuda')
        # delta[0, :,max_kernel_size//2, max_kernel_size//2]=1
        delta = torch.ones(self.basis_num, 1, 1, 1, device='cuda')

        self.delta=delta
        self.empty=empty

        for siz in self.kernel_sizes:
            # print('siz',siz)
            mask = torch.ones(siz, device='cuda', dtype=torch.float32) * (3 ** 2) / (siz[0] * siz[1])
            # mask=torch.ones(siz, device='cuda', dtype=torch.float32)*(max_kernel_size**2)/(siz[0]*siz[1])
            masks.append(mask)

            cord=shape2coordinate(shape=siz, device='cuda')
            cords.append(cord)

        self.masks=masks
        self.cords=cords


    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []
        maps = []
        for k in range(self.group_num) :
            kernels_g = []
            for i in range(self.basis_num):
                kernel = self.kernelINR_list[i](self.cords[k])  # h w    spatial kernel dict
                kernel = kernel*self.masks[k]
                ### print kernel viz
                kernels_g.append(kernel.unsqueeze(0))
            kernels_g=torch.cat(kernels_g, dim=0)  # m h w
            maps_g = F.conv2d(x, kernels_g.repeat(self.num_ch, 1, 1).unsqueeze(1),
                              padding=self.kernel_sizes[k][0]//2,
                              groups=self.num_ch)  # b 3*m h w
            maps.append(maps_g.unsqueeze(1))
            kernels.append(kernels_g)
        maps=torch.cat(maps, dim=1)  # b gn 3*m h w

        null_map=torch.zeros(b, 1, self.num_ch*self.basis_num, h, w, device='cuda')

        maps=torch.cat([null_map, maps], dim=1)  # b gn+1 3*m h w

        return maps


class ResidualBlocksWithInputConv(torch.nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)



class INRAlign(nn.Module):
    def __init__(self,in_ch= 3,base_ch=64):
        super(INRAlign, self).__init__()
        in_ch= 64
        base_ch=32
        num_res_unet=1
        max_kernel_size = 13
        basis_num = 8 # 10
        learnable_freq = True
        w_max = 16  # 12
        w_min = 2
        self.fusion1 =  ResidualBlocksWithInputConv(in_ch, base_ch, 3)
        self.fusion2 =  ResidualBlocksWithInputConv(16, base_ch, 3)
        group_num=max_kernel_size//2
        self.group_num=group_num
        self.basis_num=basis_num
        # which type?
        self.unet_basis_select = UNet_Tempal(in_ch=base_ch, base_ch=base_ch, num_res=num_res_unet)
        self.unet_scale_select = UNet_Tempal(in_ch=base_ch, base_ch=base_ch, num_res=num_res_unet) 

        self.scale_select = nn.Conv2d(base_ch, group_num+1, kernel_size=3, padding=1)
        self.basis_select = nn.Conv2d(base_ch, basis_num, kernel_size=3, padding=1)  # 1d

        self.inr_conv = SizeGroupINRConvCord(max_kernel_size=max_kernel_size, num_ch=in_ch,
                                              basis_num=basis_num, w_max=w_max, w_min=w_min,
                                              learnable_freq=learnable_freq)
        
        self.sum = nn.Conv2d((group_num+1)*basis_num*64, in_ch, kernel_size=1)

    def forward(self, x_feat, x_pre, flow):
        x_w = flow_warp(x_pre, flow)
        xs=[x_w]
        #### iter  1 
        unet_outs = []
        x_wfeat = self.fusion1(x_w)
        unet_outs.append(x_wfeat)
        x_feat = self.fusion2(x_feat)

        unet_basis_select=self.unet_basis_select(x_feat, unet_outs[0]) # 
        unet_scale_select=self.unet_scale_select(x_feat, unet_outs[0]) # 

        scale_selects=[]
        for i, select in enumerate(unet_scale_select):
            select=self.scale_select(select)
            select = torch.softmax(select, dim=1)
            scale_selects.append(select)

        basis_selects=[]
        for i, beta in enumerate(unet_basis_select):
            b, _, h, w = beta.shape
            beta = self.basis_select(beta) # b, m / (group_num+1)*m, h, w
            basis_selects.append(beta)

        inr_convs=[]
        for xi in xs:
            inr_convs.append(self.inr_conv(xi)) # b, group_num+1, 3*m, h, w

        multiplies=[]
        for inr_conv, scale_selec, basis_select in zip(inr_convs, scale_selects, basis_selects):
            multiplies.append((inr_conv * scale_selec.unsqueeze(2) * basis_select.repeat(1,64,1,1).unsqueeze(1)).flatten(1, 2))  # 1d

        outs=[]
        for multiply, xi in zip(multiplies, xs):
            outs.append(self.sum(multiply)+xi)

        return outs


class KCANet(torch.nn.Module):
    # Restoration Network
    def __init__(self, config):
        super(KCANet, self).__init__()
        in_channels = config.in_channels
        dim = config.dim
        self.dim = config.dim
        num_seq = config.num_seq
        bias = config.bias
        scale = config.scale
        self.scale = config.scale
        num_blocks = 11

        self.SPC = SPCNet()
        self.spynet = SpyNet('./preprocessing/spynet_sintel_final-3d2a1287.pth')
        self.INRAlign_forward = INRAlign() 
        self.INRAlign_backward = INRAlign() 
        
        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(dim + 16, dim, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(dim + 16, dim, num_blocks)

        # upsample
        self.fusion = ResidualBlocksWithInputConv(dim * 2, dim, 10) 
        self.upsample1 = PixelShufflePack(dim, dim, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(dim, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def pad_spatial(self, x):
        """Apply padding spatially.
        Args: x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:  Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4       
        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate') #   reflect
        x_out = x.view(n, t, c, h + pad_h, w + pad_w)

        return x_out
    
    def get_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward
    
    def forward(self, lrs):
        # lrs: [B, T, 3,  H, W]
        B, T, C, H_in, W_in = lrs.shape
        lrs = self.pad_spatial(lrs)
        B, T, C, H, W = lrs.shape
        result_dict = {}
        # backward-time propagation
        outputs_l = []
        # outputs_l = [0 for _ in range(T)]
        feat_prop = lrs.new_zeros(B, self.dim, H, W)
        lr_hat_l = []
        lr_feat_l = []
        # compute optical flow
        flows_forward, flows_backward = self.get_flow(lrs)
        for i in range(T - 1, -1, -1):
            # spatial correlation 
            lr_hat_list, lr_feat = self.SPC(lrs[:, i, :, :, :], lrs[:, i, :, :, :]) 
            lr_hat = lr_hat_list[-1]
            if i < T - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = self.INRAlign_backward(lr_feat, feat_prop, flow.permute(0, 2, 3, 1))[-1]
            feat_prop = torch.cat([lr_feat, feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs_l.append(feat_prop)
            lr_feat_l.append(lr_feat)
            lr_hat_l.append(lr_hat)
        outputs_l = outputs_l[::-1]
        lr_hat_l = lr_hat_l[::-1]

        # # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, T):
            # lr_curr = lrs[:, i, :, :, :]
            lr_curr = lr_hat_l[i]
            lr_feat = lr_feat_l[i]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = self.INRAlign_forward(lr_feat, feat_prop, flow.permute(0, 2, 3, 1))[-1]

            feat_prop = torch.cat([lr_feat, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs_l[i] = out

        outputs = torch.stack(outputs_l, dim=1)[..., :4 * H_in, :4 * W_in]
        lr_hats = torch.stack(lr_hat_l, dim=1)[..., :H_in, :W_in]
        result_dict['outputs'] = outputs 
        result_dict['lr_corrs'] = lr_hats

        return result_dict
