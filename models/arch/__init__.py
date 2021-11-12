# Add your custom network here
from .default import DRNet
from .unet import RefinedNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1)

def specularitynet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True)

def refined(in_channels, out_channels, enhance='de', ppm=True, bilinear=True):
    return RefinedNet(in_channels, out_channels, enhance=enhance, ppm=ppm, bilinear=bilinear)
