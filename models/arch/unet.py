import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def pad(x, ref=None,h=None,w=None):
    assert not (ref is None and h is None and w is None)
    _,_,h1,w1 = x.shape
    if not ref is None:
        _,_,h2,w2 = ref.shape
    else:
        h2,w2 = h,w
    if not h1==h2 or not w1==w2:
        x = F.pad(x,(0,w2-w1,0,h2-h1),mode='replicate')
    return x

# 合并卷积、归一化与激活函数
class ConvNormAct(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None, num_groups=8, act='elu', negative_slope=0.1, inplace=True,reflect=True):
        super(ConvNormAct, self).__init__()
        self.layer = nn.Sequential()
        if reflect:
            self.layer.add_module('pad',nn.ReflectionPad2d(padding))
            self.layer.add_module('conv',nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias))
        else:
            self.layer.add_module('conv',nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
        if norm == 'bn':
            self.layer.add_module('norm',nn.BatchNorm2d(num_features=out_channels))
        elif norm == 'gn':
            self.layer.add_module('norm',nn.GroupNorm(num_groups=num_groups,num_channels=out_channels))
        else:
            pass
        if act == 'relu':
            self.layer.add_module('act',nn.ReLU(inplace=inplace))
        if act == 'relu6':
            self.layer.add_module('act',nn.ReLU6(inplace=inplace))
        elif act == 'elu':
            self.layer.add_module('act',nn.ELU(alpha=1.0))
        elif act == 'lrelu':
            self.layer.add_module('act',nn.LeakyReLU(negative_slope=negative_slope,inplace=inplace))
        elif act == 'sigmoid':
            self.layer.add_module('act',nn.Sigmoid())
        else:
            pass
    
    def forward(self,x):
        y = self.layer(x)
        return y

# 门控卷积
class GatedConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None, num_groups=8, act='elu', negative_slope=0.1, inplace=True,\
            full=True,reflect=True):
        super(GatedConv, self).__init__()
        self.conv = ConvNormAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias,\
            norm=norm,num_groups=num_groups,act=act,negative_slope=negative_slope,reflect=reflect)
        if full:
            self.gate = ConvNormAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, act='sigmoid',reflect=reflect)
        else:
            self.gate = ConvNormAct(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, act='sigmoid',norm=None,num_groups=1,reflect=reflect)
    def forward(self,x):
        return self.conv(x)*self.gate(x)

# 感受野扩张，默认dilations=[2,4]，使用连续的两层卷积进行感受野扩张
class DilationBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels,dilations,gatedconv=False):
        super(DilationBlock,self).__init__()
        assert isinstance(dilations,list) or isinstance(dilations,tuple)
        i_channels = [in_channels]+[mid_channels]*(len(dilations)-1)
        o_channels = [mid_channels]*len(dilations)
        Conv = GatedConv if gatedconv else ConvNormAct
        self.conv = nn.Sequential(
            *[Conv(i_channels[i],o_channels[i],kernel_size=3,padding=dilations[i],dilation=dilations[i]) \
                    for i in range(len(dilations))]
        )
        self.out = Conv(in_channels+mid_channels,out_channels,kernel_size=1)
    def forward(self,x):
        conv = self.conv(x)
        out = self.out(torch.cat([x,conv],dim=1))
        return out

# 下采样块，kernels列表的长度即为卷积层层数
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,kernels=None,gatedconv=False):
        super(DownBlock,self).__init__()
        assert isinstance(kernels,list) or isinstance(kernels,tuple) or isinstance(kernels,int)
        Conv = GatedConv if gatedconv else ConvNormAct
        if isinstance(kernels,int):
            assert mid_channels is None
            self.conv = Conv(in_channels,out_channels,kernel_size=kernels,stride=2,padding=kernels//2)
        else:
            if mid_channels is None:
                mid_channels = out_channels
            i_channels = [in_channels]+[mid_channels]*(len(kernels)-1)
            o_channels = [mid_channels]*(len(kernels)-1)+[out_channels]
            conv = [Conv(i_channels[0],o_channels[0],kernel_size=kernels[0],stride=2,padding=kernels[0]//2)]
            for i in range(1,len(kernels)):
                conv.append(Conv(i_channels[i],o_channels[i],kernel_size=kernels[i],padding=kernels[i]//2))
            self.conv = nn.Sequential(*conv)
    def forward(self,x):
        return self.conv(x)

# specularitynet的SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#我们的DCALayer
class DELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(DELayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(channel*3, channel*3 // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel*3 // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        with torch.no_grad():
            _mean = x.mean(dim=[2,3])
            _std = x.std(dim=[2,3])
            _max = x.max(dim=2)[0].max(dim=2)[0]
        feat = torch.cat([_mean,_std,_max],dim=1)
        b, c, _, _ = x.shape
        y = self.fc(feat).view(b, c, 1, 1)
        return x * y

#残差块，blocks为残差单元数目
class ResBlock(nn.Module):
    def __init__(self,channels,blocks=3,resscale=0.1,kernel_size=0,dilations=[2],gatedconv=False,enhance='de',ppm=True):
        super(ResBlock,self).__init__()
        self.convs = nn.ModuleList(
                [ self._build_layer(channels,kernel_size,dilations,gatedconv,enhance) for i in range(blocks)]
            )
        self.resscale = resscale
        self.ppm = PyramidPooling(channels,channels,ct_channels=channels) if ppm else None
    def _build_layer(self,channels,kernel_size=0,dilations=[2],gatedconv=False,enhance='de'):
        conv = GatedConv if gatedconv else ConvNormAct
        layer = nn.Sequential(
            conv(channels,channels,kernel_size=kernel_size,padding=(kernel_size//2))
            #DilationBlock(channels,channels,channels,dilations=dilations,gatedconv=gatedconv)
        )
        if enhance == 'se':
            layer.add_module('se',SELayer(channels))
        elif enhance == 'de':
            layer.add_module('de',DELayer(channels))
        return layer
    
    def forward(self,x):
        for conv in self.convs:
            x = conv(x)+x*self.resscale
        if not self.ppm is None:
            x = self.ppm(x)
        return x

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))

# 上采样块，bilinear为True时，私用双线性插值进行上采样而不是转置卷积
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,add_channels=None,kernels=None,gatedconv=True,bilinear=False,shape=None):
        super(UpBlock,self).__init__()
        assert isinstance(kernels,list) or isinstance(kernels,tuple)
        Conv = GatedConv if gatedconv else ConvNormAct
        if mid_channels is None:
            mid_channels = out_channels
        if isinstance(mid_channels,int):
            i_channels = [in_channels]+[mid_channels]*(len(kernels)-1)
            o_channels = [mid_channels]*(len(kernels)-1)+[out_channels]
        else:
            assert isinstance(mid_channels,list) or isinstance(mid_channels,tuple)
            assert len(mid_channels) == len(kernels)-1
            i_channels = [in_channels]+list(mid_channels)
            o_channels = list(mid_channels)+[out_channels]
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if not add_channels is None:
                i_channels[0] = i_channels[0]+add_channels
            conv = []
            for i in range(len(kernels)):
                conv.append(Conv(i_channels[i],o_channels[i],kernel_size=kernels[i],padding=kernels[i]//2))
            self.conv = nn.Sequential(*conv)
        else:
            self.up = nn.ConvTranspose2d(i_channels[0],o_channels[0],kernel_size=kernels[0],stride=2,padding=kernels[0]//2,output_padding=1)
            if not add_channels is None:
                i_channels[1] = i_channels[1]+add_channels
            conv = []
            for i in range(1,len(kernels)):
                conv.append(Conv(i_channels[i],o_channels[i],kernel_size=kernels[i],padding=kernels[i]//2))
            self.conv = nn.Sequential(*conv)
    def forward(self,x,feat=None,shape=None):
        assert not feat is None or not shape is None
        up = self.up(x)
        up = pad(up,ref=feat,h=None if shape is None else shape[0] ,w=None if shape is None else shape[1])
        if not feat is None:
            return self.conv(torch.cat([up,feat],dim=1))
        else:
            return self.conv(up)

class RefinedNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,enhance='de',ppm=True,bilinear=True):
        super(RefinedNet,self).__init__()
        d_channels = [64,128,256]
        u_channels = [256,128,64]
        Conv = ConvNormAct
        # first encoder
        self.preconv = nn.Sequential(
            Conv(in_channels,64,kernel_size=7,padding=3),
            Conv(64,64,kernel_size=7,padding=3),
            Conv(64,d_channels[0],kernel_size=7,padding=3)
        )
        # second encoder
        self.down1 = nn.Sequential(
            DownBlock(d_channels[0],d_channels[1],kernels=[3,3],gatedconv=False),
            DilationBlock(d_channels[1],d_channels[1],d_channels[1],dilations=[2,4],gatedconv=False)
        )
        # third encoder
        self.down2 = nn.Sequential(
            DownBlock(d_channels[1],d_channels[2],kernels=[3,3],gatedconv=False),
            DilationBlock(d_channels[2],d_channels[2],d_channels[2],dilations=[2,4],gatedconv=False)
        )
        # four-th encoder
        self.resblock = ResBlock(d_channels[2],blocks=3,resscale=0.1,kernel_size=3,gatedconv=True,enhance=enhance,ppm=ppm,dilations=[2,4])
        # first decoder
        self.up1 = UpBlock(u_channels[0],u_channels[1],u_channels[1],add_channels=d_channels[1],kernels=[5,5],gatedconv=False,bilinear=bilinear)
        # second decoder
        self.up2 = UpBlock(u_channels[1],u_channels[2],u_channels[2],add_channels=d_channels[0],kernels=[5,5],gatedconv=False,bilinear=bilinear)
        # coarse detection block
        self.coarse_detect = nn.Sequential(
            ConvNormAct(u_channels[2]+3,64,kernel_size=3,padding=1),
            ConvNormAct(64,64,kernel_size=3,padding=1),
            ConvNormAct(64,1,kernel_size=3,padding=1,norm=None,act='sigmoid')
        )
        # coarse removal block
        self.coarse = nn.Sequential(
            ConvNormAct(u_channels[2]+4,64,kernel_size=3,padding=1),
            ConvNormAct(64,64,kernel_size=3,padding=1),
            ConvNormAct(64,out_channels,kernel_size=3,padding=1,norm=None)
        )
        # refined detection block
        self.refined_detect = nn.Sequential(
            GatedConv(u_channels[2]+7,64,kernel_size=7,padding=3),
            ResBlock(64,blocks=3,resscale=1.0,kernel_size=3,gatedconv=True,enhance=enhance,ppm=ppm,dilations=[2,4]),
            Conv(64,1,kernel_size=3,padding=1,norm=None,act='sigmoid')
        )
        # refined removal block
        self.refined = nn.Sequential(
            GatedConv(u_channels[2]+7,64,kernel_size=7,padding=3),
            ResBlock(64,blocks=3,resscale=1.0,kernel_size=3,gatedconv=True,enhance=enhance,ppm=ppm,dilations=[2,4]),
            Conv(64,3,kernel_size=3,padding=1,norm=None,act='relu6')
        )
    def forward(self,x,iters=1):
        preconv = self.preconv(x)
        down1 = self.down1(preconv)
        down2 = self.down2(down1)
        up1 = self.up1(down2,down1)
        up2 = self.up2(up1,preconv)
        detect = self.coarse_detect(torch.cat([x,up2],dim=1))
        y = self.coarse(torch.cat([x,up2,detect],dim=1))
        coarse_list = []
        detect_list = [detect.squeeze(1)]
        for i in range(iters):
            coarse_list.append(y)
            detect = self.refined_detect(torch.cat([x,y,up2,detect],dim=1))
            y = self.refined(torch.cat([x,y,up2,detect],dim=1))
            detect_list.append(detect.squeeze(1))
        return {'refined':y,'coarse':coarse_list,'detect':detect_list}

# Discriminator
class DiscFeat(nn.Module):
    def __init__(self):
        super(DiscFeat,self).__init__()
        self.feat = nn.Sequential(
            ConvNormAct(3,64,kernel_size=3,padding=1,norm='gn'),
            ConvNormAct(64,64,kernel_size=3,padding=1,stride=2,norm='gn'),
            ConvNormAct(64,128,kernel_size=3,padding=1,norm='gn'),
            ConvNormAct(128,128,kernel_size=3,padding=1,stride=2,norm='gn'),
            ConvNormAct(128,256,kernel_size=3,padding=1,norm='gn'),
            ConvNormAct(256,256,kernel_size=3,padding=1,stride=2,norm='gn'),
            ConvNormAct(256,512,kernel_size=3,padding=1,norm='gn'),
            ConvNormAct(512,512,kernel_size=3,padding=1,stride=2,norm='gn'),
            nn.AdaptiveAvgPool2d(16)
        )
        self.disc = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=16,padding=0),
            nn.GroupNorm(32,1024),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024,1,kernel_size=1,padding=0)
        )
    def forward(self,x,require_feat=False): # if require_feat, D also output the middle feature maps to calculate the loss
        feat = self.feat(x)
        y = self.disc(feat)
        y = y.view(y.shape[0],1)
        if require_feat:
            return y,feat
        else:
            return y