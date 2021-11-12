import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import itertools
from collections import OrderedDict

import util.util as util
from util.util import tensor2im
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join

class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class specularitynetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mask = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, mask = data['input'], data['target_t'], data['mask']
        elif mode == 'eval':
            input, target_t, mask, data_name = data['input'], data['target_t'], data['mask'], data['fn']
        elif mode == 'test':
            input, mask, data_name = data['input'], data['mask'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])
            if mask is not None:
                mask = mask.to(device=self.gpu_ids[0])
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.mask = mask
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         
            
    def eval(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            res_overall = {'PSNR':0, 'SSIM': 0, 'LMSE': 0, 'NCC': 0}

            for _input, target, result, mask, detect, fn in zip(self.input, self.target_t,self.output_i,self.mask,self.detect,self.data_name):
                _input = tensor2im(_input)
                target = tensor2im(target)
                result = tensor2im(result)
                mask = tensor2im(mask)
                detect = tensor2im(detect)
                #print([_input.shape,target.shape,result.shape,fn])
                if self.aligned:
                    res = index.quality_assess(result.astype(np.float32), target.astype(np.float32))
                    for key in res_overall:
                        res_overall[key] += res[key]

                if savedir is not None:
                    os.makedirs(join(savedir),exist_ok=True)
                    if self.opt.suffix is not None:
                        Image.fromarray(result).save(join(savedir,'{}_result_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                    else:
                        Image.fromarray(result).save(join(savedir, '{}_result_{}.png'.format(fn, self.opt.name)))
                    # if not os.path.exists(join(savedir, '{}_target.png'.format(fn))):
                    Image.fromarray(target).save(join(savedir, '{}_target.png'.format(fn)))
                    # if not os.path.exists(join(savedir, '{}_input.png'.format(fn))):
                    Image.fromarray(_input).save(join(savedir, '{}_input.png'.format(fn)))
                    if not mask.size == 0:
                        Image.fromarray(mask).save(join(savedir, '{}_mask_{}.png'.format(fn, self.opt.name)))
                    if not detect.size == 0:
                        spec = np.uint8(np.clip(np.int32(_input)-np.int32(result),0,255))
                        if self.opt.suffix is not None:
                            Image.fromarray(detect).save(join(savedir,'{}_detect_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                            Image.fromarray(spec).convert('L').save(join(savedir,'{}_specular_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                        else:
                            Image.fromarray(detect).save(join(savedir, '{}_detect_{}.png'.format(fn, self.opt.name)))
                            Image.fromarray(spec).convert('L').save(join(savedir, '{}_specular_{}.png'.format(fn, self.opt.name)))
                        

            for key in res_overall:
                res_overall[key] /= len(self.data_name)
            
            return res_overall

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')
 
        with torch.no_grad():
            self.forward()
            if self.data_name is not None and savedir is not None:
                os.makedirs(join(savedir),exist_ok=True)
                for _input, result, mask, detect, fn in zip(self.input,self.output_i,self.mask,self.detect,self.data_name):
                    _input = tensor2im(_input)
                    result = tensor2im(result)
                    mask = tensor2im(mask)
                    detect = tensor2im(detect)            
                    if self.opt.suffix is not None:
                        Image.fromarray(result).save(join(savedir,'{}_result_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                    else:
                        Image.fromarray(result).save(join(savedir, '{}_result_{}.png'.format(fn, self.opt.name)))
                    # if not os.path.exists(join(savedir, '{}_input.png'.format(fn))):
                    Image.fromarray(_input).save(join(savedir,'{}_input.png'.format(fn)))
                    if not mask.size == 0:
                        Image.fromarray(mask).save(join(savedir, '{}_mask.png'.format(fn)))
                    if not detect.size == 0:
                        spec = np.uint8(np.clip(np.int32(_input)-np.int32(result),0,255))
                        if self.opt.suffix is not None:
                            Image.fromarray(detect).save(join(savedir,'{}_detect_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                            Image.fromarray(spec).convert('L').save(join(savedir,'{}_specular_{}_{}.png'.format(fn, self.opt.name, self.opt.suffix)))
                        else:
                            Image.fromarray(detect).save(join(savedir, '{}_detect_{}.png'.format(fn, self.opt.name)))
                            Image.fromarray(spec).convert('L').save(join(savedir, '{}_specular_{}.png'.format(fn, self.opt.name)))

class specularitynetModel(specularitynetBase):
    def name(self):
        return 'specularitynet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None
        
        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472
        #self.net_i = arch.__dict__[self.opt.inet](in_channels, 3, self.opt.enhance, self.opt.ppm).to(self.device)
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss
            self.loss_dic['t_ssim'] = losses.MSSSIM()
            self.loss_dic['t_det'] = losses.BinaryFocalLoss()

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), 
                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self, opt.resume_epoch)
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_feat = None
        self.loss_SSIM = None
        self.loss_coarse = None
        self.loss_detect = None

        if self.opt.lambda_gan > 0:
            if self.opt.gan_type == 'rasgan' and self.opt.lambda_feat >0:
                self.loss_G_GAN, self.loss_feat = self.loss_dic['gan'].get_g_feat_loss(
                    self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
                self.loss_G += self.loss_G_GAN*self.opt.lambda_gan + self.loss_Feat*self.opt.lambda_feat
            else:
                self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                    self.netD, self.input, self.output_i, self.target_t)
                self.loss_G += self.loss_G_GAN*self.opt.lambda_gan

        if self.aligned:
            # refined pixel loss
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_icnn_pixel
            # coarse pixel loss
            if not self.coarse_list == []:
                self.loss_coarse = 0
                for coarse in self.coarse_list:
                    self.loss_coarse += self.loss_dic['t_pixel'].get_loss(coarse,self.target_t)
                    if self.opt.lambda_vgg > 0:
                        self.loss_coarse += self.loss_dic['t_vgg'].get_loss(coarse,self.target_t)*self.opt.lambda_vgg
                    self.loss_coarse /= len(self.coarse_list)
                self.loss_G += self.loss_coarse * self.opt.lambda_coarse
            # refined vgg loss
            if self.opt.lambda_vgg > 0:
                self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                    self.output_i, self.target_t)
                self.loss_G += self.loss_icnn_vgg*self.opt.lambda_vgg
            # refined ssim loss
            if self.opt.lambda_ssim > 0:
                self.loss_SSIM = self.loss_dic['t_ssim'](self.output_i, self.target_t)
                self.loss_G += self.loss_SSIM*self.opt.lambda_ssim
            # detect loss
            if (not self.mask.numel() == 0) and (not self.detect_list == []):
                self.loss_detect = 0
                for detect in self.detect_list:
                    self.loss_detect += self.loss_dic['t_det'](detect,self.mask)
                self.loss_detect /= len(self.detect_list)
                self.loss_G += self.loss_detect * self.opt.lambda_detect
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input

        if self.vgg is not None:
            hypercolumn = self.vgg(self.input)
            _, C, H, W = self.input.shape
            hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
            input_i = [input_i]
            input_i.extend(hypercolumn)
            input_i = torch.cat(input_i, dim=1)

        #output = self.net_i(input_i,self.opt.iters)
        output = self.net_i(input_i)
        self.output_i = output['refined']
        self.coarse_list = output['coarse'] if 'coarse' in output else []
        self.detect_list = output['detect'] if 'detect' in output else []
        if not self.detect_list == []:
            self.detect = (self.detect_list[-1].detach()>0.5).type(torch.float32)
        else:
            self.detect = torch.zeros([self.output_i.shape[0],0]).type(torch.float32).to(self.output_i.device)
        return self.output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        if self.loss_feat is not None:
            ret_errors['Feat'] = self.loss_feat.item()

        if self.loss_SSIM is not None:
            ret_errors['SSIM'] = self.loss_SSIM.item()

        if self.loss_coarse is not None:
            ret_errors['Coarse'] = self.loss_coarse.item()

        if self.loss_detect is not None:
            ret_errors['Detect'] = self.loss_detect.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = torch.load(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model.net_i.load_state_dict(state_dict['icnn'])
            if model.isTrain:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
        else:
            state_dict = torch.load(icnn_path)
            model.net_i.load_state_dict(state_dict['icnn'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict


class NetworkWrapper(specularitynetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[31]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net.parameters(), 
                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
