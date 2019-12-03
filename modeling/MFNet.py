# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.nn import Parameter
import math
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, norm_layer):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MFModule(nn.Module):
    def __init__(self, inplanes, input_size, factors, norm_layer, mf_epoch):
        super(MFModule, self).__init__()
        self.Y = Parameter(torch.Tensor(1, inplanes, factors), requires_grad=True)
        stdy = 1./((inplanes*factors)**(1/2))
        self.Y.data.uniform_(-stdy, stdy)    # Initialize
        
        self.Z = Parameter(torch.Tensor(1, factors, input_size**2), requires_grad=True)
        stdz = 1./(((input_size**2)*factors)**(1/2))
        self.Z.data.uniform_(-stdz, stdz)
        self.mf_epoch = mf_epoch
        
    def forward(self, x, epoch):
        x_out = x
        b, c, h, w = x_out.size()
        x_low_rank = self.Y.matmul(self.Z)
        x_low_rank = x_low_rank.repeat(b, 1, 1)
        x_low_rank = x_low_rank.view(b, c, h, w)
        
        #if epoch >= self.mf_epoch:
        #x_out = x_low_rank
        
        return x_low_rank
    
class MFNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, input_size=33, 
                 factors=64, mf_epoch=10):
        super(MFNet, self).__init__()
        
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
            
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
            
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.fc0 = ConvBNReLU(inplanes, 512, 3, 1, 1, 1, BatchNorm)
        inplanes = 512
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 256, 1, bias=False),
            BatchNorm(256)
            )
        
        self.mf = MFModule(inplanes, input_size, factors, BatchNorm, mf_epoch)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1, BatchNorm),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, num_classes, 1)
        self.mf_epoch = mf_epoch
        
        if freeze_bn:
            self.freeze_bn()
        
    def forward(self, input, epoch=0):
        x, low_level_feat = self.backbone(input)
        x = self.fc0(x)
        x = self.conv1(x)
        x_low_rank = self.mf(x, epoch)
        x_out = x_low_rank
        x_out = x_out + x
        x_out = self.conv2(x_out)
        #x_out = self.fc1(x_out)
        #x_out = self.fc2(x_out)
        
        x_out = self.decoder(x_out, low_level_feat)
        x_out = F.interpolate(x_out, size=input.size()[2:], mode='bilinear',\
                              align_corners=True)
            
        return x, x_low_rank, x_out
    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
        yield self.mf.Y
        yield self.mf.Z

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p