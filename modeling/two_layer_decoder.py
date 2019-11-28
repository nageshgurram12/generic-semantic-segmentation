import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class NonBottleNeck(nn.Module):
  
  def __init__(self, inplanes, planes, BatchNorm=None, dilation=1):
    super(NonBottleNeck, self).__init__()
    reduction = 8
    
    self.conv_pre = None
    if inplanes != planes:
      self.conv_pre = nn.Conv2d(inplanes, planes, 1, bias=False)
    
    self.conv1d_1 = nn.Conv2d(planes, planes, kernel_size=(1,3), padding=(0,1), bias=False)
    
    self.conv1d_2 = nn.Conv2d(planes, planes, kernel_size=(3,1), padding=(1,0), bias=False)
    
    red_planes = int(planes/reduction) # Reduce the planes 
    self.conv11c = nn.Conv2d(planes, red_planes, 1, bias=False)
    
    self.conv1 = nn.Conv2d(planes, planes, 1, bias=False)
    
    self.conv3 = nn.Conv2d(red_planes, red_planes, 3, dilation = dilation, padding = dilation, bias=False)
    
    self.conv5 = nn.Conv2d(red_planes, red_planes, 5, dilation = dilation, padding = 2*dilation, bias=False)
    
    self.conv11e = nn.Conv2d(red_planes, planes, 1, bias=False)
    
    self.bn1 = BatchNorm(red_planes)
    self.bn2 = BatchNorm(planes)
    
    self.relu = nn.ReLU(inplace=True)
    
    self.conv11 = nn.Sequential(nn.Conv2d(planes, planes, 1, bias=False),
                                BatchNorm(planes),
                                nn.ReLU(inplace=True))
    
    self.conv33 = nn.Sequential(self.conv11c,
                                BatchNorm(red_planes),
                                nn.ReLU(inplace=True),
                                self.conv3,
                                BatchNorm(red_planes),
                                nn.ReLU(inplace=True),
                                self.conv11e,
                                BatchNorm(planes),
                                nn.ReLU(inplace=True))
    
    self.conv55 = nn.Sequential(self.conv11c,
                                BatchNorm(red_planes),
                                nn.ReLU(inplace=True),
                                self.conv5,
                                BatchNorm(red_planes),
                                nn.ReLU(inplace=True),
                                self.conv11e,
                                BatchNorm(planes),
                                nn.ReLU(inplace=True))
    
  def forward(self, x):
    
    if self.conv_pre is not None:
      out = self.conv_pre(x)
      
    # Apply 1D convolutions
    out = self.conv1d_1(x)
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv1d_2(out)
    out = self.bn2(out)
    out = self.relu(out)
    
    # Apply 3x3 conv
    out33 = self.conv33(out)
    
    #Apply 5x5 conv
    out55 = self.conv55(out)
    
    # Apply 1x1 conv
    out11 = self.conv11(out)
    
    #merge all paths
    out = (out11 + out33 + out55)
    out += x
    
    return out

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = (256, 512)
        elif backbone == 'xception':
            low_level_inplanes = (128, 256)
        elif backbone == 'mobilenet':
            low_level_inplanes = (24, 48)
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=True)
        self.conv3x_0 = nn.Conv2d(low_level_inplanes[1], 96, 1, bias=False)
        self.bn3x_0 = BatchNorm(96)
        
        #conv after fuse with res3x low level features
        self.type1_nb_layer = nn.Sequential(nn.Conv2d(352, 256, kernel_size=1, bias=False),
                                            BatchNorm(256),
                                            NonBottleNeck(256, 256, BatchNorm, dilation=1),
                                            NonBottleNeck(256, 256, BatchNorm, dilation=1),
                                            nn.Dropout(0.5))
        
        self.conv2x_0 = nn.Conv2d(low_level_inplanes[0], 48, 1, bias=False)
        self.bn2x_0 = BatchNorm(48)
        
        #Conv after fuse with res2x low level features
        self.type2_nb_layer = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, bias=False),
                                            BatchNorm(256),
                                            NonBottleNeck(256, 256, BatchNorm, dilation=2),
                                            NonBottleNeck(256, 256, BatchNorm, dilation=2),
                                            nn.Dropout(0.1))
        
        self.conv_last = nn.Conv2d(256, num_classes, kernel_size = 1)
        '''
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        '''
        self._init_weight()


    def forward(self, x, low_level_feat_2x, low_level_feat_3x):
      '''
      low_level_feat_3x has size (512, 65, 65)
      low_level_feat_2x has size (256, 129, 129)
      '''
      low_level_feat_3x = self.conv3x_0(low_level_feat_3x)
      low_level_feat_3x = self.bn3x_0(low_level_feat_3x)
      low_level_feat_3x = self.relu(low_level_feat_3x)
      x = F.interpolate(x, size=low_level_feat_3x.size()[2:], mode='bilinear', align_corners=True)
      x = torch.cat((x, low_level_feat_3x), dim=1)
      x = self.type1_nb_layer(x)
      
      low_level_feat_2x = self.conv2x_0(low_level_feat_2x)
      low_level_feat_2x = self.bn2x_0(low_level_feat_2x)
      low_level_feat_2x = self.relu(low_level_feat_2x)

      x = F.interpolate(x, size=low_level_feat_2x.size()[2:], mode='bilinear', align_corners=True)
      x = torch.cat((x, low_level_feat_2x), dim=1)
      x = self.type2_nb_layer(x)
      
      x = self.conv_last(x)

      return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)