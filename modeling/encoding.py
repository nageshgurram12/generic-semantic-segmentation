# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from modeling.functions import *

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=None, ncodes=32, se_loss=True):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

        self.last_conv = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        outputs[0] = self.last_conv(outputs[0])
        return tuple(outputs)


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Encoding(nn.Module):
    r"""
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\times H\times W`
    as a set of C-dimentional input features :math:`X=\{x_1, ...x_N\}`, where N is total number
    of features given by :math:`H\times W`, which learns an inherent codebook
    :math:`D=\{d_1,...d_K\}` and a set of smoothing factor of visual centers
    :math:`S=\{s_1,...s_K\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\{e_1,...e_K\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

def build_encoding(backbone, num_classes, BatchNorm):
  if backbone == 'drn':
    in_channels = 512
  elif backbone == 'mobilenet':
    in_channels = 320
  else:
    in_channels = 2048

  return EncModule(in_channels, num_classes, BatchNorm)