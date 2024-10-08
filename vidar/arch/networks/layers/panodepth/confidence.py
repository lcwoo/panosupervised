import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from torch.autograd import Variable

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class AggWeightNetVolume(nn.Module):
    def __init__(self, in_channels=64):
        super(AggWeightNetVolume, self).__init__()
        self.w_net = nn.Sequential(
            Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, volume_variance):
        """
        :param x: (B, C, D, H, W), input features for the given depth hypotheses
        :param volume_variance: (B, C, H, W), computed variance between views (without depth dimension)
        :return: confidence map (B, 1, D, H, W)
        """
        # volume_variance를 깊이 차원(D)에 맞게 확장
        volume_variance_expanded = volume_variance.unsqueeze(2).repeat(1, 1, x.size(2), 1, 1)  # (B, C, D, H, W)

        # 가중치 계산
        w = self.w_net(x)
        
        # Variance 기반 confidence 계산 (Variance가 작을수록 높은 confidence)
        confidence = torch.sigmoid(w) * torch.exp(-volume_variance_expanded)  # (B, 1, D, H, W)
        
        return confidence
# class AggWeightNetVolume(nn.Module):
#     def __init__(self, in_channels=64):  # 채널 수에 맞춰 64로 설정
#         super(AggWeightNetVolume, self).__init__()
#         self.w_net = nn.Sequential(
#             nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),  # Conv2d로 변경
#             nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)  # Conv2d로 변경
#         )

#     def forward(self, x):
#         """
#         :param x: (b, c, h, w)  # 4차원 입력
#         :return: (b, 1, h, w)  # 4차원 출력
#         """
#         w = self.w_net(x)
#         return w
