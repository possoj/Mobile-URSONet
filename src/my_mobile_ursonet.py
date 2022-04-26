"""
Copyright (c) 2022 Julien Posso
"""

import torch
import torch.nn as nn
import torchvision.models


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        batchnorm=True,
        activation=True
    ):

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]

        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation:
            layers.append(nn.ReLU6())

        super(ConvBnAct, self).__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, batchnorm=True):
        super(InvertedResidual, self).__init__()
        # Only stride of 1 and 2 in Mobilenet-v2
        assert stride in [1, 2]

        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_channels = int(round(in_channels * expand_ratio))

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBnAct(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1,
                                    batchnorm=batchnorm))

        layers.extend([
            ConvBnAct(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,
                      batchnorm=batchnorm, stride=stride, groups=hidden_channels),

            ConvBnAct(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1,
                      batchnorm=batchnorm, activation=False),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            x = self.conv(x) + x
        else:
            x = self.conv(x)
        return x


class MyMobileURSONet(nn.Module):
    def __init__(self, dropout_rate, n_ori_outputs, n_pos_outputs, batchnorm=True):
        super(MyMobileURSONet, self).__init__()

        img_channel = 3
        input_channel = 32
        last_channel = 1280

        inverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building input quantization and first convolution layer
        layers = [
            ConvBnAct(in_channels=img_channel, out_channels=input_channel, stride=2, padding=1, batchnorm=batchnorm)
        ]
        in_ch = input_channel

        for t, c, n, s in inverted_residual_settings:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels=in_ch, out_channels=c, stride=stride,
                                               batchnorm=batchnorm, expand_ratio=t))
                in_ch = c

        layers.append(ConvBnAct(in_ch, last_channel, kernel_size=1, batchnorm=batchnorm))

        self.features = nn.Sequential(*layers)

        # Position branch
        self.pos = nn.Sequential(
            nn.Linear(last_channel, n_pos_outputs)
        )

        # Orientation branch
        self.ori = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, n_ori_outputs),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        pos = self.pos(x)
        ori = self.ori(x)
        return ori, pos


def import_my_mobile_ursonet(dropout_rate, ori_type, n_ori_bins, pretrained=True):

    n_ori_outputs = 4 if ori_type == 'Regression' else n_ori_bins ** 3
    model = MyMobileURSONet(dropout_rate, n_ori_outputs, n_pos_outputs=3)

    if pretrained:
        model.features.load_state_dict(copy_state_dict(torchvision.models.mobilenet_v2(pretrained=True).
                                                       features.state_dict(), model.features.state_dict()))

    return model


def copy_state_dict(state_dict_1, state_dict_2):
    """Manual copy of state dict.
    Why ? Because when copying a state dict to another with load_state_dict, the values of weight are copied only
    when keys are the same in both state_dict, even if strict=False.
    """

    state1_keys = list(state_dict_1.keys())
    state2_keys = list(state_dict_2.keys())

    for x in range(len(state1_keys)):
        state_dict_2[state2_keys[x]] = state_dict_1[state1_keys[x]]

    return state_dict_2
