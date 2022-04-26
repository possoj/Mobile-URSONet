"""
Copyright (c) 2022 Julien Posso
"""

import torch.nn as nn
import torchvision.models
from torchvision.models import MobileNetV2


class PytorchMobileURSONet(MobileNetV2):

    def __init__(self, dropout_rate, n_ori_outputs, n_pos_outputs):
        super(PytorchMobileURSONet, self).__init__()

        num_ftrs = self.classifier[1].in_features

        # Position branch
        self.pos = nn.Sequential(
            nn.Linear(num_ftrs, n_pos_outputs)
        )

        # Orientation branch
        self.ori = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, n_ori_outputs),
        )

        # Set to none => not taken into account when counting parameters
        self.classifier = None

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        pos = self.pos(x)
        ori = self.ori(x)
        return ori, pos


def import_pytorch_mobile_ursonet(dropout_rate, ori_type, n_ori_bins, pretrained=True):

    n_ori_outputs = 4 if ori_type == 'Regression' else n_ori_bins ** 3
    model = PytorchMobileURSONet(dropout_rate, n_ori_outputs, n_pos_outputs=3)

    if pretrained:
        model.features.load_state_dict(torchvision.models.mobilenet_v2(pretrained=True).features.state_dict())

    return model
