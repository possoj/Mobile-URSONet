"""
Copyright (c) 2022 Julien Posso
"""

import torch
from torch.nn.modules import Module


def pos_reg_loss(input, target, norm_distance=True):
    loss = torch.linalg.norm(input - target)
    if norm_distance:
        loss = loss / torch.linalg.norm(target)
    return loss


def ori_reg_loss(input, target, norm_distance=True, target_pos=None):
    inter_sum = torch.abs(torch.sum(input * target, dim=1, keepdim=True))
    # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1. See scoring for more details
    if True in inter_sum[inter_sum > 1.01]:
        raise ValueError("Error while computing orientation Loss")

    inter_sum[inter_sum > 1] = 0
    loss = torch.arccos(inter_sum)
    if norm_distance:
        loss = loss / torch.linalg.norm(target_pos, dim=1, keepdim=True)
    return loss


class POSREGLoss(Module):
    """ Loss function used for Position branch in regression configuration: we use the position score, which is the
    (relative) position error as defined per ESA metrics:
    https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
    """
    def __init__(self, reduction='mean', norm_distance=True):
        super(POSREGLoss, self).__init__()

        if reduction != 'mean' and reduction != 'sum':
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(self, input, target):
        loss = self.reduction(pos_reg_loss(input, target, self.norm_distance))
        return loss


class ORIREGLoss(Module):
    """ Loss function used for Orientation branch in regression configuration: we use the orientation score, which is
    the angle of the rotation, that aligns the estimated and ground truth orientations as defined per ESA metrics:
    https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
    We also propose to make this loss relative to the distance with the target. See Mobile-URSONet article for details.
    """
    def __init__(self, reduction='mean', norm_distance=True):
        super(ORIREGLoss, self).__init__()

        if reduction != 'mean' and reduction != 'sum':
            raise ValueError("reduction must be mean or sum")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(self, input, target, target_pos=None):
        loss = self.reduction(ori_reg_loss(input, target, self.norm_distance, target_pos))
        return loss


class ORICLASSLoss(Module):
    """ Loss function used for Orientation branch in classification configuration:
    we use a standard negative log-likelihood. See Mobile-URSONet article for details.
    """
    def __init__(self, reduction='mean'):
        super(ORICLASSLoss, self).__init__()
        if reduction != 'mean' and reduction != 'sum':
            raise ValueError("reduction must be mean or sum")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, input, target):
        loss = self.reduction(torch.sum(-(target * torch.log(input)), dim=1))
        if True in torch.isnan(loss):
            raise ValueError("Error while computing orientation Loss")
        return loss
