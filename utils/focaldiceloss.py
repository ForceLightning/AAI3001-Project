import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import FocalLossFlat, DiceLoss, store_attr

class CombinedLoss:
    def __init__(self, axis=1, smooth=1., alpha=1.) -> None:
        store_attr()
        self.axis = axis
        self.alpha = alpha
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)