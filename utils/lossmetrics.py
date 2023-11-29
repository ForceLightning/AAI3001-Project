import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import FocalLossFlat, store_attr, BCEWithLogitsLossFlat
from fastai.torch_core import TensorBase, flatten_check
from fastcore.basics import store_attr
from fastai.metrics import Metric


class BinaryDiceLoss():
    def __init__(
        self,
        axis=1,
        eps: float = 1e-6,
        reduction: str = "sum",
        square_in_union: bool = False,
    ):
        store_attr('axis,eps,reduction,square_in_union')

    def __call__(self, pred: torch.Tensor, targ: torch.Tensor):
        pred = pred.squeeze(self.axis)
        pred, targ = map(self._contiguous, (pred, targ))
        pred = pred.view(-1, pred.shape[-1])
        targ = targ.view(-1, targ.shape[-1])
        pred, targ = TensorBase(pred), TensorBase(targ)
        assert pred.shape == targ.shape, "input and target dimensions must match"
        pred = self.activation(pred)
        sum_dims = list(range(1, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = torch.sum(pred**2 + targ, dim=sum_dims) \
            if self.square_in_union else torch.sum(pred + targ, dim=sum_dims)
        dice_score = (2. * inter + self.eps) / (union + self.eps)
        loss = 1 - dice_score
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through a sigmoid function.

        Args:
            x (torch.Tensor): Logits.

        Returns:
            torch.Tensor: Probabilities.
        """
        return torch.sigmoid(x)

    def decodes(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the predictions from probabilities.

        Args:
            x (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Predictions.
        """
        return (x > self.threshold).float()

    def _contiguous(self, x: torch.Tensor) -> TensorBase:
        """Returns a contiguous tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Contiguous tensor.
        """
        return TensorBase(x.transpose(self.axis, -1).contiguous()) if isinstance(x, torch.Tensor) else x


class BinaryDice(Metric):
    def __init__(
        self,
        activation=torch.sigmoid,
        is_2d: bool = True,
        threshold: float = 0.5,
        axis=1
    ):
        self.is_2d = is_2d
        self.activation = activation
        self.threshold = threshold
        self.axis = axis
        self.inter = 0.
        self.union = 0.

    def reset(self):
        self.inter = 0.
        self.union = 0.

    def accumulate(self, learn):
        pred = learn.pred
        pred = pred.squeeze(self.axis)
        pred, targ = flatten_check(pred, learn.y)
        assert pred.shape == targ.shape, "input and target dimensions must match"
        pred = self.activation(pred)
        pred = (pred > self.threshold).float()
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()

    @property
    def value(self):
        return 2. * self.inter/self.union if self.union > 0 else None

    def _contiguous(self, x: torch.Tensor) -> TensorBase:
        """Returns a contiguous tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Contiguous tensor.
        """
        return TensorBase(x.transpose(self.axis, -1).contiguous()) if isinstance(x, torch.Tensor) else x


class CombinedBCEDiceLoss:
    def __init__(self, axis=1, smooth=1e-6, alpha=1., reduction='mean'):
        store_attr('axis,smooth,alpha')
        self.bce = BCEWithLogitsLossFlat(axis=-1)
        self.dice = BinaryDiceLoss(axis=axis, eps=smooth, reduction=reduction)

    def __call__(self, pred:torch.Tensor, targ:torch.Tensor):
        return self.bce(pred, targ) + self.alpha * self.dice(pred, targ)

    def decodes(self, x): return x.argmax(dim=self.axis)
    def activation(self, x): return torch.sigmoid(x)

class PixelAccuracy:
    def __init__(self, prediction_mask, target_mask, device="cpu"):
        self.predictions = prediction_mask.to(device)
        self.targets = target_mask.to(device)

    def pixel_accuracy(self):
        # Predicted values are either 0 or 1 (Binary Mask)
        predicted = self.predictions.float()

        predicted = torch.where(self.predictions > 0, torch.tensor(1.0), torch.tensor(0.0)).float()
        # Target values are either 0 or 1 (Binary Mask)
        target = self.targets.float()

        correct_pixels = (predicted == target).float().sum()

        total_pixels = target.numel()

        accuracy = correct_pixels / total_pixels

        return accuracy.item()
