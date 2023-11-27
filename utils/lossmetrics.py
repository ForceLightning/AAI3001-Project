import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import FocalLossFlat, DiceLoss, store_attr, BaseLoss
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
        threshold: float = 0.5
    ):
        store_attr('axis,eps,reduction,square_in_union,threshold')

    def __call__(self, pred: torch.Tensor, targ: torch.Tensor):
        pred = pred.squeeze(self.axis)
        pred, targ = map(self._contiguous, (pred, targ))
        pred = pred.view(-1, pred.shape[-1])
        targ = targ.view(-1, targ.shape[-1])
        pred, targ = TensorBase(pred), TensorBase(targ)
        assert pred.shape == targ.shape, "input and target dimensions must match"
        pred = self.activation(pred)
        pred = (pred > self.threshold).float()
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
        is_2d:bool=True,
        threshold:float=0.5,
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
