import os
from functools import partial
from typing import Callable, TypeVar

import eagerpy as ep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.callback.all import MixedPrecision
from fastai.data.core import DataLoader, DataLoaders
from fastai.learner import CastToTensor
from fastai.vision.all import Learner, create_unet_model, resnet50
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD
from foolbox.criteria import TargetedMisclassification
from foolbox.models.base import Model
from PIL import Image
from torchvision.transforms import v2
from tqdm.auto import tqdm

T = TypeVar("T")

class BCEPyTorchModel(PyTorchModel):
    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self._preprocess(x)
        z = ep.astensor(self._model(y.raw))
        z = z.reshape((z.shape[0], -1))
        w = restore_type(z)
        return w

class BinaryTargetedMisclassification(TargetedMisclassification):
    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        batch_size = outputs_.shape[0]
        del perturbed, outputs

        neg_outputs = 1 - outputs_
        outputs_ = ep.stack((neg_outputs, outputs_), axis=len(outputs_.shape)-1)
        classes = outputs_.argmax(axis=1)
        assert classes.shape == self.target_classes.shape, f"{classes.shape} != {self.target_classes.shape}"
        is_adv = classes == self.target_classes
        is_adv = is_adv.reshape((batch_size, -1))
        is_adv = is_adv.all(axis=-1)
        return restore_type(is_adv)

class BCELinfPGD(LinfPGD):
    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor, labels: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            # batch_size = logits.shape[0]
            logits = logits.reshape((-1, 1)).squeeze(-1)
            logits_neg = 1 - logits
            logits = ep.stack((logits_neg, logits), axis=1)
            labels = labels.reshape((-1, 1)).squeeze(-1)
            return ep.crossentropy(logits, labels).sum()
        return partial(loss_fn, labels=labels)
