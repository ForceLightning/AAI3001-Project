# %%
import os

from tqdm.auto import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from fastai.vision.all import resnet50, create_unet_model, Learner
from fastai.data.core import DataLoaders, TensorBase, DataLoader
from fastai.learner import CastToTensor
from fastai.callback.all import MixedPrecision

import eagerpy as ep
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD
from foolbox.criteria import TargetedMisclassification

from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from utils.lossmetrics import BinaryDice, CombinedBCEDiceLoss

# %%
VALID_DIR = "./data/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/"
TEST_DIR = "./data/MoNuSegTestData/"
MODELS_DIR = "./models/"
OUTPUT_DIR = "./output/"
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_FOLDS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
valid_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
])

valid_data = MoNuSegDataset(VALID_DIR, transform=valid_transform)
test_data = MoNuSegDataset(TEST_DIR, transform=valid_transform, train=False)

# %%
valid_dl = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                      pin_memory=True, sampler=None, shuffle=False, persistent_workers=True,
                      device=DEVICE)
test_dl_fastai = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            pin_memory=True, sampler=None, shuffle=False, persistent_workers=True,
                            device=DEVICE)
test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                pin_memory=True, sampler=None, shuffle=False, persistent_workers=False)

# %%
model = create_unet_model(
    resnet50, n_out=1, img_size=(256, 256), pretrained=False
).cuda()

# %%
learn = Learner(dls=DataLoaders(test_dl_fastai), model=model, metrics=BinaryDice(),
                cbs=[MixedPrecision(), CastToTensor()],
                loss_func=CombinedBCEDiceLoss(alpha=0.5, reduction="mean"))

# %%
learn.load("fold_0_best")

# %%
model.eval()

# %%
del model.layers[-1]
model = model.cuda()

# %%
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

# %%
from typing import TypeVar


T = TypeVar("T")

class BCEPyTorchModel(PyTorchModel):
    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self._preprocess(x)
        z = ep.astensor(self._model(y.raw))
        z = z.reshape((-1, 1)).squeeze(1)
        w = restore_type(z)
        return w

# %%
fmodel = BCEPyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

class BinaryTargetedMisclassification(TargetedMisclassification):
    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        neg_outputs = 1 - outputs_
        outputs_ = ep.stack((neg_outputs, outputs_), axis=1)
        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.target_classes.shape
        is_adv = classes == self.target_classes
        is_adv = is_adv.reshape((BATCH_SIZE, -1))
        is_adv = is_adv.all(axis=-1)
        return restore_type(is_adv)

# %%
desired_output = ep.astensor(torch.tensor([0] * 256 * 256 * BATCH_SIZE).to(DEVICE))
criterion = BinaryTargetedMisclassification(desired_output)

# %%
from functools import partial
from typing import Callable
import foolbox


class Attack(LinfPGD):
    def get_loss_fn(self, model: foolbox.models.base.Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor, labels: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            logits = logits.reshape((-1, 1))
            logits_neg = 1 - logits
            logits = ep.stack((logits_neg, logits), axis=1).squeeze(-1)
            labels = labels.reshape((-1, 1)).squeeze(1)
            return ep.crossentropy(logits, labels).sum()
        return partial(loss_fn, labels=labels)

# %%
attack = Attack()

# %%
epsilons = [0.0, 0.2, 0.4, 0.8]
accuracy = 0
for batch, (images, target_mask) in enumerate(tqdm(test_dl)):
    images = images.to(DEVICE)
    raw_advs, clipped_advs, success = attack(
        fmodel, images, criterion=criterion, epsilons=epsilons
    )
    robust_accuracy = 1 - success.detach().cpu().numpy().astype(float).mean(axis=-1)
    accuracy += robust_accuracy * images.shape[0]
    for i in range(len(epsilons)):
        adv = clipped_advs[i]
        adv = adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        adv = (adv * 255).astype("uint8")
        adv = Image.fromarray(adv)
        adv.save(os.path.join(OUTPUT_DIR, f"wb_adv_{batch}_{i}.png"))
    

accuracy = accuracy / len(test_dl.dataset)
print(accuracy)
