import gc
import os

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
from foolbox.models.base import Model
from PIL import Image
from torchvision.transforms import v2
from tqdm.auto import tqdm

from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from utils.foolbox_utils import (BCELinfPGD, BCEPyTorchModel,
                                 BinaryTargetedMisclassification)
from utils.lossmetrics import BinaryDice, CombinedBCEDiceLoss

VALID_DIR = "./data/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/"
TEST_DIR = "./data/MoNuSegTestData/"
MODELS_DIR = "./models/"
OUTPUT_DIR = "./output/"
BATCH_SIZE = 1 # untested with other batch sizes
NUM_WORKERS = 0
NUM_FOLDS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True)
    ])

    valid_data = MoNuSegDataset(VALID_DIR, transform=valid_transform)
    test_data = MoNuSegDataset(
        TEST_DIR, transform=valid_transform, train=False)

    valid_dl = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          pin_memory=True, sampler=None, shuffle=False, persistent_workers=True,
                          device=DEVICE)
    test_dl_fastai = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                pin_memory=True, sampler=None, shuffle=False, persistent_workers=True,
                                device=DEVICE)
    test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    pin_memory=True, sampler=None, shuffle=False, persistent_workers=False)

    for fold_idx in tqdm(range(NUM_FOLDS), position=0, leave=True, desc="Fold"):
        model = create_unet_model(
            resnet50, n_out=1, img_size=(256, 256), pretrained=False
        )
        learn = Learner(dls=DataLoaders(test_dl_fastai), model=model, metrics=BinaryDice(),
                        cbs=[MixedPrecision(), CastToTensor()],
                        loss_func=CombinedBCEDiceLoss(alpha=0.5, reduction="mean"))
        learn = learn.load(f"fold_{fold_idx}_best")
        model.eval()
        del model.layers[-1]

        if torch.cuda.is_available():
            model.cuda()

        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225], axis=-3)

        fmodel = BCEPyTorchModel(model, bounds=(
            0, 1), preprocessing=preprocessing)

        desired_output = ep.astensor(torch.tensor(
            [[0] * 256 * 256] * BATCH_SIZE).to(DEVICE))
        criterion = BinaryTargetedMisclassification(desired_output)

        attack = BCELinfPGD()
        epsilons = [0.0, 1e-2, 3e-2, 1e-1, 3e-1]

        accuracy = 0

        for batch_idx, (images, target_mask) in enumerate(tqdm(test_dl, position=1, desc="Batch")):
            images = images.to(DEVICE)
            raw_advs, clipped_advs, success = attack(
                fmodel, images, criterion, epsilons=epsilons)
            robust_accuracy = 1 - success.detach().cpu().numpy().astype(float).mean(axis=-1)
            accuracy += robust_accuracy * images.shape[0]

            for i in range(images.shape[0]):
                index = batch_idx * BATCH_SIZE + i
                for j in range(len(epsilons)):
                    adv = clipped_advs[j][i]
                    adv = adv.permute(1, 2, 0).detach().cpu().numpy()
                    adv = (adv * 255).astype("uint8")
                    adv = Image.fromarray(adv)
                    OUTPUT_REL_DIR = os.path.join(
                        OUTPUT_DIR, f"fold_{fold_idx}")
                    if not os.path.exists(OUTPUT_REL_DIR):
                        os.makedirs(OUTPUT_REL_DIR)
                    adv.save(os.path.join(OUTPUT_REL_DIR,
                             f"img_{index}_eps_{j}.png"))

        accuracy /= len(test_data)
        print(f"Accuracy: {accuracy}")
        del learn.model
        del model
        del fmodel
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
