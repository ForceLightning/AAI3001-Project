import os
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import fastai
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
from foolbox import PyTorchModel, attacks
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from fastai.vision.all import (
    BCEWithLogitsLossFlat, DataLoader, ResNet50_Weights,
    resnet50, unet_learner, DynamicUnet, MixedPrecision
)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
PERSISTENT_WORKERS = True

def bim_attack(model, images, labels, epsilon, alpha, num_iterations):
    images.requires_grad = True
    original_images = images.clone().detach()

    for _ in range(num_iterations):
        outputs = model(images).to(DEVICE)
        labels = labels.to(DEVICE)

        criterion = BCEWithLogitsLossFlat()
        loss = criterion(outputs, labels)

        model.zero_grad()
        grad = autograd.grad(loss, images)[0]  

        images = images + alpha * grad.sign()
        images = torch.clamp(images, 0, 1)

        delta = torch.clamp(images - original_images, -epsilon, epsilon)
        images = original_images + delta

    return images

def compute_dice_score(y_true, y_pred):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


if __name__ == "__main__":
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    

    test_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)
    test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS,
                                    pin_memory=True, sampler=None,
                                    persistent_workers=PERSISTENT_WORKERS)

    model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
    learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(), cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
    learn.load("fold_0_best")

    model.eval().to(DEVICE)
   
    image, annotation = test_data.__getitem__(0)
    image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)

    preds, _ = learn.get_preds(dl=test_dl)
    print(preds.size())
    image_index = 0

    epsilons = [0.2 * i for i in range(6)]
    alphas = [0.05, 0.1, 0.2]  
    num_iterations = 10 

    plt.figure(figsize=(15, 10))

    best_perturbations = {} 

    for i, epsilon in enumerate(tqdm(epsilons, desc="Epsilons"), 1):  
        best_alpha = None
        best_perturbed_image = None
        best_dice_score = 0.0 

        for alpha in tqdm(alphas, desc="Alphas", leave=False):
            perturbed_image = bim_attack(model, image_tensor, annotation, epsilon, alpha, num_iterations)

            perturbed_outputs = model(perturbed_image)
            perturbed_prediction = perturbed_outputs.cpu().detach().numpy()

            # Get best of alpha
            dice_score = compute_dice_score(annotation, perturbed_prediction[0][0])

            if dice_score > best_dice_score:
                best_dice_score = dice_score
                best_alpha = alpha
                best_perturbed_image = perturbed_image

        best_perturbations[epsilon] = best_perturbed_image.cpu().detach().numpy()

    for i, epsilon in enumerate(epsilons, 1):
        plt.subplot(2, 3, i)
        plt.imshow(best_perturbations[epsilon][0][0], cmap="gray")
        plt.title("Epsilon: {}, Best Alpha: {}".format(epsilon, best_alpha))

    plt.show()

    # plt.figure(figsize=(15, 10))

    # for i, epsilon in enumerate(epsilons, 1):
    #     for alpha in alphas:
    #         perturbed_image = bim_attack(model, image_tensor, annotation, epsilon, alpha, num_iterations)
    #         print("Epsilon value:", epsilon, "Alpha value:", alpha)

    #         perturbed_outputs = model(perturbed_image)
    #         perturbed_prediction = perturbed_outputs.cpu().detach().numpy()

    #         plt.subplot(len(epsilons), len(alphas), (i * len(alphas) + int(alpha / 0.05)) % 18 + 1)
    #         plt.imshow(perturbed_prediction[0][0], cmap="gray")
    #         plt.title("Epsilon: {}, Alpha: {}".format(epsilon, alpha))

    # plt.show()

