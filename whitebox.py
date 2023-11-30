import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import fastai
import csv
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from fastai.callback.all import (CSVLogger, MixedPrecision, SaveModelCallback,
                                 ShowGraphCallback)
from fastai.vision.all import (BCEWithLogitsLossFlat, DataLoader,
                               ResNet50_Weights, SegmentationDataLoaders,
                               resnet50, unet_learner, DynamicUnet)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2, ToPILImage
import torch
import torch.nn.functional as F
from utils.lossmetrics import PixelAccuracy, DiceCoefficient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "./data/MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
PERSISTENT_WORKERS = True

def fgsm_attack(model, images, labels, epsilon, image_index):
    images.requires_grad = True

    outputs = model(images).to(DEVICE)
    labels = labels.to(DEVICE)

    criterion = BCEWithLogitsLossFlat()
    loss = criterion(outputs, labels)

    model.zero_grad()

    loss.backward()

    perturbed_images = images + epsilon * images.grad.sign()

    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    perturbed_outputs = model(perturbed_images)
    perturbed_prediction = perturbed_outputs.cpu().detach().numpy()
    perturbed_prediction_tensor = torch.tensor(perturbed_prediction, device=DEVICE)

    
    perturbed_prediction_tensor = perturbed_prediction_tensor.to(DEVICE)
    labels = labels.to(DEVICE)

    acc = PixelAccuracy(predictions=perturbed_prediction_tensor, targets=labels)
    print("Accuracy:", acc)

    # Save perturbed images
    plt.figure(figsize=(15, 10))
    plt.imshow(perturbed_prediction[0][0], cmap="gray")
    plt.title("Epsilon: {}, Accuracy: {}".format(epsilon, acc))
    plt.savefig(f"./fgsm_images/image_{image_index}_epsilon_{epsilon}.png")
    plt.close()

    return perturbed_images, acc

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
    del model.layers[-1]

    epsilons = [10**i for i in range(4)]

    with open("metrics_results.csv", "w", newline='') as csvfile:
        fieldnames = ["Fold", "Epsilon", "Image", "Original Pixel Mean", "Original Dice Mean", "Perturbed Pixel Mean", "Perturbed Dice Mean"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for k in range(NUM_FOLDS):
            preds, _ = learn.get_preds(dl=test_dl)
            preds = preds.to(DEVICE)

            for image_index in range(len(test_data)):
                image, annotation = test_data.__getitem__(image_index)
                image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)
                annotation = annotation.to(DEVICE)

                for i, epsilon in enumerate(epsilons, 1):
                    perturbed_image, acc = fgsm_attack(model, image_tensor, annotation, epsilon, image_index)
                    print("Epsilon value:", epsilon)
                    original_prediction = preds[image_index]
                    print("Data", type(original_prediction.data))
                    print("Data", type(annotation))
                    original_accuracy = PixelAccuracy(predictions=original_prediction, targets=annotation)
                    original_dice = DiceCoefficient(original_prediction.data, annotation)
                    
                    perturbed_prediction = model(perturbed_image)
                    perturbed_prediction = torch.sigmoid(perturbed_prediction)
                    perturbed_accuracy = PixelAccuracy(predictions=perturbed_prediction, targets=annotation)
                    perturbed_dice = DiceCoefficient(perturbed_prediction.data, annotation)

                    writer.writerow({
                        "Fold": k,
                        "Epsilon": epsilon,
                        "Image": image_index,
                        "Original Pixel Mean": original_accuracy,
                        "Original Dice Mean": original_dice,
                        "Perturbed Pixel Mean": acc,
                        "Perturbed Dice Mean": perturbed_dice
                    })