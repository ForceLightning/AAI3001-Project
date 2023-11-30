"""
This file runs the conducts an adversarial attack on the model using the Basic Iterative Method (BIM).
"""
import os
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import fastai
import numpy as np
from torchvision.transforms import functional as F
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from fastai.vision.all import (
    BCEWithLogitsLossFlat, ResNet50_Weights,
    resnet50, MixedPrecision)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2
from tqdm.auto import tqdm
from utils.lossmetrics import PixelAccuracy, DiceCoefficient
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "data", "MoNuSegTestData")
BATCH_SIZE = 4
NUM_WORKERS = 4
PERSISTENT_WORKERS = True

# Basic Iterative Method Attack
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

# Plot images
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == "__main__":
    # Transform for test images
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    

    # Load test images into DataLoader
    test_data = MoNuSegDataset(ROOT_DIR, transform=test_transform)
    test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS,
                                    pin_memory=True, sampler=None,
                                    persistent_workers=PERSISTENT_WORKERS)
    # List of images and annotations
    images = []
    annotations = []

    # Attack parameters
    epsilons = [0.0, 1e-2, 1.5e-2, 3e-2, 6e-2, 1e-1, 3e-1]
    alphas = [0.2, 0.5, 1.0]
    num_iterations = 5

    sigmoid = torch.nn.Sigmoid()

    # Load images and annotations into lists
    for i in range(test_data.__len__()):
        image, annotation = test_data.__getitem__(i)
        image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
        images.append(image_tensor)

        annotation = annotation.unsqueeze(0).to(DEVICE)
        annotations.append(annotation)

    # Create directory to save images
    path = os.path.join(os.path.dirname(__file__), "bim_images")
    if not os.path.exists(path):
        os.mkdir(path)

    # Create csv file to save results
    csv_path = os.path.join(path, "results.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Fold", "Epsilon", "Alpha", "Image",
                         "Original Pixel Mean", 
                         "Original Dice Mean", 
                         "Perturbed Pixel Mean",
                         "Perturbed Dice Mean"])
    
    # Loading of the best model for each fold
    for k in tqdm(range(5)):
        model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
        learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(), cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
        learn.load("fold_"+ str(k) +"_best")
        print("Loaded model fold_"+ str(k) +"_best")

        model.eval().to(DEVICE)
        preds, _ = learn.get_preds(dl=test_dl)
        preds = preds.to(DEVICE)

        # Run the attack for each epsilon
        for i, epsilon in tqdm(enumerate(epsilons, 1)):

            # Run the attack for each alpha
            for alpha in alphas:

                original_total_acc = []
                perturbed_total_acc = []

                original_dice_scores = []
                perturbed_dice_scores = []

                # Run the attack for each image
                for j, (image, annotation) in enumerate(zip(images, annotations), 0):
                    perturbed_image = bim_attack(model, image, annotation, epsilon, alpha, num_iterations)
                    print("Epsilon value:", epsilon, "Alpha value:", alpha, "Image:", j)
                    
                    # Save the perturbed image
                    tensor_image = perturbed_image[0].cpu().detach().numpy().transpose(1, 2, 0)
                    tensor_image = np.clip(tensor_image, 0, 1)
                    plt.imsave(os.path.join(path, "{}_{}_{}_{}.png".format(k,epsilon, alpha, j)), tensor_image)

                    # Evaluate the model on the perturbed image
                    perturbed_prediction = model(perturbed_image)
                    perturbed_prediction = sigmoid(perturbed_prediction)

                    # Getting the original prediction based on index value
                    original_prediction = preds[j]
                    
                    # Compute the pixel accuracy
                    original_accuracy = PixelAccuracy(predictions=original_prediction, targets=annotation)
                    perturbed_accuracy = PixelAccuracy(predictions=perturbed_prediction, targets=annotation)

                    original_total_acc.append(original_accuracy)
                    perturbed_total_acc.append(perturbed_accuracy)

                    print("Original Accuracy: {:.2%}, Perturbed Accuracy: {:.2%}".format(original_accuracy, perturbed_accuracy))

                    # Compute the dice coefficient
                    original_dice = DiceCoefficient(predictions=original_prediction, targets=annotation)
                    perturbed_dice = DiceCoefficient(predictions=perturbed_prediction, targets=annotation)

                    original_dice_scores.append(original_dice)
                    perturbed_dice_scores.append(perturbed_dice)

                    print("Original Dice: {:.2%}, Perturbed Dice: {:.2%}".format(original_dice, perturbed_dice))            

                # Compute the mean and standard deviation of the pixel accuracy and dice coefficient
                ori_pa_mean = sum(original_total_acc) / len(images)
                
                ori_dice_mean = sum(original_dice_scores) / len(images)

                perb_pa_mean = sum(perturbed_total_acc) / len(images)

                perb_dice_mean = sum(perturbed_dice_scores) / len(images)

                # Save the mean and standard deviation of the pixel accuracy and dice coefficient
                with open(csv_path, "a") as f:
                    writer = csv.writer(f, delimiter=",")
                    writer.writerow([k, epsilon, alpha, j, 
                                    ori_pa_mean, ori_dice_mean,
                                    perb_pa_mean, perb_dice_mean,])
                    