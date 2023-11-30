import os
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd
import fastai
import numpy as np
from torchvision.transforms import functional as F, ToPILImage
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
from utils.lossmetrics import PixelAccuracy, DiceCoefficient
import csv
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "./data/MoNuSegTestData"
BATCH_SIZE = 4
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

# def compute_dice_score(y_true, y_pred):
#     y_true = torch.tensor(y_true, dtype=torch.float32)
#     y_pred = torch.tensor(y_pred, dtype=torch.float32)

#     intersection = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true) + torch.sum(y_pred)
#     dice = (2.0 * intersection) / (union + 1e-8)
#     return dice.item()


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

    
    
    images = []
    annotations = []

    epsilons = [0.0, 1e-2, 1.5e-2, 3e-2, 6e-2, 1e-1, 3e-1]
    alphas = [0.2, 0.5, 1.0]
    num_iterations = 5
    sigmoid = torch.nn.Sigmoid()
    # epsilons = [0.5 * i for i in range(5)]
    # alphas = [0.2, 0.5]
    # num_iterations = 2

    if not os.path.exists("./bim_images"):
        os.mkdir("./bim_images")
        
    for i in range(test_data.__len__()):
        image, annotation = test_data.__getitem__(i)

        image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)
        images.append(image_tensor)

        annotation = annotation.unsqueeze(0).to(DEVICE)
        annotations.append(annotation)
    
    with open("./bim_images/results.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Fold", "Epsilon", "Alpha", "Image",
                         "Original Pixel Mean", 
                         "Original Dice Mean", 
                         "Perturbed Pixel Mean",
                         "Perturbed Dice Mean"])
    for k in range(5):
        model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
        learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(), cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
        learn.load("fold_"+ str(k) +"_best")

        model.eval().to(DEVICE)

        preds, _ = learn.get_preds(dl=test_dl)
        preds = sigmoid(preds).to(DEVICE)

        # Flatten
        # preds = preds.view(preds.shape[0], preds.shape[2], preds.shape[3])

        for i, epsilon in enumerate(epsilons, 1):
            print("Epsilon value:", epsilon)

            for alpha in alphas:
                print("Alpha value:", alpha)

                original_total_acc = []
                perturbed_total_acc = []

                original_dice_scores = []
                perturbed_dice_scores = []

                for j, (image, annotation) in enumerate(zip(images, annotations), 0):
                    print("Image:", j)
                    perturbed_image = bim_attack(model, image, annotation, epsilon, alpha, num_iterations)

                #     tensor_image = perturbed_image[0].cpu().detach().numpy().transpose(1, 2, 0)
                #     tensor_image = np.clip(tensor_image, 0, 1)
                #     plt.imsave("./bim_images/{}_{}_{}.png".format(epsilon, alpha, j), tensor_image)

                #     # Evaluate the model on the perturbed image
                    perturbed_prediction = model(perturbed_image)
                    perturbed_prediction = sigmoid(perturbed_prediction)

                    original_prediction = preds[j] #.cpu().detach().numpy()
                    
                #     print(original_prediction.shape)
                #     print(annotation.shape)
                #     # Compute the pixel accuracy
                #     original_accuracy = PixelAccuracy(predictions=original_prediction, targets=annotation)
                #     perturbed_accuracy = PixelAccuracy(predictions=perturbed_prediction, targets=annotation)

                #     original_total_acc.append(original_accuracy)
                #     perturbed_total_acc.append(perturbed_accuracy)

                #     print("Original Pixel Accuracy:", original_accuracy)
                #     print("Perturbed Pixel Accuracy:", perturbed_accuracy)


                #     original_dice = DiceCoefficient(predictions=original_prediction, targets=annotation)
                #     perturbed_dice = DiceCoefficient(predictions=perturbed_prediction, targets=annotation)

                #     original_dice_scores.append(original_dice)
                #     perturbed_dice_scores.append(perturbed_dice)

                #     print("Original Dice:", original_dice)
                #     print("Perturbed Dice:", perturbed_dice)             

                # ori_pa_mean = sum(original_total_acc) / len(images)
                # ori_pa_std = np.std(original_total_acc)
                
                # ori_dice_mean = sum(original_dice_scores) / len(images)
                # ori_dice_std = np.std(original_dice_scores)

                # perb_pa_mean = sum(perturbed_total_acc) / len(images)
                # perb_pa_std = np.std(perturbed_total_acc)

                # perb_dice_mean = sum(perturbed_dice_scores) / len(images)
                # perb_dice_std = np.std(perturbed_dice_scores)

                # with open("./bim_images/results.csv", "a") as f:
                #     writer = csv.writer(f, delimiter=",")
                #     writer.writerow([epsilon, alpha, 
                #                     ori_pa_mean, ori_pa_std, ori_dice_mean, ori_dice_std,
                #                     perb_pa_mean, perb_pa_std, perb_dice_mean, perb_dice_std])

                # print("Original Accuracy: {:.2%}, Perturbed Accuracy: {:.2%}".format(original_accuracy, perturbed_accuracy))

                # if (alpha == 0.2 and epsilon == 3.0) or (alpha == 5.0 and epsilon == 7.0):
                    plt.figure(figsize=(20, 10))
                    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, index 1
                    plt.imshow(image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0))
                    plt.title("Original Image")

                    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, index 4
                    plt.imshow(perturbed_image[0].cpu().detach().numpy().transpose(1, 2, 0))
                    plt.title("Perturbed Image")

                    plt.subplot(2, 2, 3)  # 2 rows, 2 columns, index 2
                    plt.imshow(original_prediction[0].cpu().detach().numpy(), cmap="gray")
                    plt.title("Original Prediction")

                    plt.subplot(2, 2, 4)  # 2 rows, 2 columns, index 3
                    plt.imshow(perturbed_prediction[0].cpu().detach().numpy().transpose(1, 2, 0), cmap="gray")
                    plt.title("Perturbed Prediction")

                    plt.show()


                # if alpha == 5.0 and epsilon == 7.0:
                #     plt.figure(figsize=(15, 10))
                #     plt.subplot(1, 3, 1)
                #     plt.imshow(image.numpy().transpose(1, 2, 0))
                #     plt.title("Original Image")

                #     plt.subplot(1, 3, 2)
                #     plt.imshow(original_prediction[0].cpu().detach().numpy(), cmap="gray")
                #     plt.title("Original Prediction")

                #     plt.subplot(1, 3, 3)
                #     plt.imshow(perturbed_prediction[0].cpu().detach().numpy().transpose(1, 2, 0), cmap="gray")
                #     plt.title("Perturbed Prediction")

                #     plt.subplot(1, 3, 4)
                #     plt.imshow(perturbed_image[0].cpu().detach().numpy().transpose(1, 2, 0))
                #     plt.title("Perturbed Image")

                #     plt.show()
        
    # df = pd.read_csv("./bim_images/results.csv")

    # # Create subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # # Plot mean accuracy
    # for alpha in df['Alpha'].unique():
    #     subset = df[df['Alpha'] == alpha]
    #     ax1.plot(subset['Epsilon'], subset['Mean'], label=f'Alpha={alpha}')

    # ax1.set_xlabel('Epsilon')
    # ax1.set_ylabel('Mean Pixel Accuracy')
    # ax1.legend(title='Alpha')
    # ax1.set_title('Epsilon vs Mean Pixel Accuracy for different Alpha values')

    # # Plot standard deviation
    # for alpha in df['Alpha'].unique():
    #     subset = df[df['Alpha'] == alpha]
    #     ax2.plot(subset['Epsilon'], subset['Standard Deviation'], label=f'Alpha={alpha}')

    # ax2.set_xlabel('Epsilon')
    # ax2.set_ylabel('Standard Deviation')
    # ax2.legend(title='Alpha')
    # ax2.set_title('Epsilon vs Standard Deviation for different Alpha values')

    # plt.show()


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

    # plt.figure(figsize=(15, 10))

    # best_perturbations = {} 

    # for i, epsilon in enumerate(tqdm(epsilons, desc="Epsilons"), 1):  
    #     best_alpha = None
    #     best_perturbed_image = None
    #     best_dice_score = 0.0 

    #     for alpha in tqdm(alphas, desc="Alphas", leave=False):
    #         perturbed_image = bim_attack(model, image_tensor, annotation, epsilon, alpha, num_iterations)

    #         perturbed_outputs = model(perturbed_image)
    #         perturbed_prediction = perturbed_outputs.cpu().detach().numpy()

    #         # Get best of alpha
    #         dice_score = compute_dice_score(annotation, perturbed_prediction[0][0])

    #         if dice_score > best_dice_score:
    #             best_dice_score = dice_score
    #             best_alpha = alpha
    #             best_perturbed_image = perturbed_image

    #     best_perturbations[epsilon] = best_perturbed_image.cpu().detach().numpy()

    # for i, epsilon in enumerate(epsilons, 1):
    #     plt.subplot(2, 3, i)
    #     plt.imshow(best_perturbations[epsilon][0][0], cmap="gray")
    #     plt.title("Epsilon: {}, Best Alpha: {}".format(epsilon, best_alpha))

    # plt.show()