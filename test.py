import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import fastai
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader  
from fastai.callback.all import (CSVLogger, MixedPrecision, SaveModelCallback,
                                 ShowGraphCallback)
from fastai.vision.all import (BCEWithLogitsLossFlat, DataLoader,
                               ResNet50_Weights, SegmentationDataLoaders,
                               resnet50, unet_learner, DynamicUnet)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2



# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "./data/MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
PERSISTENT_WORKERS = True
if __name__ == "__main__":
	valid_transform = valid_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
					antialias=True),
			v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
						0.229, 0.224, 0.225])
		])
	test_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)
	test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE,
											shuffle=False, num_workers=NUM_WORKERS,
											pin_memory=True, sampler=None,
											persistent_workers=PERSISTENT_WORKERS)
	# model_path = "/models/fold_0_best"

	model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
	learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(), cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
	learn.load("fold_0_best")

	model.eval().to(DEVICE)

	# Get the first image and its corresponding segmentation mask
	# for i in range(test_data.__len__()):
	image, annotation = test_data.__getitem__(0)
	image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)
	# Preprocess and move to device
	preds, _ = learn.get_preds(dl=test_dl)

	# Visualize the input image and segmentation result
	plt.figure(figsize=(15, 10))
	plt.subplot(1, 3, 1)
	plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
	plt.title("Input Image")

	plt.subplot(1, 3, 2)
	plt.imshow(preds[0][0].cpu().numpy(), cmap="gray")
	plt.title("Segmentation Result")

	plt.subplot(1, 3, 3)
	plt.imshow(annotation.cpu().numpy(), cmap="gray")
	plt.title("Ground Truth")


	plt.show()