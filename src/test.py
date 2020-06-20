import os

import albumentations
import numpy as np
import pandas as pd
import torch
from wtfml.data_loaders.image import ClassificationLoader
from sklearn import metrics
from wtfml.engine import Engine

from src.model import SEResnext50_32x4d


def predict(fold):
    test_data_path = "../datasets_train_test/test/"
    df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
    device = "cuda"
    model_path=f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".jpg") for i in images]
    targets = df.label.values

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions, valid_loss = Engine.evaluate(
        test_loader, model, device=device
    )
    predictions = np.vstack((predictions)).ravel()

    f1 = metrics.f1_score(targets, predictions, average="macro")
    print(f"Test set, Macro F1 = {f1}")

    return predictions