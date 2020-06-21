import os

import albumentations
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine

from src.model import SEResnext50_32x4d


def predict(fold):
    test_data_path = "../datasets/test/"
    df = pd.read_csv("../datasets/test.csv")
    device = "cuda"
    model_path = f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image.values.tolist()
    images = [os.path.join(test_data_path, i) for i in images]
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
    stacked = np.vstack((predictions))
    predictions = stacked.argmax(axis=1)

    acc = metrics.accuracy_score(targets, predictions)
    prec = metrics.precision_score(targets, predictions, average="macro")
    rec = metrics.recall_score(targets, predictions, average="macro")
    f1 = metrics.f1_score(targets, predictions, average="macro")

    print(f"Test set performance metrics, "
          f"Fold={fold}, "
          f"Val_Loss={valid_loss}, Accuracy={acc}, Precision={prec}, Recall={rec}, Macro F1={f1}")

    return predictions


if __name__ == "__main__":
    p1 = predict(0)
    p2 = predict(1)
    p3 = predict(2)
    p4 = predict(3)
    p5 = predict(4)
    predictions = (p1 + p2 + p3 + p4 + p5) / 5
