import os

import albumentations
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping

from src.model import SEResnext50_32x4d


torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(fold):
    training_data_path = "../datasets/train/"
    df = pd.read_csv("../datasets/train.csv")
    device = "cuda"
    epochs = 10
    train_bs = 32
    valid_bs = 16
    IMG_SIZE = [224, 224]

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = SEResnext50_32x4d(pretrained="imagenet")
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.label.values

    valid_images = df_valid.image.values.tolist()
    valid_images = [os.path.join(training_data_path, i) for i in valid_images]
    valid_targets = df_valid.label.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=IMG_SIZE,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=IMG_SIZE,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=10, mode="max")

    history = []
    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )

        stacked = np.vstack((predictions))
        predictions = stacked.argmax(axis=1)

        acc = metrics.accuracy_score(valid_targets, predictions)
        prec = metrics.precision_score(valid_targets, predictions, average="macro")
        rec = metrics.recall_score(valid_targets, predictions, average="macro")
        f1 = metrics.f1_score(valid_targets, predictions, average="macro")

        history_entry = {
            "epoch": epoch,
            "Training Loss": train_loss,
            "Validation Loss": valid_loss,
            "Validation Accuracy": acc,
            "Validation Precision": prec,
            "Validation Recall": rec,
            "Validation F1-score (macro)": f1,
        }
        history.append(history_entry)

        print(f"Epoch={epoch}, Val_Loss={valid_loss}, Accuracy={acc}, Precision={prec}, Recall={rec}, Macro F1={f1}")
        scheduler.step(f1)

        es(f1, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break

    pd.DataFrame(data=history).to_csv(f"Fold-{fold}-history.csv", index=False)


if __name__ == "__main__":
    train(0)
    train(1)
    train(2)
    train(3)
    train(4)
