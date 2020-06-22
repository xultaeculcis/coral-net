import os
from pprint import pprint

import albumentations
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine

from src.model import SEResnext50_32x4d

test_data_path = "../datasets/test/"
df = pd.read_csv("../datasets/test.csv")
device = "cuda"
data_dir = "../datasets/original"
CLASS_NAMES = os.listdir(data_dir)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
images = df.image.values.tolist()
images = [os.path.join(test_data_path, i) for i in images]
targets = df.label.values
encoded_targets = df[CLASS_NAMES].values


def predict_on_cv_fold(fold):
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=[224, 224],
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    return predict(fold, test_loader)


def predict(fold, test_loader):
    model_path = f"model_fold_{fold}.bin"

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    logits, valid_loss = Engine.evaluate(test_loader, model, device=device)

    return calculate_performance_metrics(fold, logits, valid_loss)


def calculate_performance_metrics(fold, logits, loss):
    stacked = np.vstack((logits))
    predictions = stacked.argmax(axis=1)
    accuracy = metrics.accuracy_score(targets, predictions)
    precision = metrics.precision_score(targets, predictions, average="macro")
    recall = metrics.recall_score(targets, predictions, average="macro")
    f1 = metrics.f1_score(targets, predictions, average="macro")
    cm = metrics.confusion_matrix(targets, predictions)

    metrics_dictionary = {
        "Fold": fold,
        "Test Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": cm
    }

    roc_auc_scores = []
    for i, c in enumerate(CLASS_NAMES):
        roc = metrics.roc_auc_score(encoded_targets[:, i], stacked[:, i])
        metrics_dictionary["ROC AUC for class: " + c] = roc
        roc_auc_scores.append(roc)

    metrics_dictionary["Aggregated ROC AUC"] = np.array(roc_auc_scores).mean()

    return logits, predictions, metrics_dictionary


def calculate_averages(metrics_dicts):
    mdf = pd.DataFrame(data=metrics_dicts)

    keys = metrics_dicts[0].keys()
    avg_dict = {}
    for k in keys:
        if k == "Fold" or k == "Confusion Matrix":
            continue
        avg_dict["Average " + k] = mdf[k].mean()

    metrics_dicts.append(avg_dict)
    return metrics_dicts


if __name__ == "__main__":
    ls0, p0, md0 = predict_on_cv_fold(0)
    ls1, p1, md1 = predict_on_cv_fold(1)
    ls2, p2, md2 = predict_on_cv_fold(2)
    ls3, p3, md3 = predict_on_cv_fold(3)
    ls4, p4, md4 = predict_on_cv_fold(4)

    aggregated = calculate_averages([md0, md1, md2, md3, md4])
    pprint(aggregated)

    print("+"*50)
    print("Ensemble metrics:")
    print("+"*50)

    def convert_to_numpy(tensor_list):
        new_ls = []
        for batch_idx, outputs in enumerate(tensor_list):
            outputs = outputs.cpu().numpy()
            new_ls.append(outputs)
        return np.array(new_ls)

    ls0 = convert_to_numpy(ls0)
    ls1 = convert_to_numpy(ls1)
    ls2 = convert_to_numpy(ls2)
    ls3 = convert_to_numpy(ls3)
    ls4 = convert_to_numpy(ls4)

    predictions = np.vstack((ls0 + ls1 + ls2 + ls3 + ls4) / 5)

    _, _, mets = calculate_performance_metrics("Ensemble", predictions, aggregated[-1]["Average Test Loss"])

    pprint(mets)
