import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


#  --- Dataset ----
class CoralFragDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 data_dir: str = "./",
                 train: bool = True,
                 resize=None,
                 augmentations=None):
        self.image_names = df['image']
        self.targets = df['label']
        self.data_dir = data_dir
        self.train = train
        self.resize = resize
        self.augmentations = augmentations
        self.classes = sorted(df['text_label'].unique().tolist())
        self.class_lookup_by_name = dict([(c, i) for i, c in enumerate(self.classes)])
        self.class_lookup_by_index = dict([(i, c) for i, c in enumerate(self.classes)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)
        targets = self.targets[idx]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        img_tensor = torch.tensor(image, dtype=torch.float)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        return img_tensor, target_tensor

    def label_for(self, idx):
        return self.targets[idx]

    def as_pillow(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)
        targets = self.targets[idx]
        return image, self.class_lookup_by_index[targets]

    def plot_sample_batch(self):
        fig, ax = plt.subplots(8, 4, figsize=(12, 20))
        t = np.random.randint(0, self.__len__())
        for i in range(8):
            for j in range(4):
                img, target = self.as_pillow(t)
                ax[i, j].set_title(target)
                ax[i, j].imshow(img)
                t += 1

        plt.show()
