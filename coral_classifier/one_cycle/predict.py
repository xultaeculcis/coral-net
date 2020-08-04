import glob
import os
import json

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from coral_classifier.one_cycle.one_cycle_module import OneCycleModule

# ImageNet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
CONFIG = {
    'folds': 10,
    'checkpoint_store_path': '../../model-weights',
    'batch_size': 48
}
device = 'cuda'


#  --- Dataset ----
class UnlabeledDataset(Dataset):
    def __init__(self,
                 image_names: list,
                 resize=None,
                 augmentations=None):
        self.image_names = image_names
        self.resize = resize
        self.augmentations = augmentations
        self.classes = [
            "Montipora",
            "Other",
            "Acropora",
            "Zoa",
            "Euphyllia",
            "Chalice",
            "Acanthastrea"
        ]

        self.class_lookup_by_name = dict([(c, i) for i, c in enumerate(self.classes)])
        self.class_lookup_by_index = dict([(i, c) for i, c in enumerate(self.classes)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name)

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        img_tensor = torch.tensor(image, dtype=torch.float)

        return image_name, img_tensor


def main() -> None:
    folder = 'Aqua SD - Photos _ Facebook_files'
    data_dir = '../../datasets/unlabeled'
    folder_path = os.path.join(data_dir, folder)
    image_list = [os.path.join(folder_path, image) for image in os.listdir(folder_path)]

    results = dict([(img_name, []) for img_name in image_list])

    print(f"Running prediction on folder: {folder}")

    unlabeled_augs = albumentations.Compose([
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
    ])
    dataset = UnlabeledDataset(image_list, resize=(224, 224), augmentations=unlabeled_augs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

    for checkpoint in glob.glob(os.path.join(CONFIG['checkpoint_store_path'], '*.ckpt')):
        fold = checkpoint.split('-')[4]
        print(f"Running prediction for fold: {fold}")

        model = OneCycleModule.load_from_checkpoint(checkpoint, map_location=lambda storage, loc: storage).to(device)
        model.freeze()

        def forward(x):
            logits = model(x)
            probas = torch.nn.functional.softmax(logits, dim=1)
            return probas

        for names, samples in tqdm(dataloader, total=len(dataset) // CONFIG['batch_size']):
            probabilities = forward(samples.to(device))
            confidences, predicted_classes = torch.max(probabilities, dim=1)

            for (name,
                 probability,
                 confidence,
                 predicted_class) in zip(names, probabilities, confidences, predicted_classes):
                d = {
                    'img_path': name,
                    'probabilities': probability.cpu().numpy().tolist(),
                    'confidence': confidence.cpu().item(),
                    'predicted_class': predicted_class.cpu().item(),
                    'predicted_class_label': dataset.class_lookup_by_index[predicted_class.item()],
                    'fold': fold,
                    'checkpoint': checkpoint
                }
                results[name].append(d)

    with open(f'{folder}.json', 'w') as json_file:
        json.dump(results, json_file)


if __name__ == '__main__':
    pl.seed_everything(42)
    main()
