import glob
import json
import os
import shutil
from pathlib import Path
from typing import Union, List

import albumentations
import numpy as np
import pandas as pd
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
    'batch_size': 256,
    'prediction_summary_files_path': '../predictions',
    'unlabeled_data_dir': '../../datasets/unlabeled'
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
            'Acanthastrea',
            'Acanthophyllia & Cynarnia',
            'Acropora',
            'Alveopora & Goniopora',
            'Blastomussa',
            'Bubble Coral',
            'Bubble Tip Anemone',
            'Candy Cane Coral',
            'Carpet Anemone',
            'Chalice',
            'Cyphastrea',
            'Discosoma Mushroom',
            'Elegance Coral',
            'Euphyllia',
            'Favia',
            'Gorgonia',
            'Leptastrea',
            'Leptoseris',
            'Lobophyllia & Trachyphyllia & Wellsophyllia',
            'Maze Brain Coral Platygyra',
            'Mini Carpet Anemone',
            'Montipora',
            'Pavona',
            'Plate Coral Fungia',
            'Porites',
            'Psammacora',
            'Rhodactis Mushroom',
            'Ricordea Mushroom',
            'Rock Flower Anemone',
            'Scolymia',
            'Scroll Corals Turbinaria',
            'Star Polyps',
            'Styllopora & Pocillipora & Seriatopora',
            'Sun Corals',
            'Toadstool & Leather Coral',
            'Tridacna Clams',
            'Zoa'
        ]

        self.class_lookup_by_name = dict([(c, i) for i, c in enumerate(self.classes)])
        self.class_lookup_by_index = dict([(i, c) for i, c in enumerate(self.classes)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.resize is not None:
            try:
                image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
            except Exception as ex:
                print(f"Exception when resizing the image: {image_name}")
                raise

        image = np.array(image)

        if self.augmentations is not None:
            try:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
            except Exception as ex:
                print(f"Exception when augmenting the image: {image_name}")
                raise

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        img_tensor = torch.tensor(image, dtype=torch.float)

        return image_name, img_tensor


def main() -> None:
    folders = os.listdir(CONFIG['unlabeled_data_dir'])
    folders.sort()
    already_processed = glob.glob(os.path.join(CONFIG["prediction_summary_files_path"], '*.json'))
    already_processed = [folder.split('/')[-1].replace('.json', '') for folder in already_processed]

    for i, folder in enumerate(folders):
        print("=" * 80)
        print(f"Progress {i + 1}/{len(folders)}")
        if folder in already_processed:
            continue
        _process_folder(CONFIG['unlabeled_data_dir'], folder)

    for csv in glob.glob(os.path.join(CONFIG['prediction_summary_files_path'], '*.csv')):
        _move_files(csv)

    print("=" * 80)
    print("DONE")


def _process_folder(data_dir: Union[Path, str], folder: str):
    folder_path = os.path.join(data_dir, folder)
    image_list = [os.path.join(folder_path, image) for image in os.listdir(folder_path)]

    results = dict([(img_name, []) for img_name in image_list])

    print(f"Running prediction on folder: {folder}")

    unlabeled_augs = albumentations.Compose([
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
    ])
    dataset = UnlabeledDataset(image_list, resize=(224, 224), augmentations=unlabeled_augs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

    checkpoints = glob.glob(os.path.join(CONFIG['checkpoint_store_path'], '*.ckpt'))
    checkpoints.sort()

    for checkpoint in checkpoints:
        fold = checkpoint.split('-')[4]
        print(f"Running prediction for fold: {fold}")

        model = OneCycleModule.load_from_checkpoint(checkpoint, map_location=lambda storage, loc: storage).to(device)
        model.freeze()
        _predict(model, checkpoint, dataloader, dataset, fold, results)

    json_file_name = os.path.join(CONFIG['prediction_summary_files_path'], f'{folder}.json')
    with open(json_file_name, 'w') as json_file:
        json.dump(results, json_file)

    _process_json(json_file_name, dataset.class_lookup_by_index)


def _forward(model: OneCycleModule, x: torch.Tensor):
    logits = model(x)
    probas = torch.nn.functional.softmax(logits, dim=1)

    return probas


def _predict(model: OneCycleModule,
             checkpoint: Union[Path, str],
             dataloader: torch.utils.data.DataLoader,
             dataset: UnlabeledDataset,
             fold: int,
             results: dict):
    for names, samples in tqdm(dataloader, total=len(dataset) // CONFIG['batch_size']):
        _predict_on_batch(model, checkpoint, dataset, fold, names, results, samples)


def _predict_on_batch(model: OneCycleModule,
                      checkpoint: Union[Path, str],
                      dataset: UnlabeledDataset,
                      fold: int,
                      names: Union[List[str], List[Path]],
                      results: dict,
                      samples: torch.Tensor):
    probabilities = _forward(model, samples.to(device))
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


def _process_json(json_file_name: Union[Path, str], index_to_class_label_lookup: dict):
    final_results = []
    print("Processing")
    json_file_path = os.path.join(CONFIG['prediction_summary_files_path'], json_file_name)
    with open(json_file_path) as json_file:
        data = json.load(json_file)
        for key in tqdm(data.keys()):
            df = pd.DataFrame(data=data[key])

            # unanimous vote, all k-Fold classifiers agreed on the class label
            if len(df.predicted_class.unique()):
                final_results.append(
                    {
                        'img_path': df.img_path.unique()[0],
                        'avg_confidence': df.confidence.mean(),
                        'predicted_class': df.predicted_class.unique()[0],
                        'predicted_class_label': df.predicted_class_label.unique()[0]
                    }
                )
            else:
                # soft voting
                probs = np.array(df.probabilities.values.tolist())
                averages = np.average(probs, axis=0)
                majority = np.argmax(averages)
                avg_confidence = np.max(averages)
                class_label = index_to_class_label_lookup[majority]

                final_results.append(
                    {
                        'img_path': df.img_path.unique()[0],
                        'avg_confidence': avg_confidence,
                        'predicted_class': majority,
                        'predicted_class_label': class_label
                    }
                )

        final_df = pd.DataFrame(data=final_results)
        print("Saving final results DataFrame for current folder.")
        final_df.to_csv(
            os.path.join(CONFIG['prediction_summary_files_path'], json_file_name.replace(".json", ".csv")),
            index=False,
            header=True)


def _move_files(csv_file_path: Union[Path, str]):
    df = pd.read_csv(csv_file_path)
    pretty_sure = df[df.avg_confidence >= 0.95]
    unsure = df[df.avg_confidence < 0.95]

    folder = csv_file_path.split('/')[-1].replace('.csv', '')
    low_confidence = 'low_confidence'

    for c in pretty_sure.predicted_class_label.unique():
        os.makedirs(os.path.join(CONFIG['unlabeled_data_dir'], folder, c), exist_ok=True)

    os.makedirs(os.path.join(CONFIG['unlabeled_data_dir'], folder, low_confidence), exist_ok=True)

    def __move_images(frame, class_name=None):
        for i, row in tqdm(frame.iterrows(), total=len(frame)):
            img_name = row.img_path.split('/')[-1]
            shutil.move(
                row.img_path,
                os.path.join(CONFIG['unlabeled_data_dir'],
                             folder,
                             class_name if class_name is not None else row.predicted_class_label,
                             img_name)
            )

    # save high confidence images
    print(f'Moving high confidence images to new location for folder: {folder}')
    __move_images(pretty_sure)
    # save low confidence images
    print(f'Moving low confidence images to new location for folder: {folder}')
    __move_images(unsure, low_confidence)


if __name__ == '__main__':
    pl.seed_everything(42)
    main()
