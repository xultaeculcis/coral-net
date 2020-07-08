"""Computer vision example on Transfer Learning.
This computer vision example illustrates how one could fine-tune a pre-trained
network using pytorch-lightning. The training consists in three stages.
From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen
except maybe for the BatchNorm layers (depending on whether `train_bn = True`).
The BatchNorm layers (if `train_bn = True`) and the parameters of the classifier
are trained as a single parameters group with lr = 1e-3. From epoch 5 to 9,
the last two layer groups of the pre-trained network are unfrozen and added to the
optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3 for the
first parameter group in the optimizer). Eventually, from epoch 10, all the
remaining layer groups of the pre-trained network are unfrozen and added to
the optimizer as a third parameter group. From epoch 10, the parameters of the
pre-trained network are trained with lr = 1e-5 while those of the classifier
are trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import argparse
import itertools
import numbers
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pprint import pprint
from typing import Generator

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from pandas import DataFrame
from pytorch_lightning import _logger as pl_log
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import model_selection
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# reproducibility
SEED = 42  # Answer to the Ultimate Question of Life, the Universe, and Everything


#  --- Sampler  ----
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Original implementation: https://github.com/ufoym/imbalanced-dataset-sampler

    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super().__init__(dataset)
        self.dataset = dataset
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(idx)]
                   for idx in self.indices]
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def _get_label(self, idx):
        return self.dataset.label_for(idx)

    def __iter__(self):
        return (self.indices[i.item()] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


#  --- Base Pytorch Lightning module ----
class ParametersSplitsModuleMixin(pl.LightningModule, ABC):
    @abstractmethod
    def model_splits(self):
        """ Split the model into high level groups
        """
        pass

    def params_splits(self, only_trainable=False):
        """ Get parameters from model splits
        """
        for split in self.model_splits():
            params = list(filter_params(split, only_trainable=only_trainable))
            if params:
                yield params

    def trainable_params_splits(self):
        """ Get trainable parameters from model splits
            If a parameter group does not have trainable params, it does not get added
        """
        return self.params_splits(only_trainable=True)

    def freeze_to(self, n: int = None):
        """ Freezes model until certain layer
        """
        unfreeze(self.parameters())
        for params in list(self.params_splits())[:n]:
            freeze(params)

    def get_optimizer_param_groups(self, lr):
        lrs = self.get_lrs(lr)
        return [
            {"params": params, "lr": lr}
            for params, lr in zip(self.params_splits(), lrs)
        ]

    def get_lrs(self, lr):
        n_splits = len(list(self.params_splits()))
        if isinstance(lr, numbers.Number):
            return [lr] * n_splits
        if isinstance(lr, (tuple, list)):
            assert len(lr) == len(list(self.params_splits()))
            return lr


#  --- Pytorch-lightning module ---
class TransferLearningModel(ParametersSplitsModuleMixin):
    """Transfer Learning with pre-trained Model.
    Args:
        hparams: Model hyperparameters
        dl_path: Path where the data will be downloaded
    """

    def __init__(self,
                 folds: int = 5,
                 fold: int = 0,
                 classes: list = None,
                 backbone: str = 'resnet18',
                 root_data_path: str = './',
                 train_bn: bool = True,
                 batch_size: int = 32,
                 n_classes: int = 7,
                 n_outputs: int = 7,
                 num_workers: int = 8,
                 freeze_epochs: int = 2,
                 freeze_lrs: tuple = (0, 1e-2),
                 unfreeze_epochs: int = 4,
                 unfreeze_lrs: tuple = (1e-5, 1e-3),
                 weight_decay: float = 1e-4,
                 train_csv: str = None,
                 test_csv: str = None,
                 **kwargs) -> None:
        super().__init__()

        if n_classes == 7 and n_outputs != 7:
            raise ValueError(f"Invalid parameters passed to the module - for multiclass classification the number\n"
                             f"of outputs must be the same as number of classes - you passed: "
                             f"n_classes: {n_classes}, n_outputs {n_outputs}\n"
                             f"This combination is invalid.")

        if n_classes == 2 and n_outputs not in [1, 2]:
            raise ValueError(f"Invalid parameters passed to the module - for binary classification the number\n"
                             f"of outputs must be equal to either 2 (cross entropy loss is used)\n"
                             f"or 1 (binary cross entropy with logits is used) - you passed: "
                             f"n_classes: {n_classes}, n_outputs {n_outputs}\n"
                             f"This combination is invalid.")

        self.supported_architectures = ['googlenet',
                                        'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                                        'wide_resnet50_2', 'wide_resnet101_2']

        if backbone not in self.supported_architectures:
            raise Exception(f"The '{backbone}' is currently not supported as a backbone. "
                            f"Supported architectures are: {self.supported_architectures}")

        self.backbone = backbone
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.root_data_path = root_data_path
        self.classes = classes
        self.num_workers = num_workers
        self.folds = folds
        self.fold = fold
        self.freeze_epochs = freeze_epochs
        self.freeze_lrs = freeze_lrs
        self.unfreeze_epochs = unfreeze_epochs
        self.unfreeze_lrs = unfreeze_lrs
        self.weight_decay = weight_decay
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.__build_model()
        self.__setup()
        self.classes = self.train_dataset.classes

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)

        # 2. Classifier
        self.input_size = (224, 224) if self.backbone != 'googlenet' else (112, 112)
        _n_inputs = backbone.fc.in_features
        _fc_layers = [torch.nn.Linear(_n_inputs, 256),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(),
                      torch.nn.Linear(256, 32),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(),
                      torch.nn.Linear(32, self.n_outputs)]
        self.fc = torch.nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits if self.n_outputs == 1 else F.cross_entropy

    def __step(self, batch):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        loss = self.loss(y_logits, y if self.n_outputs > 1 else y.view((-1, 1)).type_as(x))

        with torch.no_grad():
            y_hat = torch.argmax(y_logits, dim=1) if self.n_outputs > 1 else (y_logits >= 0.0).squeeze(1).long()
            acc = plm.accuracy(y_hat, y, self.n_classes)

        return loss, acc

    def __metrics_per_batch(self, batch):
        # 1. Forward pass:
        x, y_true = batch
        logits = self.forward(x)

        # 2. Compute loss & performance metrics:
        # class prediction: if binary (num_outputs == 1) then class label is 0 if logit < 0 else it's 1
        # if multiclass then simply run argmax to find the index of the most confident class
        y_hat = torch.argmax(logits, dim=1) if self.n_outputs > 1 else (logits > 0.0).squeeze(1).long()
        loss = self.loss(logits, y_true if self.n_outputs > 1 else y_true.view((-1, 1)).type_as(x))
        acc = plm.accuracy(y_hat, y_true, num_classes=self.n_classes)
        prec = plm.precision(y_hat, y_true, num_classes=self.n_classes)
        rec = plm.recall(y_hat, y_true, num_classes=self.n_classes)
        f1 = plm.f1_score(y_hat, y_true, num_classes=self.n_classes)
        conf_matrix = plm.confusion_matrix(y_hat.long(), y_true.long())

        return (
            y_true,
            y_hat,
            logits,
            loss,
            acc,
            prec,
            rec,
            f1,
            conf_matrix
        )

    def __log_confusion_matrices(self, conf_matrix, stage):
        confusion_matrix_figure = _plot_confusion_matrix(
            cm=conf_matrix,
            target_names=self.classes,
            title=f"{stage} confusion matrix for fold #{self.fold}",
            normalize=False
        )
        normalized_confusion_matrix_figure = _plot_confusion_matrix(
            cm=conf_matrix,
            target_names=self.classes,
            title=f"{stage} confusion matrix for fold #{self.fold}",
            normalize=True
        )

        self.logger.experiment.add_figure(tag=f"{stage}/confusion_matrix",
                                          figure=confusion_matrix_figure,
                                          global_step=self.current_epoch)
        self.logger.experiment.add_figure(tag=f"{stage}/normalized_confusion_matrix",
                                          figure=normalized_confusion_matrix_figure,
                                          global_step=self.current_epoch)

    def forward(self, x):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def model_splits(self):
        return [self.feature_extractor, self.fc]

    def configure_optimizers(self):
        # passed lr does not matter, because scheduler will overtake
        param_groups = self.get_optimizer_param_groups(0)
        opt = AdamW(param_groups, weight_decay=self.weight_decay)
        # return a dummy lr_scheduler, so LearningRateLogger doesn't complain
        scheduler = OneCycleLR(opt, 0, 9)
        return [opt], [scheduler]

    def on_epoch_start(self):
        if self.current_epoch == 0:
            # Freeze all but last layer (imagine this is the head)
            self.freeze_to(-1)
            # Create new scheduler
            total_steps = len(self.train_dataloader()) * self.freeze_epochs
            lrs = self.get_lrs(self.freeze_lrs)
            opt = self.trainer.optimizers[0]
            scheduler = {'scheduler': OneCycleLR(opt, lrs, total_steps, pct_start=.9), 'interval': 'step'}
            scheduler = self.trainer.configure_schedulers([scheduler])
            # Replace scheduler and update lr logger
            self.trainer.lr_schedulers = scheduler
            lr_logger.on_train_start(self.trainer, self)

        if self.current_epoch == self.freeze_epochs:
            # Unfreeze all layers, we can also use `unfreeze`, but `freeze_to` has the
            # additional property of only considering parameters returned by `model_splits`
            self.freeze_to(0)
            # Create new scheduler
            total_steps = len(self.train_dataloader()) * self.unfreeze_epochs
            lrs = self.get_lrs(self.unfreeze_epochs)
            opt = self.trainer.optimizers[0]
            scheduler = {'scheduler': OneCycleLR(opt, lrs, total_steps, pct_start=.2), 'interval': 'step'}
            scheduler = self.trainer.configure_schedulers([scheduler])
            # Replace scheduler and update lr logger
            self.trainer.lr_schedulers = scheduler
            lr_logger.on_train_start(self.trainer, self)

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__step(batch)
        log = {
            'Train/loss': train_loss,
            'Train/acc': train_acc,
        }

        output = OrderedDict(
            {
                'loss': train_loss,
                'train_acc': train_acc,
                'log': log
            }
        )

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss'] for output in outputs]).mean()
        train_acc_mean = torch.stack([output['train_acc'] for output in outputs]).mean()
        print(f"\nTraining Epoch Loss: {train_loss_mean:.4f}, training epoch accuracy: {train_acc_mean:.4f}")

        log = {
            'Train/epoch/acc': train_acc_mean,
            'Train/epoch/loss': train_loss_mean,
            'step': self.current_epoch
        }

        return {
            'loss': train_loss_mean,
            'acc': train_acc_mean,
            'log': log
        }

    def validation_step(self, batch, batch_idx):
        y_true, y_hat, logits, loss, acc, prec, rec, f1, conf_matrix = self.__metrics_per_batch(batch)

        return {
            'loss': loss,
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'y_true': y_true,
            'y_logits': logits,
            'y_hat': y_hat,
        }

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        y_true, y_hat, loss, acc, prec, rec, f1, conf_matrix = self.__metrics_per_epoch(outputs)
        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Validation")

        print(f"\nValidation Epoch Loss: {loss:.4f}, validation epoch accuracy: {acc:.4f}")

        log = {
            'Validation/loss/epoch': loss,
            'Validation/acc/epoch': acc,
            'step': self.current_epoch,
            'Validation/precision/epoch': prec,
            'Validation/recall/epoch': rec,
            'Validation/f1_score/epoch': f1,
        }

        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,  # for saving the best model
            'log': log
        }

    def test_step(self, batch, batch_idx):
        y_true, y_hat, logits, loss, acc, prec, rec, f1, conf_matrix = self.__metrics_per_batch(batch)

        output = {
            'loss': loss,
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'y_true': y_true,
            'y_logits': logits,
            'y_hat': y_hat,
        }

        return output

    def test_epoch_end(self, outputs):
        """Compute and log test loss and accuracy at the epoch level."""
        y_true, y_hat, loss, acc, prec, rec, f1, conf_matrix = self.__metrics_per_epoch(outputs)

        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Test")

        print(f"\nTest Epoch Loss: {loss:.4f}, training epoch accuracy: {acc:.4f}")
        print(f"Confusion matrix: \n{conf_matrix.int()}")

        log = {
            'Test/epoch_loss': loss,
            'Test/epoch_acc': acc,
            'Test/epoch_precision': prec,
            'Test/epoch_recall': rec,
            'Test/epoch_f1_score': f1,
        }

        return {
            'log': log
        }

    def train_dataloader(self) -> DataLoader:
        pl_log.info('Training data loaded.')
        return self.__dataloader(stage='train')

    def val_dataloader(self) -> DataLoader:
        pl_log.info('Validation data loaded.')
        return self.__dataloader(stage='validation')

    def test_dataloader(self) -> DataLoader:
        pl_log.info('Test data loaded.')
        return self.__dataloader(stage='test')

    def __setup(self):
        df = pd.read_csv(self.train_csv)
        df = k_fold(df, self.folds)
        df_test = pd.read_csv(self.test_csv)

        resize_to = [224, 224] if self.backbone != 'googlenet' else [112, 112]

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

        test_image_names = df_test.image
        test_targets = df_test.label
        test_dataset = CoralFragDataset(image_names=test_image_names,
                                        targets=test_targets,
                                        binary=self.n_classes <= 2,
                                        root_dir=self.root_data_path,
                                        train=False,
                                        resize=resize_to,
                                        augmentations=valid_aug)

        df_train = df[df["kfold"] != self.fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == self.fold].reset_index(drop=True)

        train_image_names = df_train.image
        val_image_names = df_valid.image

        train_targets = df_train.label
        val_targets = df_valid.label

        train_dataset = CoralFragDataset(image_names=train_image_names,
                                         targets=train_targets,
                                         binary=self.n_classes <= 2,
                                         root_dir=self.root_data_path,
                                         train=True,
                                         resize=resize_to,
                                         augmentations=train_aug)

        val_dataset = CoralFragDataset(image_names=val_image_names,
                                       targets=val_targets,
                                       binary=self.n_classes <= 2,
                                       root_dir=self.root_data_path,
                                       train=True,
                                       resize=resize_to,
                                       augmentations=valid_aug)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def __dataloader(self, stage='train'):
        """Train/validation loaders."""

        _dataset = self.train_dataset if stage == 'train' \
            else self.val_dataset if stage == 'validation' \
            else self.test_dataset

        loader = DataLoader(dataset=_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            sampler=ImbalancedDatasetSampler(_dataset) if stage == 'train' else None)

        return loader

    @staticmethod
    def __metrics_per_epoch(outputs):
        loss_mean = torch.stack([output[f'loss'] for output in outputs]).mean()
        acc_mean = torch.stack([output[f'acc'] for output in outputs]).mean()
        prec_mean = torch.stack([output[f'precision'] for output in outputs]).mean()
        rec_mean = torch.stack([output[f'recall'] for output in outputs]).mean()
        f1_mean = torch.stack([output[f'f1_score'] for output in outputs]).mean()
        y_true = torch.cat([output['y_true'] for output in outputs], dim=-1)
        y_hat = torch.cat([output['y_hat'] for output in outputs], dim=-1)

        confusion_matrix = plm.confusion_matrix(y_hat, y_true)

        return (
            y_true,
            y_hat,
            loss_mean,
            acc_mean,
            prec_mean,
            rec_mean,
            f1_mean,
            confusion_matrix
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='resnet18',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--batch-size',
                            default=64,
                            type=int,
                            metavar='B',
                            help='Batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='Number of GPUs to use')
        parser.add_argument('--wd',
                            '--weight-decay',
                            default=1e-4,
                            type=float,
                            metavar='WD',
                            help='The weight decay value for the AdamW optimizer',
                            dest='weight_decay')
        parser.add_argument('--num-workers',
                            default=8,
                            type=int,
                            metavar='W',
                            help='Number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--num-classes',
                            default=7,
                            type=int,
                            help='Number of classes to classify',
                            dest='n_classes')
        parser.add_argument('--num-outputs',
                            default=7,
                            type=int,
                            help='Number of outputs from the final classification layer - '
                                 'will determine if binary cross-entropy with logits or cross-entropy loss is used',
                            dest='n_outputs')
        parser.add_argument('--folds',
                            default=5,
                            type=int,
                            metavar='F',
                            help='Number of folds in k-Fold Cross Validation',
                            dest='folds')
        parser.add_argument('--train-bn',
                            default=True,
                            type=bool,
                            metavar='TB',
                            help='Whether the BatchNorm layers should be trainable',
                            dest='train_bn')
        parser.add_argument('--freeze-epochs',
                            default=2,
                            type=int,
                            dest='freeze_epochs',
                            help='For how many epochs the feature extractor should be frozen')
        parser.add_argument('--freeze-lrs',
                            default=(0, 1e-2),
                            type=tuple,
                            dest='freeze_lrs',
                            help='The min and max learning rate while feature extractor is frozen')
        parser.add_argument('--unfreeze-epochs',
                            default=4,
                            type=int,
                            dest='unfreeze_epochs',
                            help='For how many epochs feature extractor should be trained together with the classifier')
        parser.add_argument('--unfreeze-lrs',
                            default=(1e-5, 1e-3),
                            type=tuple,
                            dest='unfreeze_lrs',
                            help='The min and max learning rate after feature extractor is unfrozen')
        parser.add_argument('--train-csv',
                            default="../datasets/train.csv",
                            type=str,
                            help='Path to train csv file',
                            dest='train_csv')
        parser.add_argument('--test-csv',
                            default="../datasets/test.csv",
                            type=str,
                            help='Path to test csv file',
                            dest='test_csv')
        parser.add_argument('--save-model-path',
                            default="../model-weights",
                            type=str,
                            help='Where to save the best model checkpoints',
                            dest='save_model_path')
        parser.add_argument('--save-top-k',
                            default=1,
                            type=int,
                            help='How many best k models to save',
                            dest='save_top_k')
        return parser


#  --- Dataset ----
class CoralFragDataset(Dataset):
    def __init__(self,
                 image_names: list,
                 targets: list,
                 binary=False,
                 root_dir: str = "./",
                 train: bool = True,
                 resize=None,
                 augmentations=None):
        self.image_names = image_names
        self.targets = targets
        self.root_dir = root_dir
        self.train = train
        self.resize = resize
        self.augmentations = augmentations
        self.binary = binary
        self.classes = [
            "Montipora",
            "Other",
            "Acropora",
            "Zoa",
            "Euphyllia",
            "Chalice",
            "Acanthastrea"
        ] if not binary else [
            "Other",
            "Acropora"
        ]
        self.class_lookup_by_name = dict([(c, i) for i, c in enumerate(self.classes)])
        self.class_lookup_by_index = dict([(i, c) for i, c in enumerate(self.classes)])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, "train" if self.train else "test", image_name)
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
        image_path = os.path.join(self.root_dir, "train" if self.train else "test", image_name)
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


def filter_params(module: torch.nn.Module, bn: bool = True, only_trainable=False) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        bn: If False, don't return batch norm layers
        only_trainable: If True, get only trainable params

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not isinstance(module, BN_TYPES) or bn:
            for param in module.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, bn=bn, only_trainable=only_trainable):
                yield param


#  --- Utility functions ---
def print_system_info() -> None:
    # If there's a GPU available...
    if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the:', torch.cuda.get_device_name(0))
        print('GPU capability:', torch.cuda.get_device_capability(0))
        print('GPU properties:', torch.cuda.get_device_properties(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


def _plot_confusion_matrix(cm,
                           target_names,
                           title='Confusion matrix',
                           cmap=None,
                           normalize=True):
    """
    Given a confusion matrix (cm), make a nice plot.
    Based on Scikit Learn's implementation.

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    figure = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def k_fold(df: DataFrame, k: int = 5):
    print(f"Creating folds. k = {k}")
    df["kfold"] = -1
    new_frame = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    y = new_frame.label.values
    kf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    for f, (t_, v_) in enumerate(kf.split(X=new_frame, y=y)):
        new_frame.loc[v_, 'kfold'] = f

    print(f"Created {k} folds for the cross validation")

    return new_frame


def plot_single_batch(loader: DataLoader, dataset: CoralFragDataset) -> None:
    batch = next(iter(loader))
    fig, ax = plt.subplots(8, 4, figsize=(12, 20))
    idx = 0
    for i in range(8):
        for j in range(4):
            image = batch[0][idx].permute(1, 2, 0)
            target = batch[1][idx].item()
            ax[i, j].set_title(dataset.class_lookup_by_index[target])
            ax[i, j].imshow(image)
            idx += 1

    plt.show()


def main(args: argparse.Namespace) -> None:
    """Train the model.
    Args:
        args: Model hyper-parameters
    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

    print_system_info()
    print("Using following configuration: ")
    pprint(args)

    for fold in range(1):
        print(f"Fold {fold}: Training is starting...")
        model = TransferLearningModel(fold=fold, **vars(args))
        logger = TensorBoardLogger("logs", name=f"{args.backbone}-fold-{fold}")

        nb_epochs = args.freeze_epochs + args.unfreeze_epochs
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='max'
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(args.save_model_path, f"checkpoint-fold-{fold}" + "-{epoch:02d}-{val_f1:.2f}"),
            save_top_k=args.save_top_k,
            monitor="val_f1",
            mode="max",
            verbose=True
        )
        trainer = pl.Trainer(
            weights_summary=None,
            num_sanity_val_steps=0,
            gpus=args.gpus,
            min_epochs=nb_epochs,
            max_epochs=nb_epochs,
            logger=logger,
            deterministic=True,
            benchmark=False,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            callbacks=[lr_logger]
            # fast_dev_run=True
        )

        trainer.fit(model)
        print("-" * 80)
        print(f"Testing the model on fold: {fold}")
        trainer.test(model)


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--root-data-path',
                               metavar='DIR',
                               type=str,
                               default="../datasets",
                               help='Root directory where to download the data',
                               dest='root_data_path')
    parser = TransferLearningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


if __name__ == '__main__':
    pl.seed_everything(seed=SEED)
    torch.manual_seed(SEED)
    lr_logger = LearningRateLogger()
    main(get_args())
