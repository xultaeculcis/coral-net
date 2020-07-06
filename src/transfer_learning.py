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
import os
from collections import OrderedDict
from pprint import pprint
from typing import Optional, Generator

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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import model_selection
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# reproducibility
SEED = 42  # Answer to the Ultimate Question of Life, the Universe, and Everything


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


def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group({
        'params': filter_params(module=module, train_bn=train_bn),
        'lr': params_lr / 10.
    })


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


#  --- Pytorch-lightning module ---
class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained Model.
    Args:
        hparams: Model hyperparameters
        dl_path: Path where the data will be downloaded
    """

    def __init__(self,
                 fold: int = 0,
                 classes: list = None,
                 backbone: str = 'resnet18',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 batch_size: int = 16,
                 lr: float = 1e-3,
                 lr_scheduler_gamma: float = 1e-1,
                 n_classes: int = 7,
                 n_outputs: int = 7,
                 num_workers: int = 8, **kwargs) -> None:
        super().__init__()

        self.supported_architectures = ['googlenet',
                                        'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                                        'wide_resnet50_2', 'wide_resnet101_2']

        if backbone not in self.supported_architectures:
            raise Exception(f"The '{backbone}' is currently not supported as a backbone. "
                            f"Supported architectures are: {self.supported_architectures}")

        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.classes = classes
        self.num_workers = num_workers
        self.fold = fold
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

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

    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.train_bn)

        elif self.milestones[0] <= epoch < self.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.milestones[0]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                                          optimizer=optimizer,
                                          train_bn=self.train_bn)

        elif self.current_epoch == self.milestones[1]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                                          optimizer=optimizer,
                                          train_bn=self.train_bn)

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__step(batch)
        log = {
            'Train/loss': train_loss,
            'Train/acc': train_acc,
            'Train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
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
            'Train/epoch/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'step': self.current_epoch
        }

        return {
            'loss': train_loss_mean,
            'acc': train_acc_mean,
            'log': log
        }

    def validation_step(self, batch, batch_idx):
        y_true, y_hat, logits, loss, acc, prec, rec, f1, conf_matrix = self.__metrics_per_batch(batch)
        log = {
            'Validation/loss': loss,
            'Validation/acc': acc,
            'Validation/precision': prec,
            'Validation/recall': rec,
            'Validation/f1_score': f1,
        }

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
            'log': log
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

        log = {
            'Test/loss': loss,
            'Test/acc': acc,
            'Test/precision': prec,
            'Test/recall': rec,
            'Test/f1_score': f1,
        }

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
            'log': log
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

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.lr)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.milestones,
                                gamma=self.lr_scheduler_gamma)

        return [optimizer], [scheduler]

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
        parser.add_argument('--epochs',
                            default=2,
                            type=int,
                            metavar='N',
                            help='Total number of epochs',
                            dest='nb_epochs')
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
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='Initial learning rate',
                            dest='lr')
        parser.add_argument('--lr-scheduler-gamma',
                            default=1e-1,
                            type=float,
                            metavar='LRG',
                            help='Factor by which the learning rate is reduced at each milestone',
                            dest='lr_scheduler_gamma')
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
        parser.add_argument('--milestones',
                            default=[5, 10],
                            type=list,
                            metavar='M',
                            help='List of two epochs milestones')
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


def main(args: argparse.Namespace) -> None:
    """Train the model.
    Args:
        args: Model hyper-parameters
    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

    if args.n_classes == 7 and args.n_outputs != 7:
        raise ValueError(f"Invalid parameters passed to the module - for multiclass classification the number\n"
                         f"of outputs must be the same as number of classes - you passed: "
                         f"n_classes: {args.n_classes}, n_outputs {args.n_outputs}\n"
                         f"This combination is invalid.")

    if args.n_classes == 2 and args.n_outputs not in [1, 2]:
        raise ValueError(f"Invalid parameters passed to the module - for binary classification the number\n"
                         f"of outputs must be equal to either 2 (cross entropy loss is used)\n"
                         f"or 1 (binary cross entropy with logits is used) - you passed: "
                         f"n_classes: {args.n_classes}, n_outputs {args.n_outputs}\n"
                         f"This combination is invalid.")

    # TODO use one-cycle LR scheduler after unfreezing whole model

    print_system_info()
    print("Using following configuration: ")
    pprint(args)

    resize_to = [224, 224] if args.backbone != 'googlenet' else [112, 112]

    df = pd.read_csv(args.train_csv)
    df = k_fold(df, args.folds)
    df_test = pd.read_csv(args.test_csv)

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
                                    binary=args.n_classes <= 2,
                                    root_dir=args.root_data_path,
                                    train=False,
                                    resize=resize_to,
                                    augmentations=valid_aug)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    for fold in range(1):
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)

        train_image_names = df_train.image
        val_image_names = df_valid.image

        train_targets = df_train.label
        val_targets = df_valid.label

        train_dataset = CoralFragDataset(image_names=train_image_names,
                                         targets=train_targets,
                                         binary=args.n_classes <= 2,
                                         root_dir=args.root_data_path,
                                         train=True,
                                         resize=resize_to,
                                         augmentations=train_aug)

        val_dataset = CoralFragDataset(image_names=val_image_names,
                                       targets=val_targets,
                                       binary=args.n_classes <= 2,
                                       root_dir=args.root_data_path,
                                       train=True,
                                       resize=resize_to,
                                       augmentations=valid_aug)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=ImbalancedDatasetSampler(train_dataset),
                                  num_workers=args.num_workers)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers)

        print(f"Fold {fold}: Training is starting...")
        model = TransferLearningModel(fold=fold, classes=train_dataset.classes, **vars(args))
        logger = TensorBoardLogger("logs", name=f"{args.backbone}-fold-{fold}")
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
            min_epochs=args.nb_epochs,
            max_epochs=args.nb_epochs,
            logger=logger,
            deterministic=True,
            benchmark=False,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            # fast_dev_run=True
        )

        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
        print("-" * 80)
        print(f"Testing the model on fold: {fold}")
        trainer.test(model, test_dataloaders=test_loader)


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

    main(get_args())
