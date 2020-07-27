import numbers
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Generator

import albumentations
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset import CoralFragDataset
from src.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from src.utils import _plot_confusion_matrix, k_fold
import torchvision.models as models

# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def filter_params(module: nn.Module, bn: bool = True, only_trainable=False) -> Generator:
    """
    Yields the trainable parameters of a given module.
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


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


class ParametersSplitsModule(pl.LightningModule, ABC):
    printed = False

    @abstractmethod
    def model_splits(self):
        """
        Split the model into high level groups
        """
        pass

    def params_splits(self, only_trainable=False):
        """
        Get parameters from model splits
        """
        for split in self.model_splits():
            params = list(filter_params(split, only_trainable=only_trainable))
            if params:
                yield params

    def trainable_params_splits(self):
        """
        Get trainable parameters from model splits
        If a parameter group does not have trainable params, it does not get added
        """
        return self.params_splits(only_trainable=True)

    def freeze_to(self, n: int = None):
        """
        Freezes model until certain layer
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
class DifferentialLearningRatesModule(ParametersSplitsModule):
    """Transfer Learning with pre-trained Model.
    Args:
        hparams: Model hyperparameters
    """

    def __init__(self, hparams, milestones) -> None:
        super().__init__()

        self.supported_architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                        'resnext50_32x4d', 'resnext101_32x8d',
                                        'wide_resnet50_2', 'wide_resnet101_2']

        self.__validate_args(hparams.backbone, hparams.n_classes, hparams.n_outputs)
        self.hparams = hparams

        self.backbone = hparams.backbone
        self.train_bn = hparams.train_bn
        self.batch_size = hparams.batch_size
        self.n_classes = hparams.n_classes
        self.n_outputs = hparams.n_outputs
        self.is_binary = hparams.n_outputs <= 2
        self.root_data_path = hparams.root_data_path
        self.num_workers = hparams.num_workers
        self.folds = hparams.folds
        self.fold_number = hparams.fold
        self.seed = hparams.seed
        self.epochs = hparams.epochs
        self.weight_decay = hparams.weight_decay
        self.train_csv = hparams.train_csv
        self.test_csv = hparams.test_csv
        self.pct_start = hparams.pct_start
        self.div_factor = hparams.div_factor
        self.final_div_factor = hparams.final_div_factor
        self.base_momentum = hparams.base_momentum
        self.max_momentum = hparams.max_momentum
        self.milestones = milestones
        self.input_size = (224, 224)

        self.__build_model()
        self.__setup()

        self.classes = self.train_dataset.classes
        self.train_loader = self.__dataloader(stage='train')
        self.val_loader = self.__dataloader(stage='validation')
        self.test_loader = self.__dataloader(stage='test')
        self.total_steps = len(self.train_dataloader()) * self.epochs

    def __validate_args(self, backbone, n_classes, n_outputs):
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
        if backbone not in self.supported_architectures:
            raise Exception(f"The '{backbone}' is currently not supported as a backbone. "
                            f"Supported architectures are: {', '.join(self.supported_architectures)}")

    def __build_model(self):
        """Define model layers & loss."""

        model_func = getattr(models, self.backbone)
        self.model = model_func(pretrained=True)
        _n_inputs = self.model.fc.in_features
        # 2. Classifier:
        _fc_layers = [torch.nn.Linear(_n_inputs, 256),
                      torch.nn.Linear(256, self.n_outputs)]
        self.model.fc = torch.nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits if self.n_outputs == 1 else F.cross_entropy

    def __step(self, batch):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        loss = self.loss(y_logits, y if self.n_outputs > 1 else y.view((-1, 1)).type_as(x))
        y_hat = torch.argmax(y_logits, dim=1) if self.n_outputs > 1 else (y_logits >= 0.0).squeeze(1).long()
        num_correct = torch.eq(y_hat, y.view(-1)).sum()

        return loss, num_correct

    def __metrics_per_batch(self, batch):
        # 1. Forward pass:
        x, y_true = batch
        logits = self.forward(x)

        # 2. Compute loss & performance metrics:
        # class prediction: if binary (num_outputs == 1) then class label is 0 if logit < 0 else it's 1
        # if multiclass then simply run argmax to find the index of the most confident class
        y_hat = torch.argmax(logits, dim=1) if self.n_outputs > 1 else (logits > 0.0).squeeze(1).long()
        loss = self.loss(logits, y_true if self.n_outputs > 1 else y_true.view((-1, 1)).type_as(x))
        num_correct = torch.eq(y_hat, y_true.view(-1)).sum()
        acc = num_correct.float() / self.batch_size
        prec = plm.precision(y_hat, y_true, num_classes=self.n_classes)
        rec = plm.recall(y_hat, y_true, num_classes=self.n_classes)
        f1 = plm.f1_score(y_hat, y_true, num_classes=self.n_classes)
        conf_matrix = plm.confusion_matrix(y_hat.long(), y_true.long())

        return (
            y_true,
            y_hat,
            logits,
            loss,
            num_correct,
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
            title=f"{stage} confusion matrix for fold #{self.fold_number}",
            normalize=False
        )
        normalized_confusion_matrix_figure = _plot_confusion_matrix(
            cm=conf_matrix,
            target_names=self.classes,
            title=f"{stage} confusion matrix for fold #{self.fold_number}",
            normalize=True
        )

        self.logger.experiment.add_figure(tag=f"{stage}/confusion_matrix",
                                          figure=confusion_matrix_figure,
                                          global_step=self.current_epoch)
        self.logger.experiment.add_figure(tag=f"{stage}/normalized_confusion_matrix",
                                          figure=normalized_confusion_matrix_figure,
                                          global_step=self.current_epoch)

    def __setup(self):
        df = pd.read_csv(self.train_csv)
        df = k_fold(df, self.folds, self.seed)
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
                                        binary=self.is_binary,
                                        root_dir=self.root_data_path,
                                        train=False,
                                        resize=resize_to,
                                        augmentations=valid_aug)

        df_train = df[df["kfold"] != self.fold_number].reset_index(drop=True)
        df_valid = df[df["kfold"] == self.fold_number].reset_index(drop=True)

        train_image_names = df_train.image
        val_image_names = df_valid.image

        train_targets = df_train.label
        val_targets = df_valid.label

        train_dataset = CoralFragDataset(image_names=train_image_names,
                                         targets=train_targets,
                                         binary=self.is_binary,
                                         root_dir=self.root_data_path,
                                         train=True,
                                         resize=resize_to,
                                         augmentations=train_aug)

        val_dataset = CoralFragDataset(image_names=val_image_names,
                                       targets=val_targets,
                                       binary=self.is_binary,
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

    def forward(self, x):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        x = self.model.forward(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def model_splits(self):
        groups = [torch.nn.Sequential(self.model.conv1, self.model.bn1)]
        groups += [layer for name, layer in self.model.named_children() if name.startswith("layer")]
        groups += [self.model.fc]  # Considering we already switched the head

        return groups

    def configure_optimizers(self):
        # passed lr does not matter, because scheduler will overtake
        param_groups = self.get_optimizer_param_groups(0)
        opt = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        # return a dummy lr_scheduler, so LearningRateLogger doesn't complain
        sched = OneCycleLR(opt, 0, 9)
        return [opt], [sched]

    def on_epoch_start(self):
        if self.current_epoch in self.milestones.keys():
            milestone_config = self.milestones[self.current_epoch]
            # Unfreeze all layers, we can also use `unfreeze`, but `freeze_to` has the
            # additional property of only considering parameters returned by `model_splits`
            self.freeze_to(milestone_config['freeze_to'])

            # Create new scheduler
            total_steps = len(self.train_dataloader()) * milestone_config['duration']
            lrs = self.get_lrs(milestone_config['lrs'])
            opt = self.trainer.optimizers[0]
            sched = {
                'scheduler': OneCycleLR(
                    opt,
                    lrs,
                    total_steps,
                    pct_start=milestone_config['pct_start'],
                    div_factor=self.div_factor,
                    final_div_factor=self.final_div_factor,
                    base_momentum=self.base_momentum,
                    max_momentum=self.max_momentum
                ),
                'interval': 'step'
            }
            scheds = self.trainer.configure_schedulers([sched])
            # Replace scheduler and update lr logger
            self.trainer.lr_schedulers = scheds
            # lr_logger.on_train_start(self.trainer, self)

    def training_step(self, batch, batch_idx):
        train_loss, num_correct = self.__step(batch)
        train_acc = num_correct.float() / self.batch_size
        log = {
            'Train/loss': train_loss,
            'Train/acc': train_acc,
            'Train/num_correct': num_correct
        }

        output = OrderedDict(
            {
                'loss': train_loss,
                'train_acc': train_acc,
                'num_correct': num_correct,
                'log': log
            }
        )

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss'] for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct'] for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)

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
        y_true, y_hat, logits, loss, num_correct, acc, prec, rec, f1, conf_matrix = self.__metrics_per_batch(batch)

        return {
            'loss': loss,
            'num_correct': num_correct,
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
        y_true, y_hat, loss, num_correct, acc, prec, rec, f1, conf_matrix = self.__metrics_per_epoch(outputs)
        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Validation")

        print(f"\nValidation Epoch Loss: {loss:.4f}, validation epoch accuracy: {acc:.4f}")

        log = {
            'Validation/loss/epoch': loss,
            'Validation/num_correct/epoch': num_correct,
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
        y_true, y_hat, logits, loss, num_correct, acc, prec, rec, f1, conf_matrix = self.__metrics_per_batch(batch)

        output = {
            'loss': loss,
            'num_correct': num_correct,
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
        y_true, y_hat, loss, num_correct, acc, prec, rec, f1, conf_matrix = self.__metrics_per_epoch(outputs)

        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Test")

        print(f"\nTest Epoch Loss: {loss:.4f}, test epoch accuracy: {acc:.4f}")
        print(f"Confusion matrix: \n{conf_matrix.int()}")

        log = {
            'Test/epoch_loss': loss,
            'Test/epoch_acc': acc,
            'Test/epoch_precision': prec,
            'Test/epoch_recall': rec,
            'Test/epoch_f1_score': f1,
        }

        self.logger.experiment.add_graph(
            self,
            torch.rand(self.batch_size, 3, self.input_size[0], self.input_size[1]).to(self.device)
        )

        self.logger.experiment.flush()

        return {
            'log': log
        }

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader

    @staticmethod
    def __metrics_per_epoch(outputs):
        num_correct = torch.stack([output['num_correct'] for output in outputs]).sum()
        loss_mean = torch.stack([output['loss'] for output in outputs]).mean()
        acc_mean = torch.stack([output['acc'] for output in outputs]).mean()
        prec_mean = torch.stack([output['precision'] for output in outputs]).mean()
        rec_mean = torch.stack([output['recall'] for output in outputs]).mean()
        f1_mean = torch.stack([output['f1_score'] for output in outputs]).mean()
        y_true = torch.cat([output['y_true'] for output in outputs], dim=-1)
        y_hat = torch.cat([output['y_hat'] for output in outputs], dim=-1)

        confusion_matrix = plm.confusion_matrix(y_hat, y_true)

        return (
            y_true,
            y_hat,
            loss_mean,
            num_correct,
            acc_mean,
            prec_mean,
            rec_mean,
            f1_mean,
            confusion_matrix
        )
