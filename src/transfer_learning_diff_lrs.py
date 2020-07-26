import numbers
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pl_examples import LightningTemplateModel
from pytorch_lightning import _logger as log
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Copied from pl_examples (with small changes)
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
FREEZE_LR = (0, 0, 0, 0, 0, 1e-3)
UNFREEZE_LR = (1e-7, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4)
FREEZE_EPOCHS = 2
UNFREEZE_EPOCHS = 4


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


class ParametersSplitsModuleMixin(pl.LightningModule, ABC):
    printed = False

    # TODO: Implement default splitter (just return a list of layers)
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


class MyModel(ParametersSplitsModuleMixin):
    def __init__(self, hparams):
        super(MyModel, self).__init__()
        import torchvision.models as models

        self.model = models.resnet18(pretrained=True)
        _n_inputs = self.model.fc.in_features
        _n_outputs = 10
        # 2. Classifier:
        _fc_layers = [torch.nn.Linear(_n_inputs, 256),
                      torch.nn.Linear(256, 10)]
        self.model.fc = torch.nn.Sequential(*_fc_layers)

        self.hparams = hparams
        self.batch_size = hparams["batch_size"]
        self.data_root = os.path.join("./", 'cifar10')

    def model_splits(self):
        groups = [nn.Sequential(self.model.conv1, self.model.bn1)]
        groups += [layer for name, layer in self.model.named_children() if name.startswith("layer")]
        groups += [self.model.fc]  # Considering we already switched the head

        return groups

    def configure_optimizers(self):
        # passed lr does not matter, because scheduler will overtake
        param_groups = self.get_optimizer_param_groups(0)
        opt = torch.optim.Adam(param_groups)
        # return a dummy lr_scheduler, so LearningRateLogger doesn't complain
        sched = OneCycleLR(opt, 0, 9)
        return [opt], [sched]

    def on_epoch_start(self):
        if self.current_epoch == 1:
            # Freeze all but last layer (imagine this is the head)
            self.freeze_to(-1)
            # Create new scheduler
            total_steps = len(model.train_dataloader()) * (FREEZE_EPOCHS - 1)
            lrs = self.get_lrs(FREEZE_LR)
            opt = self.trainer.optimizers[0]
            sched = {'scheduler': OneCycleLR(opt, lrs, total_steps, pct_start=.9), 'interval': 'step'}
            scheds = self.trainer.configure_schedulers([sched])
            # Replace scheduler and update lr logger
            self.trainer.lr_schedulers = scheds
            lr_logger.on_train_start(self.trainer, self)

        if self.current_epoch == FREEZE_EPOCHS:
            # Unfreeze all layers, we can also use `unfreeze`, but `freeze_to` has the
            # additional property of only considering parameters returned by `model_splits`
            self.freeze_to(0)
            # Create new scheduler
            total_steps = len(model.train_dataloader()) * (UNFREEZE_EPOCHS + 1)
            lrs = self.get_lrs(UNFREEZE_LR)
            opt = self.trainer.optimizers[0]
            sched = {'scheduler': OneCycleLR(opt, lrs, total_steps, pct_start=.2), 'interval': 'step'}
            scheds = self.trainer.configure_schedulers([sched])
            # Replace scheduler and update lr logger
            self.trainer.lr_schedulers = scheds
            lr_logger.on_train_start(self.trainer, self)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.model.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def prepare_data(self):
        CIFAR10(self.data_root, train=True, download=True, transform=transforms.ToTensor())
        CIFAR10(self.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.cifar10_train = CIFAR10(self.data_root, train=True, download=False, transform=transform)
        self.cifar10_test = CIFAR10(self.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'cifar10'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser


# HACK: Have to define `lr_logger` globally because we're calling `lr_logger.on_train_start` inside `model.on_epoch_start`
lr_logger = pl.callbacks.LearningRateLogger()
model = MyModel({"batch_size": 64})

max_epochs = FREEZE_EPOCHS + UNFREEZE_EPOCHS
trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, callbacks=[lr_logger])
trainer.fit(model)
