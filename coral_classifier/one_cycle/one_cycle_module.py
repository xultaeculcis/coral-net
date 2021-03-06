import argparse
import os
from collections import OrderedDict

import albumentations
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models

from coral_classifier.dataset import CoralFragDataset
from coral_classifier.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from coral_classifier.make_splits import k_fold
from coral_classifier.utils import _plot_confusion_matrix

# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


#  --- Pytorch-lightning module ---
class OneCycleModule(pl.LightningModule):
    """Transfer Learning with pre-trained Model.
    Args:
        hparams: Model hyperparameters
    """

    def __init__(self, hparams) -> None:
        super().__init__()

        self.supported_architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                        'resnext50_32x4d', 'resnext101_32x8d',
                                        'wide_resnet50_2', 'wide_resnet101_2']

        if type(hparams) is dict:
            a = argparse.ArgumentParser()
            for key in hparams.keys():
                a.add_argument(
                    f'--{key}',
                    default=hparams[key]
                )
            args = a.parse_args()
            hparams = args

        self.__validate_args(hparams.backbone)
        self.hparams = hparams

        self.backbone = hparams.backbone
        self.train_bn = hparams.train_bn
        self.batch_size = hparams.batch_size
        self.root_data_path = hparams.root_data_path
        self.num_workers = hparams.num_workers
        self.folds = hparams.folds
        self.fold_number = hparams.fold
        self.seed = hparams.seed
        self.epochs = hparams.epochs
        self.weight_decay = hparams.weight_decay
        self.train_csv = hparams.train_csv
        self.test_csv = hparams.test_csv
        self.max_lr = hparams.max_lr
        self.pct_start = hparams.pct_start
        self.div_factor = hparams.div_factor
        self.final_div_factor = hparams.final_div_factor
        self.base_momentum = hparams.base_momentum
        self.max_momentum = hparams.max_momentum
        self.input_size = (224, 224)

        self.__setup()
        self.__build_model()

        self.train_loader = self.__dataloader(stage='train')
        self.val_loader = self.__dataloader(stage='validation')
        self.test_loader = self.__dataloader(stage='test')
        self.total_steps = len(self.train_dataloader()) * self.epochs

    def __validate_args(self, backbone):
        if backbone not in self.supported_architectures:
            raise Exception(f"The '{backbone}' is currently not supported as a backbone. "
                            f"Supported architectures are: {', '.join(self.supported_architectures)}")

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)

        # 2. Classifier
        _n_inputs = backbone.fc.in_features
        _fc_layers = [torch.nn.Linear(_n_inputs, self.n_outputs)]
        self.fc = torch.nn.Sequential(*_fc_layers)

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

        acc1, acc5 = self.__accuracy(y_logits, y, topk=(1, 5))

        return loss, num_correct, acc1, acc5

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
        acc1, acc5 = self.__accuracy(logits, y_true, topk=(1, 5))

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
            conf_matrix,
            acc1,
            acc5
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
        resize_to = [224, 224]

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

        test_dataset = CoralFragDataset(df=df_test,
                                        data_dir=os.path.join(self.root_data_path, 'labeled'),
                                        train=False,
                                        resize=resize_to,
                                        augmentations=valid_aug)

        df_train = df[df["kfold"] != self.fold_number].reset_index(drop=True)
        df_valid = df[df["kfold"] == self.fold_number].reset_index(drop=True)

        train_dataset = CoralFragDataset(df=df_train,
                                         data_dir=os.path.join(self.root_data_path, 'labeled'),
                                         train=True,
                                         resize=resize_to,
                                         augmentations=train_aug)

        val_dataset = CoralFragDataset(df=df_valid,
                                       data_dir=os.path.join(self.root_data_path, 'labeled'),
                                       train=True,
                                       resize=resize_to,
                                       augmentations=valid_aug)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.classes = sorted(df_train['label'].unique().tolist())
        self.n_classes = len(self.classes)
        self.n_outputs = self.n_classes
        self.is_binary = self.n_outputs <= 2

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
        opt = AdamW(self.parameters(), weight_decay=self.weight_decay)
        # return a dummy lr_scheduler, so LearningRateLogger doesn't complain
        scheduler = OneCycleLR(opt,
                               max_lr=self.max_lr,
                               total_steps=self.total_steps,
                               pct_start=self.pct_start,
                               div_factor=self.div_factor,
                               final_div_factor=self.final_div_factor,
                               base_momentum=self.base_momentum,
                               max_momentum=self.max_momentum)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        train_loss, num_correct, train_acc1, train_acc5 = self.__step(batch)
        train_acc = num_correct.float() / self.batch_size
        log = {
            'Train/loss': train_loss,
            'Train/acc': train_acc,
            'Train/acc1': train_acc1,
            'Train/acc5': train_acc5,
            'Train/num_correct': num_correct
        }

        output = OrderedDict(
            {
                'loss': train_loss,
                'train_acc': train_acc,
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'num_correct': num_correct,
                'log': log
            }
        )

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss'] for output in outputs]).mean()
        train_acc1_mean = torch.stack([output['acc1'] for output in outputs]).mean()
        train_acc5_mean = torch.stack([output['acc5'] for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct'] for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)

        print(f"\nTraining Epoch Loss: {train_loss_mean:.4f}, training epoch accuracy: {train_acc_mean:.4f}")

        log = {
            'Train/epoch/acc': train_acc_mean,
            'Train/epoch/acc1': train_acc1_mean,
            'Train/epoch/acc5': train_acc5_mean,
            'Train/epoch/loss': train_loss_mean,
            'step': self.current_epoch
        }

        return {
            'loss': train_loss_mean,
            'acc': train_acc_mean,
            'log': log
        }

    def validation_step(self, batch, batch_idx):
        y_true, y_hat, logits, loss, num_correct, acc, prec, rec, f1, conf_matrix, acc1, acc5 = self.__metrics_per_batch(
            batch)

        return {
            'loss': loss,
            'num_correct': num_correct,
            'acc': acc,
            'acc1': acc1,
            'acc5': acc5,
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
        y_true, y_hat, loss, num_correct, acc, prec, rec, f1, conf_matrix, acc1, acc5 = self.__metrics_per_epoch(
            outputs)
        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Validation")

        print(f"\nValidation Epoch Loss: {loss:.4f}, validation epoch accuracy: {acc:.4f}")

        log = {
            'Validation/loss/epoch': loss,
            'Validation/num_correct/epoch': num_correct,
            'Validation/acc/epoch': acc,
            'Validation/acc1/epoch': acc1,
            'Validation/acc5/epoch': acc5,
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
        y_true, y_hat, logits, loss, num_correct, acc, prec, rec, f1, conf_matrix, acc1, acc5 = self.__metrics_per_batch(
             batch)

        output = {
            'loss': loss,
            'num_correct': num_correct,
            'acc': acc,
            'acc1': acc1,
            'acc5': acc5,
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
        y_true, y_hat, loss, num_correct, acc, prec, rec, f1, conf_matrix, acc1, acc5 = self.__metrics_per_epoch(
            outputs)

        self.__log_confusion_matrices(conf_matrix.cpu().numpy().astype('int'), "Test")

        print(f"\nTest Epoch Loss: {loss:.4f}, test epoch accuracy: {acc:.4f}")

        log = {
            'Test/epoch_loss': loss,
            'Test/epoch_acc': acc,
            'Test/epoch_acc1': acc1,
            'Test/epoch_acc5': acc5,
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
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

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
        acc1_mean = torch.stack([output['acc1'] for output in outputs]).mean()
        acc5_mean = torch.stack([output['acc5'] for output in outputs]).mean()

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
            confusion_matrix,
            acc1_mean,
            acc5_mean
        )
