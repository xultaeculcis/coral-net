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

from src.dataset import CoralFragDataset
from src.imbalanced_dataset_sampler import ImbalancedDatasetSampler
from src.utils import _plot_confusion_matrix, k_fold

# imagenet normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


#  --- Pytorch-lightning module ---
class TransferLearningModel(pl.LightningModule):
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
                 batch_size: int = 256,
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
                 seed=42,
                 max_lr=1e-4,
                 pct_start=.3,
                 div_factor=5,
                 final_div_factor=1e2,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 **kwargs) -> None:
        super().__init__()

        self.supported_architectures = ['googlenet',
                                        'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                                        'wide_resnet50_2', 'wide_resnet101_2']

        self.__validate_args(backbone, n_classes, n_outputs)

        self.backbone = backbone
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_outputs = n_outputs
        self.is_binary = n_outputs <= 2
        self.root_data_path = root_data_path
        self.num_workers = num_workers
        self.folds = folds
        self.fold_number = fold
        self.seed = seed
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
        self.train_loader = self.__dataloader(stage='train')
        self.val_loader = self.__dataloader(stage='validation')
        self.test_loader = self.__dataloader(stage='test')
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.total_steps = len(self.train_dataloader()) * (self.unfreeze_epochs + self.freeze_epochs)

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
                            f"Supported architectures are: {self.supported_architectures}")

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

        print(f"\nTest Epoch Loss: {loss:.4f}, test epoch accuracy: {acc:.4f}")
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
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader

    @staticmethod
    def __metrics_per_epoch(outputs):
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
            acc_mean,
            prec_mean,
            rec_mean,
            f1_mean,
            confusion_matrix
        )
