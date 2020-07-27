"""Computer vision example on Transfer Learning.
This computer vision example illustrates how one could fine-tune a pre-trained
network using pytorch-lightning.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import argparse
import os
from pprint import pprint

import pytorch_lightning as pl
import torch
from PIL import ImageFile
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

from src.transfer_learning.transfer_learning_module import TransferLearningModule
from src.utils import print_system_info

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def add_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument('--backbone',
                        default='resnext50_32x4d',
                        # default='resnet18',
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
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-3,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lr-scheduler-gamma',
                        default=1e-1,
                        type=float,
                        metavar='LRG',
                        help='Factor by which the learning rate is reduced at each milestone',
                        dest='lr_scheduler_gamma')
    parser.add_argument('--train-bn',
                        default=True,
                        type=bool,
                        metavar='TB',
                        help='Whether the BatchNorm layers should be trainable',
                        dest='train_bn')
    parser.add_argument('--epochs',
                        default=6,
                        type=int,
                        dest='epochs',
                        help='For how many epochs the model should be trained')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        dest='seed',
                        help='The random seed for the reproducibility purposes')
    parser.add_argument('--train-csv',
                        default="../../datasets/train.csv",
                        type=str,
                        help='Path to train csv file',
                        dest='train_csv')
    parser.add_argument('--test-csv',
                        default="../../datasets/test.csv",
                        type=str,
                        help='Path to test csv file',
                        dest='test_csv')
    parser.add_argument('--save-model-path',
                        default="../../model-weights",
                        type=str,
                        help='Where to save the best model checkpoints',
                        dest='save_model_path')
    parser.add_argument('--save-top-k',
                        default=1,
                        type=int,
                        help='How many best k models to save',
                        dest='save_top_k')
    parser.add_argument('--precision',
                        default=16,
                        type=int,
                        help='Training precision - 16 bit by default',
                        dest='precision')
    parser.add_argument('--milestones',
                        default=[2, 4],
                        type=list,
                        metavar='M',
                        help='List of two epochs milestones')
    return parser


def main(arguments: argparse.Namespace) -> None:
    """Train the model.
    Args:
        arguments: Model hyper-parameters
    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

    print_system_info()
    print("Using following configuration: ")
    pprint(vars(arguments))

    for fold in range(1):
        print(f"Fold {fold}: Training is starting...")
        arguments.fold = fold
        model = TransferLearningModule(arguments)
        logger = TensorBoardLogger("../logs", name=f"{arguments.backbone}-fold-{fold}")

        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='max'
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(arguments.save_model_path,
                                  f"checkpoint-{arguments.backbone}-fold-{fold}" + "-{epoch:02d}-{val_f1:.2f}"),
            save_top_k=arguments.save_top_k,
            monitor="val_f1",
            mode="max",
            verbose=True
        )
        trainer = pl.Trainer(
            weights_summary=None,
            num_sanity_val_steps=0,
            gpus=arguments.gpus,
            min_epochs=arguments.epochs,
            max_epochs=arguments.epochs,
            logger=logger,
            deterministic=True,
            benchmark=False,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            callbacks=[lr_logger],
            precision=arguments.precision,
            # fast_dev_run=True
        )

        trainer.fit(model)

        logger.log_hyperparams(arguments, {"hparams/val_f1": checkpoint_callback.best_model_score.item()})
        logger.save()

        print("-" * 80)
        print(f"Testing the model on fold: {fold}")
        trainer.test(model)


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--root-data-path',
                               metavar='DIR',
                               type=str,
                               default="../../datasets",
                               help='Root directory where to download the data',
                               dest='root_data_path')
    parser = add_model_specific_args(parent_parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(seed=args.seed)
    torch.manual_seed(args.seed)
    lr_logger = LearningRateLogger()
    main(args)
