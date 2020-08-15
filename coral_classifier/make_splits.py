import argparse
import glob
import os
from typing import List

import pandas as pd
from sklearn import model_selection


def build_img_dict(dirs: List[str], classes: List[str]):
    print("Building image dictionary")
    image_dict = {}
    for cls, directory in zip(classes, dirs):
        cls_images = glob.glob(directory + "/*.jpg")
        image_dict[cls] = cls_images
        print(cls, len(cls_images))
    return image_dict


def build_data_frame(image_dict: dict, classes: List[str]):
    print("Building Data Frame with images")
    output_df = pd.DataFrame()
    for i, cls in enumerate(classes):
        df_lst = []
        for img_name in image_dict[cls]:
            df_data = {
                "image": img_name,
                "label": i,
                "text_label": cls
            }
            df_lst.append(df_data)

        tmp_df = pd.DataFrame(df_lst)
        output_df = output_df.append(tmp_df)

    return output_df.reset_index().drop(labels="index", axis=1)


def normalize_image_names_in_df(df: pd.DataFrame):
    print("Normalizing image names in the Data Frame")
    # for every row, use image name instead of image path
    df["image"] = df["image"].map(lambda image: os.path.join(image.split("/")[-2], image.split("/")[-1]))

    return df


def k_fold(df: pd.DataFrame, k: int = 5, seed: int = 42):
    df["kfold"] = -1
    new_frame = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = new_frame.label.values
    kf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for f, (t_, v_) in enumerate(kf.split(X=new_frame, y=y)):
        new_frame.loc[v_, 'kfold'] = f

    return new_frame


def main(args: argparse.Namespace) -> None:
    d = build_img_dict(args.dirs, args.classes)
    X_train = build_data_frame(d, args.classes)

    if args.include_test:
        print("Splitting the data into train and test sets")
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X_train,
            X_train.label.values,
            test_size=args.test_size,
            random_state=args.seed
        )
        X_test = normalize_image_names_in_df(X_test)
        X_test.to_csv(os.path.join(args.output_data_dir, "test.csv"), index=False)

    X_train = k_fold(X_train, args.k, args.seed)
    X_train = normalize_image_names_in_df(X_train)
    X_train.to_csv(os.path.join(args.output_data_dir, "train.csv"), index=False)


def add_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument('--data-dir',
                        default='../datasets/train',
                        type=str,
                        help='Path to the image folders',
                        dest='data_dir')
    parser.add_argument('--output-data-dir',
                        default='../datasets',
                        type=str,
                        help='Where to put train/test CSVs.',
                        dest='output_data_dir')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='Random seed for reproducibility. 42 by default.',
                        dest='seed')
    parser.add_argument('--k',
                        default=10,
                        type=int,
                        help='Number of splits for the k-fold cross validation',
                        dest='k')
    parser.add_argument('--test-size',
                        default=.1,
                        type=float,
                        help='Percent of the entire dataset that will be used to build the test set. '
                             'Only used if include-test flag is set tot True. 10% by default',
                        dest='test_size')
    parser.add_argument('--include-test',
                        default=False,
                        type=bool,
                        help='Indicates if the test set should also be created. False by default.',
                        dest='include_test')
    return parser


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--root-data-path',
                               metavar='DIR',
                               type=str,
                               default="../../datasets",
                               help='Root directory where to download the data',
                               dest='root_data_path')
    return add_args(parent_parser).parse_args()


if __name__ == "__main__":
    arguments = get_args()
    arguments.classes = sorted(os.listdir(arguments.data_dir))
    arguments.dirs = sorted([os.path.join(arguments.data_dir, c) for c in arguments.classes])
    main(arguments)
    print("Done")
