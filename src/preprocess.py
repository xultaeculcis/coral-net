import glob
import os
import shutil

import pandas as pd
from sklearn import model_selection
from tqdm import tqdm

data_dir = "../datasets/original"
CLASS_NAMES = os.listdir(data_dir)
dirs = [os.path.join(data_dir, c) for c in CLASS_NAMES]


def build_img_dict():
    image_dict = {}
    for cls, directory in zip(CLASS_NAMES, dirs):
        cls_images = glob.glob(directory + "/*.jpg")
        image_dict[cls] = cls_images
        print(cls, len(cls_images))
    return image_dict


def build_data_frame(image_dict):
    output_df = pd.DataFrame()
    for i, cls in enumerate(CLASS_NAMES):
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


def create_folds(df):
    # create folds
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv(os.path.join(data_dir, "train.csv"), index=False)


def train_test_split(X, y, test_fraction=.20):
    return model_selection.train_test_split(X, y, test_size=test_fraction)


def move_data(train_df : pd.DataFrame, test_df):
    for i, row in tqdm(train_df.iterrows()):
        shutil.copyfile(row.image, )


if __name__ == "__main__":
    d = build_img_dict()
    df = build_data_frame(d)
    X_train, X_test, y_train, y_test = train_test_split(df, df.label.values)
    X_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    df = X_train
    create_folds(X_train)
    move_data()