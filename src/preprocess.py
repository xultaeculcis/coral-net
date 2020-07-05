import glob
import os
import shutil

import pandas as pd
from sklearn import model_selection
from tqdm import tqdm

data_dir = "../datasets/original"
datasets_dir = "../datasets"
output_data_dir = "../datasets"
CLASS_NAMES = os.listdir(data_dir)
RANDOM_SEED = 42
dirs = [os.path.join(data_dir, c) for c in CLASS_NAMES]
n_splits = 5
positive_class = "sps acropora"


def build_img_dict():
    print("Building image dictionary")
    image_dict = {}
    for cls, directory in zip(CLASS_NAMES, dirs):
        cls_images = glob.glob(directory + "/*.jpg")
        image_dict[cls] = cls_images
        print(cls, len(cls_images))
    return image_dict


def build_data_frame(image_dict, binary=False):
    print("Building Data Frame with images")
    output_df = pd.DataFrame()
    for i, cls in enumerate(CLASS_NAMES):
        df_lst = []
        for img_name in image_dict[cls]:
            if binary:
                df_data = {
                    "image": img_name,
                    "label": 1 if cls == positive_class else 0,
                    "text_label": positive_class if cls == positive_class else "other"
                }
            else:
                df_data = {
                    "image": img_name,
                    "label": i,
                    "text_label": cls
                }

            df_lst.append(df_data)

        tmp_df = pd.DataFrame(df_lst)
        output_df = output_df.append(tmp_df)

    return output_df.reset_index().drop(labels="index", axis=1)


def create_folds(frame_path):
    print("Creating folds")
    frame = pd.read_csv(frame_path)

    frame["kfold"] = -1
    new_frame = frame.sample(frac=1).reset_index(drop=True)
    y = new_frame.label.values
    kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for f, (t_, v_) in enumerate(kf.split(X=new_frame, y=y)):
        print(f, t_, v_)
        new_frame.loc[v_, 'kfold'] = f

    new_frame.to_csv(frame_path, index=False)

    return new_frame


def move_data(train_df: pd.DataFrame, test_df):
    print("Moving data to train and test folders")
    # make sure the dirs exist
    train_dir = os.path.join(output_data_dir, "train")
    test_dir = os.path.join(output_data_dir, "test")
    try:
        os.mkdir(train_dir)
        os.mkdir(test_dir)
    except Exception as e:
        print(e)
        # nothing to do, if dirs already exist
        return

    for i, row in tqdm(train_df.iterrows()):
        image_name = row.image.split("/")[-1]
        dest_location = os.path.join(train_dir, image_name)
        shutil.copyfile(row.image, dest_location)

    for i, row in tqdm(test_df.iterrows()):
        image_name = row.image.split("/")[-1]
        dest_location = os.path.join(test_dir, image_name)
        shutil.copyfile(row.image, dest_location)


def normalize_image_names_in_df(df):
    print("Normalizing image names in the Data Frame")
    # for every row, use image name instead of image path
    df["image"] = df["image"].map(lambda image: image.split("/")[-1])

    return df


def main(binary=False):
    d = build_img_dict()

    df = build_data_frame(d, binary)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df,
        df.label.values,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    X_train.to_csv(os.path.join(datasets_dir, "train.csv"), index=False)

    df = create_folds(os.path.join(datasets_dir, "train.csv"))

    move_data(X_train, X_test)

    X_train = normalize_image_names_in_df(df)
    X_test = normalize_image_names_in_df(X_test)

    X_train.to_csv(os.path.join(datasets_dir, "train.csv"), index=False)
    X_test.to_csv(os.path.join(datasets_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main(binary=False)
