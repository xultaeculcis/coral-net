import multiprocessing
from multiprocessing import Pool
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import skimage.io as io
import time
from datetime import datetime
import pandas as pd
import os

data_dir = "E:/Datasets/coral-classifier/final_dataset_merged"
augment_dir = "E:/Datasets/coral-classifier/augmented_dataset_2"
classes = [
    "acan",
    "acro",
    "chalice",
    "hammer",
    "monti",
    "torch",
    "zoa"
]
num_images_to_generate = 20
network_input_size = 224

dirs = [data_dir + "/" + c for c in classes]

seq = iaa.Sequential([
    iaa.Resize((network_input_size, network_input_size)),
    iaa.Fliplr(p=0.5),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.AddToBrightness((-30, 30)),
    iaa.LinearContrast((0.7, 1.4))
])

def make_dirs():
    try:
        os.mkdir(augment_dir)
        for cl in classes:
            os.mkdir(augment_dir + "/" + cl)
    except Exception:
        pass


def augment_images_sync(dir):
    print(f"Augmenting images in '{dir}'")

    image_names = np.array(io.collection.glob(dir + "/*.jpg"))[50:100]
    results = []
    for image_name in image_names:
        results.append(augment_single(image_name))
    return len(image_names), sum(results)


def augment_single_with_metrics(image_path):
    start_processing_time = datetime.utcnow()
    image = io.imread(image_path)
    splitted = image_path.split("/")
    file_name = splitted[-1]
    class_name = splitted[-2]

    aug_times = []
    insert_times = []
    for i in range(num_images_to_generate):
        start = datetime.utcnow()
        augmented_img = seq(image=image)
        padded = str(i).rjust(3, "0")
        aug_name_suffix = f"_aug{padded}.png"
        aug_name = f"{augment_dir}/{file_name.replace('.jpg', aug_name_suffix)}"
        aug_times.append((datetime.utcnow() - start).total_seconds())

        start = datetime.utcnow()
        io.imsave(aug_name, augmented_img)
        insert_times.append((datetime.utcnow() - start).total_seconds())

    avg_aug_time = sum(aug_times) / num_images_to_generate
    avg_insert_time = sum(insert_times) / num_images_to_generate
    total_time = (datetime.utcnow() - start_processing_time).total_seconds()
    return [file_name, total_time, avg_insert_time, avg_aug_time]


def augment_images_with_metrics(dir, pool):
    print(f"Augmenting images in '{dir}'")

    image_names = np.array(io.collection.glob(dir + "/*.jpg"))
    result = pool.map_async(augment_single_with_metrics, image_names)
    return len(image_names), pd.DataFrame(result.get(), columns=["image_name", "total_time", "avg_insert_time", "avg_aug_time"])


def augment_single(image_path):
    image = io.imread(image_path)
    splitted = image_path.split("/")
    file_name = splitted[-1]
    class_name = splitted[-2]

    for i in range(num_images_to_generate):
        augmented_img = seq(image=image)
        padded = str(i).rjust(3, "0")
        aug_name_suffix = f"_aug{padded}.jpg"
        aug_name = f"{augment_dir}/{file_name.replace('.jpg', aug_name_suffix)}"

        io.imsave(aug_name, augmented_img)

    return file_name


def augment_images(dir, pool):
    print(f"Augmenting images in '{dir}'")

    image_names = np.array(io.collection.glob(dir + "/*.jpg"))
    result = pool.map_async(augment_single, image_names)
    return len(image_names), result.get()


if __name__ == "__main__":
    pool = Pool(processes=64)
    make_dirs()
    for path in dirs:
        start = datetime.utcnow()
        image_count, results = augment_images(path, pool)
        end = datetime.utcnow()
        print(
            f"Augmented {image_count} images. Execution time: {end - start}.")
        print("*" * 150)

    # with metrics
    # for path in dirs:
    #     start = datetime.utcnow()
    #     image_count, results = augment_images_with_metrics(path, pool)
    #     end = datetime.utcnow()
    #     print(f"Augmented {image_count} images. Execution time: {end - start}.")
    #     print(results.describe())
    #     print("*" * 150)
    #
    # sync
    # for path in dirs:
    #     start = datetime.utcnow()
    #     image_count, results = augment_images_sync(path)
    #     end = datetime.utcnow()
    #     print(f"Augmented {image_count} images. Execution time: {end - start}.")
    #     print("*" * 150)
