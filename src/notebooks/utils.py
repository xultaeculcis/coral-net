import pandas as pd 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import skimage.io as ioimage_stack
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import os
import glob
import uuid
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_dir = "../datasets/final_dataset_merged"
CLASS_NAMES = np.array(["acan", "acro", "chalice", "hammer", "monti", "torch", "zoa"])

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224


def to_jpg_from(dateset_dir, classes, from_extension="jfif"):
    for cl in classes:
        for file in glob.glob(f"{dataset_dir}/{cl}/*/*.{from_extension}"):
            im = Image.open(file)
            rgb_im = im.convert('RGB')
            rgb_im.save(file.replace(from_extension, "jpg"), quality=100)


def rename_files(dateset_dir, classes, extension="jpg", with_delete=False):
    for cl in classes:
        for i, file in tqdm(enumerate(glob.glob(f"{dateset_dir}/{cl}/*.{extension}"))):
            num = str(i).rjust(6, "0")
            os.rename(file, f"{dateset_dir}/{cl}/{cl}_{num}.{extension}")
            if with_delete:
                os.remove(file)
                
    
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(15,15))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
        
        
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds