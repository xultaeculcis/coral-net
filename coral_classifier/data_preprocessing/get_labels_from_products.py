from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import os
import glob
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

data_dir = "../../datasets/scrape/"


def as_shopify(df: pd.DataFrame):
    logging.info("Data comes from Shopify site")


def as_woocommerce(df: pd.DataFrame):
    logging.info("Data comes from Woocommerce site")


def assign_classes(df: pd.DataFrame) -> None:
    logging.info(f"File contains {len(df)} products")

    if "Collection" in df.columns:
        as_shopify(df)
    else:
        as_woocommerce(df)


if __name__ == "__main__":
    logging.info("Running class assignment script")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    logging.info(f"Found {len(csv_files)} files to process: {csv_files}")

    for csv in csv_files:
        # csv = "../../datasets/scrape/www.coral.zone.csv"

        logging.info(f"Loading file {csv}")

        frame = pd.read_csv(csv)
        assign_classes(frame)
        # break
