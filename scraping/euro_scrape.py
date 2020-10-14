from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
import ssl
from dask.distributed import Client
import dask.bag as db
import re
import numpy as np
import pandas as pd
import logging
from bs4 import BeautifulSoup
import requests
import time
import pprint

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

DRIVER_PATH = '/usr/bin/chromedriver'
SCROLL_PAUSE_TIME = 0.5


def __scroll_down_page(wd, speed=20):
    current_scroll_position, new_height = 0, 1
    while current_scroll_position <= new_height:
        current_scroll_position += speed
        wd.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
        new_height = wd.execute_script("return document.body.scrollHeight")


def get_product_links():
    logging.info(f"Starting scraping of links for: {base_url}")

    # get all product links from all pages
    logging.info(f"Getting links on infinite scroll page")

    current_page_products = []

    headers = {
        "User-Agent": "PostmanRuntime/7.26.5",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    for i in range(1, 6):
        content = requests.get(f"{base_url}/shop/page/{i}", headers=headers).content

        # get soup
        soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

        try:
            # find the tag : <img ... >
            divs = soup.findAll('div', {"class": "product-small box"})
            for div in divs:
                anchor = div.findChildren("a")[0]
                link = anchor.get("href")
                img = anchor.find(
                    "img",
                    {"class": "show-on-hover absolute fill hide-for-small back-image"}
                ).get("srcset").split(" ")[-2]

                category = div.findChildren(
                    "p",
                    {"class": "category uppercase is-smaller no-text-overflow product-cat op-7"}
                )[0].text.strip()

                product_name = div.findChildren(
                    "p",
                    {"class": "name product-title woocommerce-loop-product__title"}
                )[0].text

                item = [product_name, link, category, img]
                current_page_products.append(item)

            logging.info(f"Found {len(divs)} products")
        except Exception as ex:
            logging.error(f"Exception, ex: {ex}")

    return current_page_products


def save_image(url, output_dir, pn=None):
    img_name = url.split("/uploads/")[-1].replace("/", "_")

    try:
        headers = {
            "User-Agent": "PostmanRuntime/7.26.5",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

        result = requests.get(url, headers=headers)
        img_data = result.content

        if result.status_code != 200:
            logging.warning(f"No success for image {url}. Status code: {result.status_code}, nothing else I can do...")
            return img_name, False

        with open(os.path.join(output_dir, img_name), 'wb') as handle:
            handle.write(img_data)
        logging.info(f"Downloaded image {url} for product {pn}")
        return img_name, True
    except Exception as ex:
        logging.error(f"Error while downloading image {url}, ex: {ex}")
        return img_name, False


def save_image_if_not_exists(url, output_dir, pn=None):
    img_name = url.split("/uploads/")[-1].replace("/", "_")

    if os.path.exists(os.path.join(output_dir, img_name)):
        logging.info(f"Nothing to do for image: {img_name}")
        return img_name, True

    return save_image(url, output_dir, pn)


def get_product_image(product_url):
    logging.info(f"Getting product categories for: {product_url}")
    while True:
        try:
            # get product page
            headers = {
                "User-Agent": "PostmanRuntime/7.26.5",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }

            result = requests.get(product_url, headers=headers)
            content = result.content
            break
        except requests.exceptions.RequestException as ex:
            logging.error(f"Connection error... Exception: {ex}. Sleeping for 2s")
            time.sleep(2)

    # get soup
    soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

    categories = []
    product_name = None
    img_url = None

    try:
        anchors = soup.findAll('a', {"rel": "tag"})
        for anchor in anchors:
            categories.append(anchor.text)

        img = soup.findAll("img", {"class": "wp-post-image"})[0]

        product_name = soup.findAll("h1", {"class": "product_title"})[0].text
        img_url = img.get("src")
    except Exception as ex:
        logging.error(f"Exception, ex: {ex}")

    return [product_url, product_name, ";".join(categories), img_url]


def generate_stats(df):
    df["year"] = df.apply(
        lambda row: row["img_url"].split("/")[-3],
        axis=1
    )
    df["month"] = df.apply(
        lambda row: row["img_url"].split("/")[-2],
        axis=1
    )
    df["img_name"] = df.apply(
        lambda row: row["img_url"].split("/")[-1],
        axis=1
    )
    df["img_base_name"] = df.apply(
        lambda row: row["img_url"].split("/")[-1].split("-")[0],
        axis=1
    )

    def get_product_number(row):
        try:
            return int(re.findall(r"\d+", row["img_name"])[0])
        except IndexError:
            return 1

    df["img_number"] = df.apply(
        lambda row: get_product_number(row),
        axis=1,
    )
    df["img_number_suffix"] = df.apply(
        lambda row: "-".join(row["img_url"].split("/")[-1].split("-")[1:]).replace(".jpg", ""),
        axis=1
    )
    df["is_first"] = df.apply(
        lambda row: not row["product_url"].strip("/").split("/")[-1].split("-")[-1].isdigit(),
        axis=1
    )
    df["product_basename"] = df.apply(
        lambda row: "-".join(row["product_url"].strip("/").split("/")[-1].split("-")[:-1]),
        axis=1
    )
    df["product_number"] = df.apply(
        lambda row: np.nan if row["is_first"] else int(row["product_url"].strip("/").split("/")[-1].split("-")[-1]),
        axis=1
    )

    return df


def get_max_prod_nums(df):
    uniq = df["product_basename"].unique()
    resutls = []
    for pbn in uniq:
        products = df[df["product_basename"] == pbn].iloc[0]
        max_num = df[df["product_basename"] == pbn]["product_number"].max()
        if max_num is np.nan:
            continue
        resutls.append([pbn, int(max_num), products["category"], products["product_name"]])
    return resutls


def extend_product_list(item, output_path):
    logging.info(f"Getting links on infinite scroll page")

    product_basename = item[0]
    number = item[1]
    category = item[2]
    product_name = item[3]
    results = []

    headers = {
        "User-Agent": "PostmanRuntime/7.26.5",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    def is_20k(image_url: str) -> bool:
        return not image_url.endswith("-2.jpg")

    def generate_additional_image(image_url):
        return image_url.replace(".jpg", "-2.jpg") if is_20k(image_url) else image_url.replace("-2.jpg", ".jpg")

    for i in range(0, number + 2):
        url = f"{base_url}/product/{product_basename}" if i == 0 else f"{base_url}/product/{product_basename}-{i}"

        logging.info(f"Getting URL  {url} for product: {product_basename}")

        content = requests.get(url, headers=headers).content

        # get soup
        soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

        try:
            div = soup.find("div", {"class": "woocommerce-product-gallery__image"})
            img_url = div.findChild("a").get("href")
            additional_image_url = generate_additional_image(img_url)

            logging.info(f"Found new images for product {product_basename}")

            if save_image_if_not_exists(img_url, output_path, product_basename)[1]:
                results.append([product_name, category, url, img_url, product_basename])

            if save_image_if_not_exists(additional_image_url, output_path, product_basename)[1]:
                results.append([product_name, category, url, additional_image_url, product_basename])
        except Exception as ex:
            logging.error(f"Exception, ex: {ex}")

    return results


if __name__ == "__main__":
    base_name = "www.eurocorals.com"
    base_url = f"https://{base_name}"

    client = Client(n_workers=16, threads_per_worker=4)

    # links = get_product_links()
    #
    # logging.info(f"Found total of {len(links)} products.")
    #
    # df = pd.DataFrame(list(links), columns=["product_name", "product_url", "category", "img_url"])
    # df.to_csv(f"../datasets/scrape/{base_name}.csv", header=True, index=False)
    #
    output_dir = f"../datasets/scrape/{base_name}"
    # os.makedirs(output_dir, exist_ok=True)
    #
    df = pd.read_csv(f"{output_dir}.csv")
    df = generate_stats(df)
    df.to_csv(f"{output_dir}.csv", header=True, index=False)

    base_product_list = get_max_prod_nums(df)

    result = db.from_sequence(base_product_list).map(extend_product_list, output_dir).compute()

    extended_products = []
    for product_list in result:
        extended_products.extend(product_list)

    extended_df = pd.DataFrame(
        extended_products,
        columns=["product_name", "category", "product_url", "img_url", "product_basename"]
    )

    result_df = pd.concat([df, extended_df]) \
        .drop_duplicates(subset=["product_name", "product_url", "category", "img_url"])

    result_df = generate_stats(result_df)

    result_df.to_csv(f"{output_dir}-extended.csv", header=True, index=False)

    # img_urls = []
    #
    # df = pd.read_csv("../datasets/corals_com_aggregated.csv")

    # for i, row in df.iterrows():
    #     year = row['year']
    #     month = row["month"]
    #     row_min = row["min"] if row["min"] != row["max"] else 0
    #     row_max = row["max"]
    #     for number in range(row_min, row_max + 1):
    #         img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/IMG_{number:04}.jpg")

    # ============================= smart img scrape based on previous product postage dates
    # import glob
    #
    # files = glob.glob(os.path.join(output_dir, "*.jpg"))
    # df = pd.DataFrame(files, columns=["file"])
    # df["year"] = df.apply(
    #     lambda row: row["file"].split("/")[-1].split("_")[-3],
    #     axis=1
    # )
    # df["month"] = df.apply(
    #     lambda row: row["file"].split("/")[-1].split("_")[-2],
    #     axis=1
    # )
    # df["day"] = df.apply(
    #     lambda row: row["file"].split("/")[-1].split("_")[-1][:4].replace("EC", ""),
    #     axis=1
    # )
    #
    # grouped = df.groupby(["year", "month", "day"])["file"].count().reset_index()
    #
    # for i, row in grouped.iterrows():
    #     img_urls.append(f"{base_url}/wp-content/uploads/{row.year}/{row.month}/EC{row.day}{row.month}.jpg")
    #     img_urls.append(f"{base_url}/wp-content/uploads/{row.year}/{row.month}/EC{row.day}{row.month}-2.jpg")
    #     for num in range(200):
    #         img_urls.append(f"{base_url}/wp-content/uploads/{row.year}/{row.month}/EC{row.day}{row.month}-{num}.jpg")
    #         img_urls.append(f"{base_url}/wp-content/uploads/{row.year}/{row.month}/EC{row.day}{row.month}-{num}-2.jpg")

    # ============================= brute force img scrape

    # for year in [2018, 2019]:
    #     for month in range(1, 13):
    #         for day in range(1, 32):
    #             # img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/EC{day:02}{month:02}.jpg")
    #             # img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/EC{day:02}{month:02}-2.jpg")
    #             for i in range(1,2):
    #                 if year == 2020 and month > 10:
    #                     continue
    #                 img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/EC{day:02}{month:02}-{i}.jpg")
    #                 # img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/EC{day:02}{month:02}-{i}-2.jpg")
    #

    # logging.info(f"Generated total of {len(img_urls)} img urls")
    # logging.info(f"Some URLs: {img_urls[:5]}")
    #
    # time.sleep(5)
    #
    # db.from_sequence(img_urls).map(save_image_if_not_exists, output_dir).compute()

    client.close()
