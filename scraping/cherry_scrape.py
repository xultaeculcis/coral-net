from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
import ssl
from dask.distributed import Client
import dask.bag as db
import re
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

    current_page_links = []

    with open("/home/xultaeculcis/Downloads/Shop - Cherry Corals.html", "r") as file:
        content = "\n".join(file.readlines())

    # get soup
    soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

    try:
        # find the tag : <img ... >
        a_tags = soup.findAll('a', {"class": "product-image-link"})
        current_page_links = [link.get('href') for link in a_tags]

    except Exception as ex:
        logging.error(f"Exception, ex: {ex}")

    logging.info(f"Found {len(current_page_links)} products")

    return set(current_page_links)


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


def get_product_categories_with_image(product_url):
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


if __name__ == "__main__":
    # base_url = "file:///home/xultaeculcis/Downloads/"
    base_url = "https://cherrycorals.com"
    client = Client(n_workers=16, threads_per_worker=4)

    # links = get_product_links()
    #
    # logging.info(f"Found total of {len(links)} products.")
    #
    # df = pd.DataFrame(list(links), columns=["product_link"])
    # df.to_csv("../datasets/cherrycorals_links.csv", header=True, index=False)

    # df = pd.read_csv("../datasets/cherrycorals_links.csv")

    # for p in df["product_link"].values:
    #     get_product_categories_with_image(p)

    # extended_products = db.from_sequence(df["product_link"].values).map(get_product_categories_with_image).compute()

    # extended_products_df = pd.DataFrame(data=extended_products, columns=["product_link", "product_name", "categories", "img_url"])
    # extended_products_df.to_csv("../datasets/cherrycorals_links_with_categories.csv", header=True, index=False)

    # extended_products_df = pd.read_csv("../datasets/cherrycorals_links_with_categories.csv")
    # extended_products_df["year"] = extended_products_df.apply(
    #     lambda row: row["img_url"].split("/")[-3],
    #     axis=1
    # )
    # extended_products_df["month"] = extended_products_df.apply(
    #     lambda row: row["img_url"].split("/")[-2],
    #     axis=1
    # )
    # extended_products_df["img_name"] = extended_products_df.apply(
    #     lambda row: row["img_url"].split("/")[-1],
    #     axis=1
    # )
    # extended_products_df["img_base_name"] = extended_products_df.apply(
    #     lambda row: row["img_url"].split("/")[-1].split("-")[0],
    #     axis=1
    # )
    #
    # def get_product_number(row):
    #     try:
    #         return int(re.findall(r"\d+", row["img_name"])[0])
    #     except IndexError:
    #         return 1
    #
    # extended_products_df["img_number"] = extended_products_df.apply(
    #     lambda row: get_product_number(row),
    #     axis=1,
    #
    # )
    # extended_products_df["img_number_suffix"] = extended_products_df.apply(
    #     lambda row: "-".join(row["img_url"].split("/")[-1].split("-")[1:]).replace(".jpg", ""),
    #     axis=1
    # )
    # extended_products_df.to_csv("../datasets/cherrycorals_links_with_categories.csv", header=True, index=False)


    # df = pd.read_csv("../datasets/cherrycorals_links_with_categories.csv")
    # df.to_csv("../datasets/scrape/cherrycorals.com.csv", header=True, index=False)
    #
    # os.makedirs("../datasets/scrape/cherrycorals.com", exist_ok=True)
    #
    # db.from_sequence(df["img_url"].values) \
    #     .map(save_image_if_not_exists, "../datasets/scrape/cherrycorals.com") \
    #     .compute()

    img_urls = []
    #
    # df = pd.read_csv("../datasets/corals_com_aggregated.csv")
    #
    # for i, row in df.iterrows():
    #     year = row['year']
    #     month = row["month"]
    #     row_min = row["min"] if row["min"] != row["max"] else 0
    #     row_max = row["max"]
    #     for number in range(row_min, row_max + 1):
    #         img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/IMG_{number:04}.jpg")

    for year in [2020]:
        for month in range(8, 11):
            for number in range(13000):
                if year == 2020 and month > 10:
                    continue
                img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/newcoral{number}.jpg")

    logging.info(f"Generated total of {len(img_urls)} img urls")
    logging.info(f"Some URLs: {img_urls[:5]}")

    output_dir = "../datasets/scrape/cherrycorals.com/"

    db.from_sequence(img_urls).map(save_image_if_not_exists, output_dir).compute()

    client.close()
