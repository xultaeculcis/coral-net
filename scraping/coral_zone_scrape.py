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


def __scroll_down_page(wd, speed=8):
    current_scroll_position, new_height = 0, 1
    while current_scroll_position <= new_height:
        current_scroll_position += speed
        wd.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
        new_height = wd.execute_script("return document.body.scrollHeight")


def get_product_links():
    service = Service(DRIVER_PATH)
    service.start()
    driver = webdriver.Remote(service.service_url)
    ssl._create_default_https_context = ssl._create_unverified_context

    logging.info(f"Starting scraping of links for: {base_url}")

    # get all product links from all pages
    i = 1
    products_per_page = 20
    links = []
    while True:
        logging.info(f"Getting links on page # {i}")
        driver.get(f'{base_url}/product-category/novita/page/{i}/')

        __scroll_down_page(driver)

        # get the image source
        l = [element.get_attribute("href") for element in driver.find_elements_by_class_name(
            'woocommerce-LoopProduct-link')]

        logging.info(f"Found {len(l)} products on page # {i}")

        i += 1
        links.extend(l)

        if len(l) < products_per_page:
            logging.info("No more links available")
            break

    driver.quit()

    return links


def save_image(url, output_dir, pn=None):
    img_name = url.split("/")[-1]

    try:
        result = requests.get(url)
        img_data = result.content

        if result.status_code != 200:
            logging.warning(f"No success for image {url}. Status code: {result.status_code}")
            return img_name, False

        with open(os.path.join(output_dir, img_name), 'wb') as handle:
            handle.write(img_data)
        logging.info(f"Downloaded image {url} for product {pn}")
        return img_name, True
    except Exception as ex:
        logging.error(f"Error while downloading image {url}, ex: {ex}")
        return img_name, False


def save_image_if_not_exists(url, output_dir, pn=None):
    img_name = url.split("/")[-1]

    if os.path.exists(os.path.join(output_dir, img_name)):
        logging.info(f"Nothing to do for image: {img_name}")
        return img_name, True

    return save_image(url, output_dir, pn)


def download_images(product_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    i = 1
    dicts = []

    while True:
        # get contents from url
        if i > 1:
            pn = f"{product_name}-{i}"
        else:
            pn = product_name

        product_url = f"{base_url}/prodotto/{pn}/"

        # logging.info(f"Getting images for product: {product_url}")

        content = requests.get(product_url).content

        # get soup
        soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

        if soup.title.text == "NOVITA’ – CORAL ZONE":
            # logging.info(f"No more results for {product_name}")
            break

        try:
            # find the tag : <img ... >
            image_tag = soup.findAll('img', {"class": "wp-post-image"})[0]
        except Exception as ex:
            logging.error(f"Exception for product {pn}, ex: {ex}")
            continue

        # print out image urls
        try:
            links = re.findall(r'(https?://\S+)', image_tag.get('srcset').split(", ")[-1])
            for link in links:
                dicts.append({"product": pn, "product_num": pn, "class": product_name.split("-")[0], "image": link})

        except Exception as ex:
            # if image_tag.get('src') == "https://www.coral.zone/wp-content/uploads/woocommerce-placeholder.png":
            #     logging.warning(f"Only placeholder found for product: {product_url}")
            # else:
            #     logging.error(f"Exception for {product_url}, image tag: {image_tag.get('src')}, ex: {ex}")
            i += 1
            continue

        if i > 400:
            break
        i += 1

    print(dicts)

    return dicts


if __name__ == "__main__":
    base_url = "https://www.coral.zone"
    client = Client(n_workers=8, threads_per_worker=4)

    links = get_product_links()

    logging.info(f"Found total of {len(links)} products.")

    df = pd.DataFrame(links, columns=["product_link"])
    df.to_csv("./cz_links.csv", header=True, index=False)

    df = pd.read_csv("../datasets/cz_links.csv")

    def get_product_number(row):
        try:
            return int(re.findall(r"\d+", row["product_link"])[0])
        except IndexError:
            return 1

    df["product_name"] = df.apply(
        lambda row: re.sub(r"\d", "", row["product_link"]).strip("-/").split("/")[-1], axis=1
    )
    df["product_number"] = df.apply(lambda row: get_product_number(row), axis=1)
    df["class"] = df.apply(lambda row: row["product_name"].split("-")[0], axis=1)
    classes = df["class"].unique()
    additional_products = [c + "-sp" for c in classes]
    df2 = pd.DataFrame(additional_products, columns=["product_name"])
    pd.concat([df, df2]).reset_index(drop=True)

    df["product_name"].drop_duplicates().to_csv("cz_products.csv", header=True, index=False)

    products = pd.read_csv("../datasets/cz_products.csv")["product_name"].values

    scrape_results = db.from_sequence(products).map(download_images, "../datasets/scrape/www.coral.zone/").compute()

    results = []
    for result in scrape_results:
        results.extend(result)

    pprint.pprint(results)

    results = pd.DataFrame(data=results)
    results.to_csv("../datasets/scrape/www.coral.zone.csv", index=False, header=True)

    results = pd.read_csv("../datasets/scrape/www.coral.zone.csv")
    db.from_sequence(results["image"].values) \
        .map(save_image_if_not_exists, "../datasets/scrape/www.coral.zone") \
        .compute()

    # imgs = [f"c{i}.jpg" for i in range(5100, 11000)]
    #
    # dirs = []
    # for year in [2018]:
    #     for month in range(2, 13):
    #             dirs.append(f"{year}/{month:02}")
    #
    # img_urls = []
    # for d in dirs:
    #     for i in imgs:
    #         img_urls.append(f"{base_url}/wp-content/uploads/{d}/{i}")
    #
    # logging.info(f"Generated total of {len(img_urls)} img urls")
    # logging.info(f"Some URLs: {img_urls[:5]}")
    #
    # output_dir = "../datasets/scrape/www.coral.zone/"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # db.from_sequence(img_urls).map(save_image, output_dir).compute()
    client.close()
