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
    products_per_page = 24
    links = []
    while True:
        logging.info(f"Getting links on page # {i}")
        driver.get(f'{base_url}/product-category/all-livestock/page/{i}/')

        # __scroll_down_page(driver)

        # get the image source
        current_page_links = [element.get_attribute("href") for element in driver.find_elements_by_tag_name('a')]
        current_page_links = [link for link in current_page_links if link.startswith(f"{base_url}/product/")]

        logging.info(f"Found {len(current_page_links)} products on page # {i}")

        i += 1
        links.extend(current_page_links)

        if len(current_page_links) < products_per_page:
            logging.info("No more links available")
            break

    driver.quit()

    return set(links)


def save_image(url, output_dir, pn=None):
    img_name = url.split("/uploads/")[-1].replace("/", "_")

    try:
        result = requests.get(url)
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

        product_url = f"{base_url}/product/{pn}/"

        logging.info(f"Getting images for product: {product_url}")

        content = requests.get(product_url).content

        # get soup
        soup = BeautifulSoup(content, 'lxml')  # choose lxml parser

        if soup.title.text.startswith("Nothing found"):
            logging.info(f"No more results for {product_name}")
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
            if image_tag.get('src') == "https://www.corals.com/wp-content/uploads/woocommerce-placeholder.png":
                logging.warning(f"Only placeholder found for product: {product_url}")
            else:
                logging.error(f"Exception for {product_url}, image tag: {image_tag.get('src')}, ex: {ex}")
            i += 1
            continue

        if i > 400:
            break
        i += 1

    print(dicts)

    return dicts


if __name__ == "__main__":
    base_url = "https://www.corals.com"
    client = Client(n_workers=16, threads_per_worker=4)

    links = get_product_links()

    logging.info(f"Found total of {len(links)} products.")

    df = pd.DataFrame(list(links), columns=["product_link"])
    df.to_csv("./corals_links.csv", header=True, index=False)

    df = pd.read_csv("../datasets/corals_links.csv")

    def get_product_number(row):
        try:
            return int(re.findall(r"\d+", row["product_link"])[0])
        except IndexError:
            return 1

    df["product_name"] = df.apply(
        lambda row: re.sub(r"\d", "", row["product_link"]).strip("-/").split("/")[-1], axis=1
    )
    df["product_number"] = df.apply(lambda row: get_product_number(row), axis=1)
    df["product_name"].drop_duplicates().to_csv("corals_products.csv", header=True, index=False)

    products = pd.read_csv("../datasets/corals_products.csv")["product_name"].values

    scrape_results = db.from_sequence(products).map(download_images, "../datasets/scrape/www.corals.com/").compute()

    results = []
    for result in scrape_results:
        results.extend(result)

    pprint.pprint(results)

    results = pd.DataFrame(data=results)
    results.to_csv("../datasets/scrape/www.corals.com.csv", index=False, header=True)

    results = pd.read_csv("../datasets/scrape/www.corals.com.csv")
    db.from_sequence(results["image"].values) \
        .map(save_image_if_not_exists, "../datasets/scrape/www.corals.com") \
        .compute()

    img_urls = []

    df = pd.read_csv("../datasets/corals_com_aggregated.csv")

    for i, row in df.iterrows():
        year = row['year']
        month = row["month"]
        row_min = row["min"] if row["min"] != row["max"] else 0
        row_max = row["max"]
        for number in range(row_min, row_max + 1):
            img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/IMG_{number:04}.jpg")

    # for year in [2018, 2019, 2020]:
    #     for month in range(1, 13):
    #         for number in range(10000):
    #             if year == 2018 and month < 10:
    #                 continue
    #             if year == 2020 and month > 10:
    #                 continue
    #             img_urls.append(f"{base_url}/wp-content/uploads/{year}/{month:02}/IMG_{number:04}.jpg")
    #
    # logging.info(f"Generated total of {len(img_urls)} img urls")
    # logging.info(f"Some URLs: {img_urls[:5]}")
    #
    # output_dir = "../datasets/scrape/www.corals.com/"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # db.from_sequence(img_urls).map(save_image_if_not_exists, output_dir).compute()

    client.close()
