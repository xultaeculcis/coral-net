from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import requests
import os
import ssl
import uuid
import dask.bag as db
from pprint import pprint
from dask.distributed import Client
import time

DRIVER_PATH = '/usr/bin/chromedriver'
SCROLL_PAUSE_TIME = 0.5


def dasked_download(name_img_url_tuple, dest):
    name = name_img_url_tuple[0]
    img_url = name_img_url_tuple[1]
    try:
        result = requests.get(img_url)
        img_data = result.content
        with open(os.path.join(dest, name + ".jpg"), 'wb') as handle:
            handle.write(img_data)
        time.sleep(1)
        return [name, 1, ""]
    except Exception as ex:
        time.sleep(1)
        return [name, 0, str(ex)]


def __scroll_down_page(wd, speed=8):
    current_scroll_position, new_height = 0, 1
    while current_scroll_position <= new_height:
        current_scroll_position += speed
        wd.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
        new_height = wd.execute_script("return document.body.scrollHeight")


def dasked_get_images(url):
    service = Service(DRIVER_PATH)
    service.start()
    driver = webdriver.Remote(service.service_url)
    ssl._create_default_https_context = ssl._create_unverified_context

    title = url.split("/")[-1]
    driver.get(url)
    local_images = driver.find_elements_by_tag_name("img")
    img_set = set()

    for i in local_images:
        try:
            srcset = i.get_attribute("srcset")
            cdn_link = srcset.split("900w, ")[-1].replace("1080w", "").strip()

            if "/logo_" in cdn_link:
                continue

            if "-logo-" in cdn_link:
                continue

            if cdn_link:
                img_set.add(f"https:{cdn_link}")
        except Exception:
            pass

    img_set = list(img_set)

    ret = [(title + "_" + str(uuid.uuid4()).replace("-", ""), cdn_link) for cdn_link in img_set]

    driver.quit()

    return ret


def get_product_links():
    service = Service(DRIVER_PATH)
    service.start()
    driver = webdriver.Remote(service.service_url)
    ssl._create_default_https_context = ssl._create_unverified_context

    base_url = "https://worldwidecorals.com"
    output_path = f'./wwc/'

    # get all product links from all pages
    links = []
    for i in range(1, 8):
        driver.get(f'{base_url}/collections/all?page={i}')

        __scroll_down_page(driver)

        os.makedirs(output_path, exist_ok=True)

        # get the image source
        links.extend([element.get_attribute("href") for element in driver.find_elements_by_class_name('grid-product__link')])
    driver.quit()

    return links


if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1)
    links = get_product_links()

    images = db.from_sequence(links).map(dasked_get_images).compute()
    links = []
    for i in images:
        links.extend(i)
    pprint(links)

    download_results = db.from_sequence(links).map(dasked_download, "./wwc").compute()
    pprint(download_results)

    client.close()
