from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import requests
import os
import ssl
from tqdm import tqdm
import traceback
import time
import dask.bag as db
import numpy as np
from pprint import pprint

DRIVER_PATH = '/usr/bin/chromedriver'
SCROLL_PAUSE_TIME = 0.5


def dasked_download(img_url, dest):
    source = img_url.get_attribute("src")
    try:
        result = requests.get(source)
        img_data = result.content
        img_name = os.path.basename(source)
        with open(os.path.join(dest, img_name), 'wb') as handle:
            handle.write(img_data)
        return [img_name, 1, ""]
    except Exception as ex:
        return [source, 0, str(ex)]


def __scroll_down_page(wd, speed=8):
    current_scroll_position, new_height = 0, 1
    while current_scroll_position <= new_height:
        current_scroll_position += speed
        wd.execute_script("window.scrollTo(0, {});".format(current_scroll_position))
        new_height = wd.execute_script("return document.body.scrollHeight")


if __name__ == "__main__":
    service = Service(DRIVER_PATH)
    service.start()
    driver = webdriver.Remote(service.service_url)
    ssl._create_default_https_context = ssl._create_unverified_context

    for i in range(0, 62):
        driver.get(f'https://eyecatchingcoral.com/coral-shipments/shipments/{i}')

        __scroll_down_page(driver)

        output_path = f'./shipments/{i}/'
        os.makedirs(output_path, exist_ok=True)

        # print("sleeping....")
        # time.sleep(5)

        # get the image source
        images = driver.find_elements_by_tag_name('img')
        # results = []
        results = db.from_sequence(images).map(dasked_download, output_path).compute()
        # for i in tqdm(images):
        #     results.append(dasked_download(i, output_path))

        pprint(results)

    driver.quit()
