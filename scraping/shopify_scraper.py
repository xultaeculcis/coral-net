from dask.distributed import Client
import dask.bag as db
import os
import pandas as pd
import csv
import json
import time
import numpy as np
import urllib.request
from urllib.error import HTTPError
import logging
import uuid
import requests
import glob
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'


def get_page(url, page, collection_handle=None):
    logging.info(f"Getting products for collection: {collection_handle} on page #{page} for URL: {url}")

    full_url = url
    if collection_handle:
        full_url += '/collections/{}'.format(collection_handle)
    full_url += '/products.json'
    req = urllib.request.Request(
        full_url + '?page={}'.format(page),
        data=None,
        headers={
            'User-Agent': USER_AGENT
        }
    )
    while True:
        try:
            data = urllib.request.urlopen(req, timeout=30).read()
            break
        except HTTPError:
            logging.warning('Blocked! Sleeping...')
            time.sleep(60)
            logging.info('Retrying')
        except Exception:
            import traceback
            ex = traceback.format_exc()
            logging.error(f'Something bad happened... Ex: {ex}')
            data = None
            break

    if data:
        products = json.loads(data.decode())['products']
    else:
        products = []

    logging.info(f"Found {len(products)} products on page #{page} for collection: {collection_handle} for URL: {url}")

    return products


def get_page_collections(url):
    logging.info(f"Getting collections JSON for URL: {url}")

    full_url = url + '/collections.json'
    page = 1
    while True:
        req = urllib.request.Request(
            full_url + '?page={}'.format(page),
            data=None,
            headers={
                'User-Agent': USER_AGENT
            }
        )
        while True:
            try:
                data = urllib.request.urlopen(req).read()
                break
            except HTTPError:
                print('Blocked! Sleeping...')
                time.sleep(180)
                print('Retrying')

        cols = json.loads(data.decode())['collections']

        logging.info(f"Found {len(cols)} in collections.json for URL: {url}")

        if not cols:
            break
        for col in cols:
            yield col
        page += 1


def check_shopify(url):
    logging.info(f"Checking shopify URL: {url}")
    try:
        get_page(url, 1)
        return True
    except Exception:
        logging.error("Link check failed")
        return False


def fix_url(url):
    logging.info(f"Normalizing URL: {url}")

    fixed_url = url.strip()
    if not fixed_url.startswith('http://') and \
            not fixed_url.startswith('https://'):
        fixed_url = 'https://' + fixed_url

    return fixed_url.rstrip('/')


def extract_products_collection(url, col):
    logging.info(f"Extracting products from collection {col} for URL: {url}")
    page = 1

    products = get_page(url, page, col)

    while products:
        for product in products:
            title = product['title']
            product_type = product['product_type']
            product_url = url + '/products/' + product['handle']
            product_handle = product['handle']

            def get_image(variant_id):
                images = product['images']
                for i in images:
                    k = [str(v) for v in i['variant_ids']]
                    if str(variant_id) in k:
                        return i['src']

                return ''

            for i, variant in enumerate(product['variants']):
                price = variant['price']
                option1_value = variant['option1'] or ''
                option2_value = variant['option2'] or ''
                option3_value = variant['option3'] or ''
                option_value = ' '.join([option1_value, option2_value,
                                         option3_value]).strip()
                sku = variant['sku']
                main_image_src = ''
                if product['images']:
                    main_image_src = product['images'][0]['src']

                image_src = get_image(variant['id']) or main_image_src
                stock = 'Yes'
                if not variant['available']:
                    stock = 'No'

                row = {'sku': sku, 'product_type': product_type,
                       'title': title, 'option_value': option_value,
                       'price': price, 'stock': stock, 'body': str(product['body_html']),
                       'variant_id': product_handle + str(variant['id']),
                       'product_url': product_url, 'image_src': image_src}
                for k in row:
                    row[k] = str(row[k].strip()) if row[k] else ''
                yield row

        page += 1
        products = get_page(url, page, col)


def extract_products(url, path, collections=None):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Code', 'Collection', 'Category',
                         'Name', 'Variant Name',
                         'Price', 'In Stock', 'URL', 'Image URL', 'Body'])
        seen_variants = set()
        for col in get_page_collections(url):
            if collections and col['handle'] not in collections:
                continue
            handle = col['handle']
            title = col['title']

            logging.info(f"Processing collection {title}")

            for product in extract_products_collection(url, handle):
                variant_id = product['variant_id']
                if variant_id in seen_variants:
                    continue

                seen_variants.add(variant_id)
                writer.writerow([product['sku'], str(title),
                                 product['product_type'],
                                 product['title'], product['option_value'],
                                 product['price'],
                                 product['stock'], product['product_url'],
                                 product['image_src'], product['body']])

        logging.info(f"Written total of {len(seen_variants)} product variants to the CSV")


def download_image(collection_category_name_img_url: np.ndarray, dest: str):
    collection = collection_category_name_img_url[0]
    category = collection_category_name_img_url[1]
    name = collection_category_name_img_url[2]
    img_url = collection_category_name_img_url[3]
    name = f"{collection}_{category}_{name}"

    # ensure not nan
    if img_url == np.nan or img_url is None:
        logging.error(f"Provided URL was NAN")
        return ["", False, "np.NaN as img url"]

    # sanitize name
    interps = ",;:<>'\"[]{}=+()!@#$%^&*/|\\"
    for i in interps:
        name = name.replace(i, "_")

    name = f"{name.lower().replace(' ', '_')}_{str(uuid.uuid4()).replace('-', '')}.jpg"

    logging.info(f"Downloading image {img_url}")

    # download
    try:
        result = requests.get(img_url)
        img_data = result.content
        with open(os.path.join(dest, name), 'wb') as handle:
            handle.write(img_data)
        time.sleep(1)
        return [img_url, True, ""]
    except Exception as ex:
        logging.error(ex)
        time.sleep(1)
        return [img_url, False, str(ex)]


def download_images(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    images = df[["Collection", "Category", "Name", "Image URL"]].values
    client = Client(n_workers=8, threads_per_worker=2)
    results = db.from_sequence(images).map(download_image, image_folder).compute()
    for r in results:
        if not r[1]:
            logging.error(f"Downloading of image {r[0]} failed with following message: {r[2]}")
    client.close()


def get_csv_file(url, output_dir):
    sanitized_dir_name = url.replace('https://', '')
    csv_path = os.path.join(output_dir, f"{sanitized_dir_name}.csv")
    image_folder = os.path.join(output_dir, sanitized_dir_name)
    os.makedirs(image_folder, exist_ok=True)

    collections = []
    extract_products(url, csv_path, collections)
    return image_folder


if __name__ == '__main__':
    URLS = [
        "https://worldwidecorals.com",
        "https://battlecorals.com",
        "https://uniquecorals.com",
        "https://www.clickcorals.com",
        "https://cornbredcorals.com",
        "https://deepbluereef.net",
        "https://pacificeastaquaculture.com",
        "https://www.exoticaquacultureaustralia.com",
        "https://reefshackllc.com",
        "https://www.reefsecrets.com.au",
        "https://www.whitlynaquatics.com",
        "https://fraghousecorals.com",
        "https://www.coralsellerz.com",
        "https://mosreef.com",
        "https://thecornerreefonline.com",
        "https://underwatergardeners.com",
        "https://www.tntfishcoral.com",
        "https://savagereefers.com",
        "https://www.dkreeftreasures.com",
        "https://insanecoral.com",
        "https://fishofhex.com",
        "https://jpcoralfrags.com",
        "https://www.freshnmarine.com",
        "https://vividaquariums.com",
        "https://www.reef2land.com",
        "https://aquasd.com",
        "https://tckcorals.com",
        "https://www.canadacorals.com",
        "https://www.house-of-sticks.com",
        "https://suncoralsdirect.com",
        "https://apfrags.com",
        "https://boomcorals.com",
        "https://www.houseofcorals.com",
        "https://bayareacoral.com",
        "https://tsmaquatics.com",
        "https://www.aquaticbrethren.com",
        "https://www.e-marineaquatics.co.uk",
        "https://www.piratesreefcorals.com",
        "https://defugiumsreef.com",
        "https://tjmcorals.com"
    ]
    OUTPUT_DIR = "../datasets/scrape"

    logging.info(f"Running scrape script with following URLs: {URLS}")

    # for url in URLS:
    #     get_csv_file(url, OUTPUT_DIR)

    for csv_path in glob.glob(os.path.join(OUTPUT_DIR, "*.csv")):
        download_images(csv_path, csv_path.replace(".csv", ""))
