import asyncio
import aiohttp
import aiofiles

from aiohttp import ClientSession, ClientConnectorError

async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
    except ClientConnectorError:
        return

    if resp.status == 200:
        img_name = url.split("/")[-1]
        f = await aiofiles.open(f'./download/wwc/{img_name}.jpg', mode='wb')
        await f.write(await resp.read())
        await f.close()
        return
        
    if resp.status != 404:
        print(f"500! - {url}")

    del resp

async def make_requests(urls: set, **kwargs) -> None:
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                fetch_html(url=url, session=session, **kwargs)
            )
        results = await asyncio.gather(*tasks)

    print("DONE")

if __name__ == "__main__":
    import pathlib
    import sys

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent

    chars = ['a', 'd']

    # w_urls = set()
    # for i in range(380,500):
    #     padded_i = str(i).rjust(3,"0")
    #     for j in range(100):
    #         padded_j = str(j).rjust(2,"0")
    #         for c in chars:
    #             w_urls.add(f"https://cdn.shopify.com/s/files/1/0021/4958/0912/products/W-0{padded_i}20-{padded_j}{c}_900x.jpg")

    # print(f"generated W {len(w_urls)} urls")
    # asyncio.run(make_requests(urls=w_urls))

    col_urls = set()
    for i in range(100,333):
        padded_i = str(i).rjust(3,"0")
        for j in range(100):
            padded_j = str(j).rjust(2,"0")
            for c in chars:
                col_urls.add(f"https://cdn.shopify.com/s/files/1/0021/4958/0912/products/COL-0{padded_i}20-{padded_j}{c}_900x.jpg")

    print(f"generated COL {len(col_urls)} urls")
    asyncio.run(make_requests(urls=col_urls))