import asyncio
import aiohttp
import aiofiles

from aiohttp import ClientSession, ClientConnectorError

async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
    except ClientConnectorError:
        return (url, 404)

    if resp.status == 200:
        img_name = url.split("_")[1]
        f = await aiofiles.open(f'./download/bc/{img_name}.png', mode='wb')
        await f.write(await resp.read())
        await f.close()
    return (url, resp.status)

async def make_requests(urls: set, **kwargs) -> None:
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                fetch_html(url=url, session=session, **kwargs)
            )
        results = await asyncio.gather(*tasks)

    for result in results:
        if (result[1] != 200):
            continue
        print(f'{result[1]} - {str(result[0])}')

    print("DONE")

if __name__ == "__main__":
    import pathlib
    import sys

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent

    urls = set()
    for i in range(10000):
        img = str(i).rjust(4, "0")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}_1024x1024.jpg?v=1584152225")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}-2_1024x1024.jpg?v=1584152225")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}-3_1024x1024.jpg?v=1584152225")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}-Edit_1024x1024.jpg?v=1584152225")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}-Edit-2_1024x1024.jpg?v=1584152225")
        urls.add(f"https://cdn.shopify.com/s/files/1/0756/8419/products/IMG_{img}-Edit-2-2_1024x1024.jpg?v=1584152225")

    asyncio.run(make_requests(urls=urls))