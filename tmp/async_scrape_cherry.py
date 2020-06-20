import asyncio
import aiohttp
import aiofiles

from aiohttp import ClientSession, ClientConnectorError

async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
    except:
        return

    if resp.status == 200:
        img_name = url.split("/")[-1]
        try:
            f = await aiofiles.open(f'./download/wc/{img_name}', mode='wb')
            await f.write(await resp.read())
            await f.close()
        except:
            pass
        
        del resp
        return

    print(f"{resp.status} - {url}")
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

    urls = set()
    for i in range(15000):
        padded_img = str(i).rjust(5, "0")

        urls.add(f"https://www.whitecorals.com/media/images/org/d{padded_img}.jpg")

    asyncio.run(make_requests(urls=urls))

    