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
        img_name = url.split("-")[-1].replace("/", "_").replace("content_uploads_", "")
        try:
            f = await aiofiles.open(f'./download/coralscom/{img_name}', mode='wb')
            await f.write(await resp.read())
            await f.close()
        except:
            return
        
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    import pathlib
    import sys

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent

    years = [2019,2020]

    urls = set()
    for year in years:
        for month in range(1,13):
            padded_month = str(month).rjust(2, "0")
            for i in range(10000):
                if (year == 2020 and month > 4):
                    break
                padded_img = str(i).rjust(4, "0")

                urls.add(f"https://www.corals.com/wp-content/uploads/{year}/{padded_month}/IMG_{padded_img}.jpg")

    print(f"generated {len(urls)} urls")

    no = 1
    for splitted in chunks(list(urls), 10000):
        print(f"Downloading chunk #{no}")
        asyncio.run(make_requests(urls=set(splitted)))
        no += 1