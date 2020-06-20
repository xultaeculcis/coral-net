import asyncio
import aiofiles

from aiohttp import ClientSession


async def fetch_html(url: str, session: ClientSession, **kwargs):
    try:
        resp = await session.request(method="GET", url=url, timeout=60, **kwargs)
    except Exception as ex:
        print(type(ex))  # the exception instance
        print(ex.args)
        return

    if resp.status == 200:
        img_name = url.split("uploads/")[-1].replace("/", "_")
        f = await aiofiles.open(f'D:/Repos/DeepLearning/coral-classifier/src/notebooks/download/cz/{img_name}', mode='wb')
        await f.write(await resp.read())
        await f.close()

        return

    print(f"{resp.status} - {url}")


async def make_requests(urls: list, **kwargs):
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                fetch_html(url=url, session=session, **kwargs)
            )
        await asyncio.gather(*tasks)

    print("DONE")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    import pathlib
    import sys
    from datetime import datetime, timedelta
    import numpy as np
    from time import sleep

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+."
    here = pathlib.Path(__file__).parent

    current = datetime.utcnow()
    dates = []
    for i in range(2019, 2020):
        for j in range(1, 13):
            dates.append((str(i), str(j).rjust(2, "0")))

    for date in dates:
        urls = []
        for i in np.arange(16000, 18000, dtype=int):
            urls.append(
                f"https://www.coral.zone/wp-content/uploads/{date[0]}/{date[1]}/c{i}-1024x683.jpg")

        print(date)
        asyncio.run(make_requests(urls=urls))
        print("*" * 150)
        sleep(2)
