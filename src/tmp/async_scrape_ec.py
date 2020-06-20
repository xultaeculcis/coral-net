import asyncio
import aiohttp
import aiofiles
from time import sleep

from aiohttp import ClientSession, ClientConnectorError


async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        resp = await session.request(method="GET", url=url, **kwargs)
    except:
        return

    if resp.status == 200:
        img_name = url.split("uploads/")[-1].replace("/", "_")
        f = await aiofiles.open(f'./datasets/download/ec/{img_name}', mode='wb')
        await f.write(await resp.read())
        await f.close()

        return

    print(f"{resp.status} - {url}")


async def make_requests(urls: list, **kwargs) -> None:
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

    urls = []
    dates = [
        # "2020/02/08",
        # "2020/02/21",
        "2020/03/21",
        "2019/01/17",
        "2019/02/08",
        "2019/02/15",
        "2019/02/22",
        "2019/03/01",
        "2019/03/07",
        "2019/03/15",
        "2019/03/29",
        "2019/04/05",
        "2019/05/10",
        "2019/05/17",
        "2019/06/07",
        "2019/07/04",
        "2019/07/26",
        "2019/08/02",
        "2019/08/30",
        "2019/09/27",
        "2019/10/18",
        "2019/11/01",
        "2019/11/02",
        "2019/11/16",
        "2019/11/22",
        "2019/12/06",
        "2019/12/13"
    ]

    for date in dates:
        splitted = date.split("/")

        for i in range(300):
            urls.append(f"https://www.eurocorals.com/wp-content/uploads/{splitted[0]}/{splitted[1]}/EC{splitted[2]}{splitted[1]}-{i}.jpg")

    print(len(dates), len(urls))
    i = 1
    for chunk in chunks(urls, 30):
        print(f"Processing chunk #{i}")
        asyncio.run(make_requests(urls=chunk))
        i += 1
        sleep(2)

