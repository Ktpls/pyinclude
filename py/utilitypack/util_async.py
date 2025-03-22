import asyncio
import aiohttp

class DownloadFileAsync:
    class DownloadCallback:
        def onSuccess(self, url, content): ...
        def onError(self, url, e): ...
    @staticmethod
    async def downloadUrlAioHttp(
        url,
        client: aiohttp.ClientSession,
        callback: DownloadCallback = None,
        timeout=None,
    ):
        timeout = timeout or 5
        try:
            r = await client.get(url, timeout=timeout)
            r.raise_for_status()
            r = await r.content.read()
            if callback is not None:
                callback.onSuccess(url, r)
        except Exception as e:
            if callback is not None:
                callback.onError(url, e)

    @staticmethod
    async def downloadUrlList(
        urlList: list[str],
        callback: DownloadCallback = None,
        timeout=None,
    ):
        # use asyncio.run(downloadList(urlList))
        async with aiohttp.ClientSession() as client:
            await asyncio.gather(
                *[
                    DownloadFileAsync.downloadUrlAioHttp(url, client, callback, timeout)
                    for url in urlList
                ]
            )
