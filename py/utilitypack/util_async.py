import asyncio
import httpx


class DownloadFileAsync:
    class DownloadCallback:
        def onSuccess(self, url, content): ...
        def onError(self, url, e): ...
    @staticmethod
    async def downloadUrlHttpx(
        url,
        client: httpx.AsyncClient,
        callback: DownloadCallback = None,
        timeout=None,
    ):
        timeout = timeout or 5
        try:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            content = response.content
            if callback is not None:
                callback.onSuccess(url, content)
        except httpx.RequestError as e:
            if callback is not None:
                callback.onError(url, e)

    @staticmethod
    async def downloadUrlList(
        urlList: list[str],
        callback: DownloadCallback = None,
        timeout=None,
    ):
        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *[DownloadFileAsync.downloadUrlHttpx(url, client, callback, timeout) for url in urlList]
            )
