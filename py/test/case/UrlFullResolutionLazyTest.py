from test.autotest_common import *


class UrlFullResolutionLazyTest(unittest.TestCase):
    class example:
        url = r"https://picx.zhimg.com:8080/the_folder/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c&needBackground=1"
        baseHost = "zhimg.com"
        domain = "com"
        extName = "jpg"
        fileBaseName = "v2-abed1a8c04700ba7d72b45195223e0ff_l"
        fileName = "v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg"
        folder = "/the_folder/"
        host = "picx.zhimg.com:8080"
        param = "source=32738c0c&needBackground=1"
        path = "/the_folder/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg"
        port = "8080"
        protocol = "https"
        secondaryHost = "picx"

    def test_fields(self):
        res = UrlFullResolution.of(self.example.url)
        res.calcAll()
        fields = [
            "baseHost",
            "domain",
            "extName",
            "fileBaseName",
            "fileName",
            "folder",
            "host",
            "param",
            "path",
            "port",
            "protocol",
            "secondaryHost",
        ]
        self.assertDictEqual(
            {k: getattr(res, k) for k in fields},
            {k: getattr(self.example, k) for k in fields},
        )

    def test_lazy_resolving(self):
        res = UrlFullResolution.of(self.example.url)
        fields = [
            "protocol",
            "port",
            "folder",
        ]
        self.assertDictEqual(
            {k: getattr(res, k) for k in fields},
            {k: getattr(self.example, k) for k in fields},
        )

    def test_corner_dot_in_path(self):
        res = UrlFullResolution.of(
            r"C:\file\Gs\Storage\mc\.minecraft\versions\1.21.1-NeoForge_21.1.168\saves\Dragon Island"
        )
        self.assertEqual(res.fileName, r"Dragon Island")
