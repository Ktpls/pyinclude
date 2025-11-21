from test.autotest_common import *


class MlambdaTest(unittest.TestCase):
    def test_newLined(self):
        f = mlambda(
            """
            def (a, b):
                c=a+b
                return c
            """
        )
        ret = f(1, 1)
        self.assertEqual(ret, 2)

    def test_oneLined(self):
        f = mlambda("""def (a, b): return a+b""")
        ret = f(1, 1)
        self.assertEqual(ret, 2)
