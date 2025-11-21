from test.autotest_common import *


class AutoFunctionalTest(unittest.TestCase, RedirectedPrint):
    def test_AutoFunctional(self):
        varStatic = 0

        @AutoFunctional
        class Clz:
            def __init__(self2, val):
                self2.val = val

            def inc(self2) -> typing.Self:
                self2.val += 1

            def assertVal(self2, val) -> typing.Self:
                self.assertEqual(self2.val, val)

            @staticmethod
            def staticMethod() -> typing.Self:
                nonlocal varStatic
                varStatic += 1

        a = Clz(1)
        a.inc().assertVal(2).inc().assertVal(3).staticMethod()
        self.assertEqual(a.val, 3)
        self.assertEqual(varStatic, 1)
