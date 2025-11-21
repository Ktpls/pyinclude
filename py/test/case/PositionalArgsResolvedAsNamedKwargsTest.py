from test.autotest_common import *


class PositionalArgsResolvedAsNamedKwargsTest(unittest.TestCase):
    def test_basicUsage(self):
        @EasyWrapper
        def auto_inc(func, arg_names):
            reso = PositionalArgsResolvedAsNamedKwargs(func)

            @functools.wraps(func)
            def new_func(*args, **kwargs):
                args, kwargs = reso.apply(
                    lambda k, v: v + 1 if k in arg_names else v, args, kwargs
                )
                return func(*args, **kwargs)

            return new_func

        @auto_inc(["a", "b"])
        def test(a, b, c, d):
            return {"a": a, "b": b, "c": c, "d": d}

        expected = {"a": 1, "b": 1, "c": 0, "d": 0}
        self.assertEqual(test(0, 0, 0, 0), expected)
        self.assertEqual(test(0, b=0, c=0, d=0), expected)
        self.assertEqual(test(a=0, b=0, c=0, d=0), expected)
