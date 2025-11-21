from test.autotest_common import *


class UnionableClassTest(unittest.TestCase):
    def test_basic_usage(self):
        @dataclasses.dataclass
        class Setting(UnionableClass):
            v1: int = None
            v2: int = None

            def __all_field_names__(self):
                return [k.name for k in dataclasses.fields(self)]

        self.assertDictEqual(
            dataclasses.asdict(Setting(v1=1) | Setting(v2=2)), {"v1": 1, "v2": 2}
        )
