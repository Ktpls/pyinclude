from test.autotest_common import *


class DataclassInitReorderTest(unittest.TestCase):
    def test_inherit(self):
        @dataclasses.dataclass
        class Parent:
            x: str = ""  # 默认值
            y: typing.Any = dataclasses.field(
                default_factory=lambda: None
            )  # 使用 default_factory 作为默认值

        # 子类
        @dataclasses.dataclass
        class Child(Parent):
            z: str  # 没有默认值

        child = Child("z", "x", "y")
        self.assertDictEqual(
            BeanUtil.toJsonCompatible(child),
            {"x": "x", "y": "y", "z": "z"},
        )
