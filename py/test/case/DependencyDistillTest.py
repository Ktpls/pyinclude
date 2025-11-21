from test.autotest_common import *


class DependencyDistillTest(unittest.TestCase):
    from utilitypack.cold.util_solid import DistillLibraryFromDependency

    def test_distill(self):
        code = ["Stream()"]
        lib = [
            f"class Stream:...",
            f"import Stream",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, f"class Stream:...")

    def test_definition_covers_importing(self):
        code = ["Stream()"]
        lib = [
            f"from somewhere import Stream",
            f"import Stream",
            f"class Stream:...",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, f"class Stream:...")

    def test_indented_definition(self):
        code = ["Stream()"]
        lib = [
            f"""\
if True:
    class Stream:
        ...
else:...\
""",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, lib[0])

    def test_usage_before_definition(self):
        code = f"""\
def foo(a:typing.Any):...
import typing
"""
        result = DistillLibraryFromDependency.find_undefined_variables(code)
        self.assertSetEqual(result, {"typing"})
