from test.autotest_common import *


class LazyFieldEvaluationTest(unittest.TestCase):
    def test_basicUsage(self):
        inst = None
        evaluation_result = 1

        class EvaluationLazy(LazyLoading):
            def _init_value(selfel):
                self.assertIsNotNone(inst)
                return evaluation_result

            value = LazyLoading.LazyField(_init_value)

        inst = EvaluationLazy()
        value_got = inst.value
        self.assertEqual(value_got, evaluation_result)
