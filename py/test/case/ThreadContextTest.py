from test.autotest_common import *


class ThreadContextTest(unittest.TestCase):

    class ChildClass(ThreadLocalSingleton):
        def __init__(self):
            super().__init__()
            self.v = None

    class GrandsonClass(ChildClass): ...

    def if_existance(self, cls: ChildClass, expected_existance: bool):
        existance = cls.summon().v is not None
        self.assertEqual(existance, expected_existance)

    def test_accessable_in_thread(self):
        ThreadContextTest.ChildClass.summon().v = 1
        self.if_existance(ThreadContextTest.ChildClass, True)

    def test_inaccessable_in_another_thread(self):
        ThreadContextTest.ChildClass.summon().v = 1
        th = threading.Thread(
            target=ThreadContextTest.if_existance,
            args=(
                self,
                ThreadContextTest.ChildClass,
                False,
            ),
        )
        th.start()
        th.join()

    def test_inherited_again(self):
        ThreadContextTest.GrandsonClass.summon().v = 1
        self.if_existance(ThreadContextTest.GrandsonClass, True)
        th = threading.Thread(
            target=ThreadContextTest.if_existance,
            args=(
                self,
                ThreadContextTest.GrandsonClass,
                False,
            ),
        )
        th.start()
        th.join()

    def test_with_me_method_decorator(self):
        """
        测试WithMe方法作为装饰器使用时的功能
        """

        # 使用WithMe装饰器装饰一个函数
        @ThreadContextTest.ChildClass.WithMe()
        def test_function():
            instance = ThreadContextTest.ChildClass.summon()
            instance.v = 42
            return instance.v

        def new_thread():
            # 创建一个线程，并调用装饰后的函数
            result = test_function()
            # 验证函数被调用并且返回了正确值
            self.assertEqual(result, 42)
            # 资源在decorator的上下文结束后被立即释放，即使线程还未结束
            self.assertFalse(
                ThreadContextTest.ChildClass.__thread_local_singleton_test_val__()
            )

        t = threading.Thread(target=new_thread)
        t.start()
        t.join()
