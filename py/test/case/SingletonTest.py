from test.autotest_common import *


class SingletonTest(unittest.TestCase):
    @staticmethod
    def decl_singleton():
        @Singleton
        class Sg:
            def __init__(self, param):
                self.param = param

        return Sg

    def test_basic(self):
        Sg = SingletonTest.decl_singleton()
        ins = Sg(1)
        self.assertEqual(ins.param, 1)
        ins2 = Sg(2)
        self.assertEqual(ins2.param, 1)

    # def test_inheritance(self):
    #     Sg = SingletonTest.decl_singleton()

    #     class ChildSg(Sg):
    #         def __init__(self, param):
    #             # actually not called in current implementation
    #             super().__init__(param)
    #             self.is_child = True

    #     ins = ChildSg(1)
    #     self.assertEqual(ins.param, 1)
    #     self.assertEqual(ins.is_child, True)
    #     ins2 = ChildSg(2)
    #     self.assertEqual(ins2.param, 1)
    #     self.assertEqual(ins.is_child, True)


class SingletonTest(unittest.TestCase):

    def test_global_isolation(self):
        class A(SingletonGlobalIsolation): ...

        class B(A): ...

        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b = B.get_instance()
        self.assertTrue(b is not a1)

    def test_thread_isolation(self):

        class A(SingletonThreadIsolation): ...

        class B(A): ...

        # 在主线程中获取实例
        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b1 = B.get_instance()
        self.assertTrue(b1 is not a1)

        # 结果存储用于从另一个线程返回数据
        results = {}

        def worker():
            # 在工作线程中获取实例
            at = A.get_instance()
            bt = B.get_instance()

            # 验证线程隔离：不同线程应该有不同的实例
            results["a_same_thread"] = at is A.get_instance()
            results["b_same_thread"] = bt is B.get_instance()
            results["a_diff_thread"] = at is not a1
            results["b_diff_thread"] = bt is not b1

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        # 检查结果
        self.assertTrue(results["a_same_thread"])
        self.assertTrue(results["b_same_thread"])
        self.assertTrue(results["a_diff_thread"])
        self.assertTrue(results["b_diff_thread"])

    def test_context_isolation(self):
        class A(SingletonContextIsolation): ...

        class B(A): ...

        # 在当前上下文中获取实例
        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b1 = B.get_instance()
        self.assertTrue(b1 is not a1)

        # 结果存储用于从另一个上下文返回数据
        results = {}

        def worker():
            # 在新上下文中获取实例
            at = A.get_instance()
            bt = B.get_instance()

            # 验证上下文隔离：不同上下文应该有不同的实例
            results["a_same_context"] = at is A.get_instance()
            results["b_same_context"] = bt is B.get_instance()
            results["a_diff_context"] = at is not a1
            results["b_diff_context"] = bt is not b1

        # 创建新上下文并运行
        ctx = contextvars.Context()
        ctx.run(worker)

        # 检查结果
        self.assertTrue(results["a_same_context"])
        self.assertTrue(results["b_same_context"])
        self.assertTrue(results["a_diff_context"])
        self.assertTrue(results["b_diff_context"])

    def test_async_isolation(self):
        class A(SingletonContextIsolation): ...

        class B(A): ...

        async def main():
            import contextvars

            # 在主异步上下文中获取实例
            a1 = A.get_instance()
            a2 = A.get_instance()
            self.assertTrue(a1 is a2)
            b1 = B.get_instance()
            self.assertTrue(b1 is not a1)

            # 结果存储用于从另一个异步任务返回数据
            results = {}

            async def worker():
                # 在新的异步任务中获取实例
                at = A.get_instance()
                bt = B.get_instance()

                # 验证异步任务隔离：不同任务应该有不同的实例
                results["a_same_task"] = at is A.get_instance()
                results["b_same_task"] = bt is B.get_instance()
                results["a_diff_task"] = at is not a1
                results["b_diff_task"] = bt is not b1

            # 创建新任务并运行
            ctx = contextvars.Context()
            task = asyncio.create_task(worker(), context=ctx)
            await task

            # 检查结果
            self.assertTrue(results["a_same_task"])
            self.assertTrue(results["b_same_task"])
            self.assertTrue(results["a_diff_task"])
            self.assertTrue(results["b_diff_task"])

        asyncio.run(main())
