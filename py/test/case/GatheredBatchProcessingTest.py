import asyncio
import concurrent.futures.thread
import dataclasses
import threading
import time
import unittest

from test.case.autotest_common import *
from utilitypack.modules.concurrency import coroutine as gbp_coroutine
from utilitypack.modules.concurrency import thread as gbp_thread
from utilitypack.modules.concurrency.shared import TaskState


class GatheredBatchProcessingTest(unittest.TestCase):

    @dataclasses.dataclass
    class _task:
        # _task, for make it clear to unittest lib that this is not a test case(not startswith('test'))
        content: str
        max_delay: float = 10
        state: TaskState = TaskState.WAITING

        def finished(self, gbp) -> bool:
            # 任务状态由单一字段决定，直接检查即可
            return self.state in (TaskState.SUCCEEDED, TaskState.FAILED)

    def _make_task(selfTest, content, max_delay=10):
        return GatheredBatchProcessingTest._task(content=content, max_delay=max_delay)

    # ---------- thread version ----------

    def test_thread_batchFillTriggersProcessing(selfTest):
        batches = list()

        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                batches.append([t.content for t in ltask])
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        gbp = Gbp()
        pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=10)
        ltask = [
            pool.submit(gbp.submit, selfTest._make_task(f"task{i}")) for i in range(3)
        ]
        for t in concurrent.futures.as_completed(ltask):
            t.result()
        selfTest.assertEqual(len(batches), 1)
        selfTest.assertEqual(sorted(batches[0]), ["task0", "task1", "task2"])
        selfTest.assertEqual(gbp.ltask_waiting, [])

    def test_thread_timeoutForcesExecution(selfTest):
        processed = list()

        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                processed.extend(t.content for t in ltask)
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        gbp = Gbp()
        t0 = time.time()
        # 单任务，不足一个 batch，等待 max_delay 超时后被强制执行
        gbp.submit(selfTest._make_task("lonely", max_delay=0.2))
        selfTest.assertLessEqual(time.time() - t0, 5)
        selfTest.assertEqual(processed, ["lonely"])

    def test_thread_batchProcessRunsHoldingTaskStateLock(selfTest):
        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                # 约定：自动调用 batch_process 时已持有任务状态/批处理执行锁
                selfTest.assertTrue(gbp.task_state_lock._is_owned())
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        gbp = Gbp()
        gbp.submit(selfTest._make_task("lock", max_delay=0.2))

    def test_thread_manyConcurrentTasks(selfTest):
        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        gbp = Gbp()
        pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=20)
        ltask = [
            pool.submit(gbp.submit, selfTest._make_task(f"t{i}", max_delay=0.5))
            for i in range(20)
        ]
        for t in concurrent.futures.as_completed(ltask):
            t.result()
        selfTest.assertTrue(all(t.done() and t.exception() is None for t in ltask))

    def test_thread_failureMarksFailed(selfTest):
        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                raise ValueError("boom")

        gbp = Gbp()
        task = selfTest._make_task("bad", max_delay=0.2)
        gbp.submit(task)
        selfTest.assertEqual(task.state, TaskState.FAILED)
        selfTest.assertIsInstance(task.result, ValueError)
        selfTest.assertTrue(task.finished(gbp))

    def test_thread_fallbackMarksSucceeded(selfTest):
        # 批处理实现忘记更新任务状态时，fallback 保证任务被标记为 SUCCEEDED
        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                pass

        gbp = Gbp()
        task = selfTest._make_task("lazy", max_delay=0.2)
        gbp.submit(task)
        selfTest.assertEqual(task.state, TaskState.SUCCEEDED)

    def test_thread_noDuplicateSubmit(selfTest):
        processed = list()

        class Gbp(gbp_thread.GatheredBatchProcessing):
            def batch_process(self, ltask):
                processed.extend(t.content for t in ltask)
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        gbp = Gbp()
        task = selfTest._make_task("dup", max_delay=0.2)
        gbp.submit(task)
        selfTest.assertEqual(processed, ["dup"])
        # 已完成的任务再次提交，直接返回，不重复处理
        gbp.submit(task)
        selfTest.assertEqual(processed, ["dup"])

    # ---------- coroutine version ----------

    def test_coroutine_batchFillTriggersProcessing(selfTest):
        batches = list()

        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                batches.append([t.content for t in ltask])
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        async def amain():
            gbp = Gbp()

            async def submit_one(content):
                await gbp.submit(selfTest._make_task(content))

            await asyncio.gather(*(submit_one(f"task{i}") for i in range(3)))
            selfTest.assertEqual(gbp.ltask_waiting, [])

        asyncio.run(amain())
        selfTest.assertEqual(len(batches), 1)
        selfTest.assertEqual(sorted(batches[0]), ["task0", "task1", "task2"])

    def test_coroutine_batchProcessRunsInWorkerThread(selfTest):
        thread_names = list()

        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                thread_names.append(threading.current_thread().name)
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        async def amain():
            gbp = Gbp()
            await gbp.submit(selfTest._make_task("inthr", max_delay=0.2))

        asyncio.run(amain())
        selfTest.assertEqual(len(thread_names), 1)
        selfTest.assertNotEqual(thread_names[0], threading.main_thread().name)

    def test_coroutine_timeoutForcesExecution(selfTest):
        processed = list()

        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                processed.extend(t.content for t in ltask)
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        async def amain():
            gbp = Gbp()
            t0 = time.time()
            await gbp.submit(selfTest._make_task("lonely", max_delay=0.2))
            selfTest.assertLessEqual(time.time() - t0, 5)

        asyncio.run(amain())

    def test_coroutine_failureMarksFailed(selfTest):
        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                raise ValueError("boom")

        async def amain():
            gbp = Gbp()
            task = selfTest._make_task("bad", max_delay=0.2)
            await gbp.submit(task)
            selfTest.assertEqual(task.state, TaskState.FAILED)
            selfTest.assertIsInstance(task.result, ValueError)
            selfTest.assertTrue(task.finished(gbp))

        asyncio.run(amain())

    def test_coroutine_fallbackMarksSucceeded(selfTest):
        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                pass

        async def amain():
            gbp = Gbp()
            task = selfTest._make_task("lazy", max_delay=0.2)
            await gbp.submit(task)
            selfTest.assertEqual(task.state, TaskState.SUCCEEDED)

        asyncio.run(amain())

    def test_coroutine_noDuplicateSubmit(selfTest):
        processed = list()

        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                processed.extend(t.content for t in ltask)
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        async def amain():
            gbp = Gbp()
            task = selfTest._make_task("dup", max_delay=0.2)
            await gbp.submit(task)
            selfTest.assertEqual(processed, ["dup"])
            await gbp.submit(task)
            selfTest.assertEqual(processed, ["dup"])

        asyncio.run(amain())

    def test_coroutine_earlierTaskFinishedByLaterSubmission(selfTest):
        processed = list()

        class Gbp(gbp_coroutine.GatheredBatchProcessing):
            def batch_process(self, ltask):
                processed.append([t.content for t in ltask])
                for t in ltask:
                    t.state = TaskState.SUCCEEDED

        async def amain():
            gbp = Gbp()

            async def submit_one(content):
                await gbp.submit(selfTest._make_task(content, max_delay=5))

            first = asyncio.create_task(submit_one("first"))
            await asyncio.sleep(0.05)
            second = asyncio.create_task(submit_one("second"))
            # second 超时等待后强制执行时会把还在队列中的 first 一并取走
            await asyncio.gather(first, second)

        asyncio.run(amain())
        selfTest.assertEqual(len(processed), 1)
        selfTest.assertEqual(sorted(processed[0]), ["first", "second"])
