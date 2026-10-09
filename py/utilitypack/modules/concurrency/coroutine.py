from .shared import TaskState
import asyncio
import typing


class TaskBase:
    # 不保证时间精确，只是尽可能立刻地完成
    max_delay: float = 10
    state: TaskState = TaskState.WAITING
    result: typing.Any = None

    def finished(self, gbp: GatheredBatchProcessing) -> bool:
        # 单线程事件循环内，无需加锁
        return self.state in (TaskState.SUCCEEDED, TaskState.FAILED)


class GatheredBatchProcessing:
    """
    协程版本。所有公共方法均为协程，必须在同一个事件循环内使用。
    单线程事件循环天然保证原子性（await 点之间不会被其他协程插入），
    因此不需要锁，锁由事件循环隐式承担。
    """

    ltask_waiting: list[TaskBase]
    batch_size: int

    def __init__(self):
        self.ltask_waiting = []
        self.batch_size = 3
        self._state_changed: asyncio.Condition = asyncio.Condition()

    def swap_queue(self) -> list[TaskBase]:
        ltask_waiting = self.ltask_waiting
        self.ltask_waiting = []
        return ltask_waiting

    def batch_process(self, ltask: list[TaskBase]) -> None:
        """
        批处理实现。禁止操作gbp的内部状态。
        不可手动调用。在gbp自动调用此方法时确保已经在持有事件循环（未被await中断的同步代码段内）。
        批处理本身可能是耗时的阻塞/同步操作，由 to_thread 在线程中执行。
        """

    async def _batch_process_safe_checked(self, ltask: list[TaskBase]) -> None:
        # 保证批处理空队列无成本
        if len(ltask) == 0:
            return
        try:
            # 实际执行批处理时用 to_thread 创建，避免阻塞事件循环
            await asyncio.to_thread(self.batch_process, ltask)
        except Exception as e:
            for t in ltask:
                t.state = TaskState.FAILED
                t.result = e
        # fallback: 保证批处理实现正确更新任务状态
        for t in ltask:
            if not t.finished(self):
                t.state = TaskState.SUCCEEDED
        async with self._state_changed:
            self._state_changed.notify_all()

    async def submit(self, task: TaskBase):
        # 挂起直到任务完成
        if task.finished(self):  # 避免重复提交
            return
        self.ltask_waiting.append(task)
        if len(self.ltask_waiting) >= self.batch_size:
            ltask_waiting = self.swap_queue()
            await self._batch_process_safe_checked(ltask_waiting)
        else:
            # 等待批处理，或 max_delay 超时后主动强制执行
            ltask_waiting = []
            async with self._state_changed:
                while not task.finished(self):
                    # max_delay不一定精确
                    try:
                        await asyncio.wait_for(
                            self._state_changed.wait(), timeout=task.max_delay
                        )
                    except asyncio.TimeoutError:
                        pass
                    # 任务可能已经被取走，正在待执行。不再强制执行
                    if not any(t is task for t in self.ltask_waiting):
                        continue
                    ltask_waiting = self.swap_queue()
                    break
            # 在条件锁外执行批处理
            if ltask_waiting:
                await self._batch_process_safe_checked(ltask_waiting)
