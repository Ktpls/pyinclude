from .shared import TaskState
import threading
import typing


class TaskBase:
    # 不保证时间精确，只是尽可能立刻地完成
    max_delay: float = 10
    state: TaskState = TaskState.WAITING
    thread: threading.Thread = None
    result: typing.Any = None

    def finished(self, gbp: GatheredBatchProcessing):
        with gbp.task_state_lock:
            return self.state in (TaskState.SUCCEEDED, TaskState.FAILED)


class GatheredBatchProcessing:
    queue_lock: threading.Lock
    task_state_lock: threading.Lock
    batch_finish_condition: threading.Condition
    ltask_waiting: list[TaskBase]
    batch_size: int

    def __init__(self):
        # 保护队列，但不保护里面的元素
        self.queue_lock = threading.RLock()
        # 保护所有任务元素，不论在不在队列。也保护批处理执行过程
        self.task_state_lock = threading.RLock()
        self.batch_finish_condition = threading.Condition(self.task_state_lock)
        self.ltask_waiting = []
        self.batch_size = 3

    def swap_queue(self):
        with self.queue_lock:
            ltask_waiting = self.ltask_waiting
            self.ltask_waiting = []
        return ltask_waiting

    def batch_process(self, ltask: list[TaskBase]) -> None:
        """
        批处理实现。
        禁止操作gbp的内部状态
        不可手动调用。在gbp自动执行此方法时确保已经持有任务状态/批处理执行锁
        """

    def _batch_process_safe_checked(self, ltask: list[TaskBase]) -> None:
        with self.task_state_lock:
            # 保证批处理空队列无成本
            if len(ltask) == 0:
                return
            try:
                self.batch_process(ltask)
            except Exception as e:
                for t in ltask:
                    t.state = TaskState.FAILED
                    t.result = e
            # fallback: 保证批处理实现正确更新任务状态
            for t in ltask:
                if not t.finished(self):
                    t.state = TaskState.SUCCEEDED
            self.batch_finish_condition.notify_all()

    def submit(self, task: TaskBase):
        # 阻塞直到任务完成
        if task.finished(self):  # 避免重复提交
            return
        queue_full = False
        with self.queue_lock:
            self.ltask_waiting.append(task)
            if len(self.ltask_waiting) >= self.batch_size:
                ltask_waiting = self.swap_queue()
                queue_full = True
        if queue_full:
            # 首先释放队列锁，再取执行锁，避免死锁
            self._batch_process_safe_checked(ltask_waiting)
        else:
            with self.batch_finish_condition:
                while not task.finished(self):
                    # max_delay不一定精确
                    self.batch_finish_condition.wait(timeout=task.max_delay)
                    with self.queue_lock:
                        # 任务可能已经被取走，正在待执行。不再强制执行
                        if not any(t is task for t in self.ltask_waiting):
                            continue
                        ltask_waiting = self.swap_queue()
                    self._batch_process_safe_checked(ltask_waiting)
