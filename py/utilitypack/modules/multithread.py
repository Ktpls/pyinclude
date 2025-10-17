from __future__ import annotations
import concurrent.futures
import enum
import threading
import traceback
import typing
import functools
from .misc import FunctionalWrapper, EasyWrapper, Switch, SingletonIsolation
from .time import PreciseSleep

UTS_DEFAULT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor()


class StoppableSomewhat:
    class StrategyRunOnRunning(enum.Enum):
        # ignore = 0
        raise_error = 1
        stop_and_rerun = 2
        skip_and_return = 3

    class StrategyError(enum.Enum):
        ignore = 0
        raise_error = 1
        print_error = 2

    def __init__(
        self,
        strategy_runonrunning: "StoppableThread.StrategyRunOnRunning" = None,
        strategy_error: "StoppableThread.StrategyError" = None,
    ):
        self.strategy_runonrunning = (
            strategy_runonrunning
            if strategy_runonrunning is not None
            else StoppableThread.StrategyRunOnRunning.raise_error
        )
        self.strategy_error = (
            strategy_error
            if strategy_error is not None
            else StoppableThread.StrategyError.raise_error
        )

    def foo(self): ...

    def isRunning(self) -> bool: ...

    @FunctionalWrapper
    def go(self, *args, **kwargs): ...

    def stop(self): ...

    def timeToStop(self) -> bool: ...

    @staticmethod
    @EasyWrapper
    def EasyUse(f, implType=None, **kwStoppableSomewhat):
        """
        wrapper on function, and ready for use
        define function as:
            def f(self:StoppableProcess, [your arg]):
                ...
        so f can know when to stop
        every calling produces an instance
        """
        if implType is None:
            implType = StoppableThread
        # if not issubclass(implType, StoppableSomewhat):
        # sometimes not subclass even implType is passed with StoppableThread
        #     raise NotImplementedError("doesn't work")
        # if implType == StoppableProcess:
        #     raise NotImplementedError("doesn't work")

        class Thread4LongScript(implType):
            def foo(self, *arg, **kw) -> None:
                f(self, *arg, **kw)

        instance: StoppableSomewhat = Thread4LongScript(**kwStoppableSomewhat)

        def newF(*arg, **kw):
            instance.go(*arg, **kw)

        return newF


class StoppableThread(StoppableSomewhat):
    # TODO merge with StoppableSomewhat, cuz StoppableProcess is no longer existed
    """
    derivate from it and override foo()
    """

    class State(enum.Enum):
        running = enum.auto()
        stopping = enum.auto()
        stopped = enum.auto()

    # TODO give option to handle error by user. thats for wtutily system to log error
    def __init__(
        self,
        strategy_runonrunning: "StoppableSomewhat.StrategyRunOnRunning" = None,
        strategy_error: "StoppableSomewhat.StrategyError" = None,
        pool: concurrent.futures.ThreadPoolExecutor = None,
    ) -> None:
        super().__init__(strategy_runonrunning, strategy_error)
        self.state: StoppableThread.State = StoppableThread.State.stopped
        self.pool: concurrent.futures.ThreadPoolExecutor = (
            pool or UTS_DEFAULT_THREAD_POOL
        )
        self.submit = None
        self.result = None

    def foo(self, *arg, **kw) -> None:
        raise NotImplementedError("should never run without overriding foo")

    def isRunning(self) -> bool:
        return self.state == StoppableThread.State.running

    @FunctionalWrapper
    def go(self, *arg, **kw) -> None:
        if self.submit is not None:
            if (
                self.strategy_runonrunning
                == StoppableThread.StrategyRunOnRunning.raise_error
            ):
                raise RuntimeError("already running")
            elif (
                self.strategy_runonrunning
                == StoppableThread.StrategyRunOnRunning.stop_and_rerun
            ):
                self.stop()
            elif (
                self.strategy_runonrunning
                == StoppableThread.StrategyRunOnRunning.skip_and_return
            ):
                return

        def call() -> None:
            """
            wrapper so can call the passed "self"'s foo
            if not, can never know which overwritten foo should be called
            check if self.stopsignal when any place to break
            """
            try:
                self.result = self.foo(*arg, **kw)
            except Exception as e:
                if self.strategy_error == StoppableThread.StrategyError.raise_error:
                    raise e
                elif self.strategy_error == StoppableThread.StrategyError.print_error:
                    traceback.print_exc()
                elif self.strategy_error == StoppableThread.StrategyError.ignore:
                    pass
            finally:
                self.state = StoppableThread.State.stopped

        self.state = StoppableThread.State.running
        self.submit = self.pool.submit(call)

    def _signal_stop(self):
        if self.submit is None:
            return
        if self.state != StoppableThread.State.running:
            return
        self.state = StoppableThread.State.stopping

    def _wait_until_stop(self):
        if self.submit is None:
            return
        self.submit.result()
        # TODO consider do not remove future after finished, so we can get its result
        self.submit = None

    @FunctionalWrapper
    def stop(self) -> None:
        if self.submit is None:
            return
        self._signal_stop()
        self._wait_until_stop()

    @staticmethod
    def stop_multiple(*stoppable_threads: "StoppableThread"):
        for t in stoppable_threads:
            t._signal_stop()
        for t in stoppable_threads:
            t._wait_until_stop()

    def timeToStop(self) -> bool:
        return self.state != StoppableThread.State.running


class Stage:
    # stage is something with t readable
    t: float

    def step(self, dt):
        raise NotImplementedError()


class TimeNonconcernedStage(Stage):
    def step(self, dt):
        pass

    @property
    def t(self):
        return 0


class SyncExecutableManager:
    def __init__(
        self, pool: concurrent.futures.ThreadPoolExecutor, stage: Stage
    ) -> None:
        self.pool = pool
        self.selist: list[SyncExecutable] = []
        self.executionLock = threading.RLock()
        self.stage = stage

    def _SubmitSyncExecutable(self, se: SyncExecutable, *a, **kw):
        def foo():
            try:
                se._ConfirmExecutionPrivilege()
                se.main(*a, **kw)
            finally:
                se.state = se.STATE.stopped
                se._ExitExecution()

        self.executionLock.acquire()
        se.state = se.STATE.running
        se.future = self.pool.submit(foo)
        self.selist.append(se)
        self.executionLock.release()

    def _GiveExecutionPrivilegeToSe(self, se: "SyncExecutable"):
        # consider wait asyncly here and below
        se._GiveExecutionPrivilege()
        if se.future:
            ...

    def step(self):
        # call this on wolf update
        # make sure setting running before submitting. or would be possibly kicked out here
        self.executionLock.acquire()

        self.selist = [
            e for e in self.selist if e.state != SyncExecutable.STATE.stopped
        ]
        for se in self.selist:
            if se.state == SyncExecutable.STATE.waitingCondition:
                # knowing not satisfied, skip waking up
                if se.waitCondition():
                    se.state = SyncExecutable.STATE.running
                    self._GiveExecutionPrivilegeToSe(se)
            elif se.state == SyncExecutable.STATE.running:
                self._GiveExecutionPrivilegeToSe(se)
            elif se.state == SyncExecutable.STATE.stopped:
                pass
            else:
                pass

        self.executionLock.release()


class SyncExecutable:
    # for impl sequential but sync mechanization in async foo
    class STATE(enum.Enum):
        stopped = 0
        running = 1
        waitingCondition = 2

    def __init__(self, sem: SyncExecutableManager) -> None:
        self.sem = sem
        self.cond = threading.Condition(sem.executionLock)
        self.state: SyncExecutable.STATE = self.STATE.stopped
        self.future: typing.Optional[concurrent.futures.Future] = None
        self.waitCondition: typing.Optional[typing.Callable[[], bool]] = None

    # override
    def main(self, **arg):
        raise NotImplementedError("not implemented")

    def _ConfirmExecutionPrivilege(self):
        self.cond.acquire(True)

    def _GiveExecutionPrivilege(self):
        # called by other threads
        self.cond.notify_all()
        self.cond.wait()

    def _GiveAwayExecutionPrivilege(self):
        self.cond.notify_all()
        self.cond.wait()

    def _ExitExecution(self):
        self.cond.notify_all()  # no more sleep, aks sem to get up
        self.cond.release()

    def run(self, *a, **kw):
        if not self.isworking():
            self.sem._SubmitSyncExecutable(self, *a, **kw)
        return self

    # available in main
    def sleepUntil(self, untilWhat, timeout=None):
        overduetime = self.sem.stage.t + timeout if timeout else None

        def untilWhatOrTimeOut():
            return untilWhat() or (overduetime and self.sem.stage.t >= overduetime)

        self.waitCondition = untilWhatOrTimeOut
        self.state = self.STATE.waitingCondition
        while True:
            if untilWhatOrTimeOut():
                break
            self._GiveAwayExecutionPrivilege()
        self.waitCondition = None
        self.state = self.STATE.running

    # available in main
    def sleep(self, delaytime):
        self.sleepUntil(lambda: False, delaytime)

    # available in main
    def stepOneFrame(self):
        # buggy
        self.waitCondition = lambda: True
        self.state = self.STATE.waitingCondition
        self._GiveAwayExecutionPrivilege()
        self.waitCondition = None
        self.state = self.STATE.running

    def isworking(self):
        return self.state != self.STATE.stopped


class Pwm(StoppableThread):
    on_set_on: typing.Callable[[], None] = None
    on_set_off: typing.Callable[[], None] = None
    ratio = 0.0
    period = 1
    precision = 0.1

    def __init__(
        self,
        on_set_on=None,
        on_set_off=None,
        ratio=None,
        period=None,
        precision=None,
        *a,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.on_set_on = on_set_on or self.on_set_on
        self.on_set_off = on_set_off or self.on_set_off
        self.ratio = ratio or self.ratio
        self.period = period or self.period
        self.precision = precision or self.precision

    def foo(self):
        switch = Switch(self.on_set_on, self.on_set_off)
        while not self.timeToStop():
            gamma = self.period * self.ratio
            if gamma < self.precision:
                switch.setTo(False)
                PreciseSleep(self.period)
            elif gamma > self.period - self.precision:
                switch.setTo(True)
                PreciseSleep(self.period)
            else:
                switch.setTo(True)
                PreciseSleep(gamma)
                switch.setTo(False)
                PreciseSleep(self.period - gamma)
        switch.setTo(False)


class ThreadLocalSingleton:
    def __init_subclass__(cls, **kwargs):
        """为每个子类初始化独立的线程局部存储"""
        super().__init_subclass__(**kwargs)
        # 每个子类拥有独立的线程局部变量空间
        cls._thread_local = threading.local()  # [[3]]

    @classmethod
    def summon(cls) -> typing.Self:
        """线程局部实例获取方法"""
        if not cls.__thread_local_singleton_test_val__():
            cls.__thread_local_singleton_create_instance__()
        return cls.__thread_local_singleton_get_val__()

    @classmethod
    def __thread_local_singleton_set_val__(cls, val):
        cls._thread_local.instance = val

    @classmethod
    def __thread_local_singleton_get_val__(cls):
        return cls._thread_local.instance

    @classmethod
    def __thread_local_singleton_test_val__(cls):
        return (
            hasattr(cls._thread_local, "instance")
            and cls.__thread_local_singleton_get_val__()
        )

    @classmethod
    def __thread_local_singleton_create_instance__(cls):
        cls.__thread_local_singleton_set_val__(cls())

    @classmethod
    def __thread_local_singleton_release__(cls):
        del cls._thread_local.instance

    @classmethod
    @EasyWrapper
    def WithMe(f: typing.Callable, cls: ThreadLocalSingleton):
        @functools.wraps(f)
        def f2(*a, **kw):
            if not (
                built_before_this_frame := cls.__thread_local_singleton_test_val__()
            ):
                cls.__thread_local_singleton_create_instance__()
            try:
                return f(*a, **kw)
            except Exception as e:
                raise e
            finally:
                if not built_before_this_frame:
                    cls.__thread_local_singleton_release__()

        return f2


class ReadWriteLock:
    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0  # 当前活跃读线程数
        self._writers = 0  # 当前活跃写线程数（0或1）
        self._pending_writers = 0  # 等待中的写线程数（用于避免写饥饿）

    def acquire_read(self):
        """获取读锁"""
        with self._condition:
            # 如果有写者正在写或有写者在等待，则等待（避免写饥饿）
            while self._writers > 0 or self._pending_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self):
        """释放读锁"""
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                # 如果没有读者了，唤醒一个等待的写者（如果有）
                self._condition.notify_all()  # 或 notify() 也可以，但 notify_all 更安全

    def acquire_write(self):
        """获取写锁"""
        with self._condition:
            self._pending_writers += 1  # 增加等待写者计数
            try:
                # 等待直到没有读者和写者
                while self._readers > 0 or self._writers > 0:
                    self._condition.wait()
                self._writers = 1  # 标记写者进入
            finally:
                self._pending_writers -= 1  # 无论成功与否，减少等待计数

    def release_write(self):
        """释放写锁"""
        with self._condition:
            self._writers = 0
            self._condition.notify_all()  # 唤醒所有等待的读者和写者

    # 辅助类：支持 with 语句
    class ReadLock:
        def __init__(self, rwlock: ReadWriteLock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.acquire_read()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.rwlock.release_read()

    class WriteLock:
        def __init__(self, rwlock: ReadWriteLock):
            self.rwlock = rwlock

        def __enter__(self):
            self.rwlock.acquire_write()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.rwlock.release_write()

    # 支持 with 语句的上下文管理器（可选，方便使用）
    def gen_rlock(self):
        return ReadWriteLock.ReadLock(self)

    def gen_wlock(self):
        return ReadWriteLock.WriteLock(self)


class SingletonThreadIsolation(SingletonIsolation):
    __inst_dict = threading.local()
    __lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with SingletonThreadIsolation.__lock:
            if not hasattr(SingletonThreadIsolation.__inst_dict, "store"):
                SingletonThreadIsolation.__inst_dict.store = dict()
            store: dict = SingletonThreadIsolation.__inst_dict.store
            if cls.__qualname__ not in store:
                store[cls.__qualname__] = cls()
            return store[cls.__qualname__]
