import concurrent.futures
import enum
import threading
import traceback
from .misc import *
from .time import *

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

    # TODO give option to handle error by user. thats for wtutily system to log error
    def __init__(
        self,
        strategy_runonrunning: "StoppableSomewhat.StrategyRunOnRunning" = None,
        strategy_error: "StoppableSomewhat.StrategyError" = None,
        pool: concurrent.futures.ThreadPoolExecutor = None,
    ) -> None:
        super().__init__(strategy_runonrunning, strategy_error)
        self.running: bool = False
        self.stopsignal: bool = True
        self.pool: concurrent.futures.ThreadPoolExecutor = (
            pool or UTS_DEFAULT_THREAD_POOL
        )
        self.submit = None
        self.result = None

    def foo(self, *arg, **kw) -> None:
        raise NotImplementedError("should never run without overriding foo")

    def isRunning(self) -> bool:
        return self.running

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
        self.running = True
        self.stopsignal = False

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
                self.running = False

        self.submit = self.pool.submit(call)

    def _signal_stop(self):
        if self.submit is None:
            return
        self.stopsignal = True

    def _wait_until_stop(self):
        if self.submit is None:
            return
        self.submit.result()
        self.running = False
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
        return self.stopsignal


class Stage:
    def step(self, dt):
        raise NotImplementedError("")

    @property
    def t(self):
        raise NotImplementedError("")


class TimeNonconcernedStage(Stage):
    def step(self, dt):
        pass

    @property
    def t(self):
        return 0


class SyncExecutableManager:
    def __init__(self, pool: concurrent.futures.ThreadPoolExecutor) -> None:
        self.pool = pool
        self.selist: list[SyncExecutable] = []
        self.executionLock = threading.RLock()

    def _GiveExecutionPrivilegeToSe(self, se: "SyncExecutable"):
        # consider wait asyncly here and below
        se.cond.notify_all()
        se.cond.wait()

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
                    self._GiveExecutionPrivilegeToSe(se)
            elif se.state == SyncExecutable.STATE.running:
                self._GiveExecutionPrivilegeToSe(se)
            elif se.state == SyncExecutable.STATE.stopped:
                pass
            else:
                pass

        self.executionLock.release()


class SyncExecutable:
    # for impl serialized but sync mechanization in async foo
    # stage is something with t readable
    class STATE(enum.Enum):
        stopped = 0
        running = 1
        waitingCondition = 2

    def __init__(
        self, stage: Stage, sem: SyncExecutableManager, raiseOnErr=True
    ) -> None:
        self.stage = stage
        self.sem = sem
        self.cond = threading.Condition(sem.executionLock)
        self.state = self.STATE.stopped
        self.future = None
        self.raiseOnErr = raiseOnErr
        self.waitCondition = None

    # override
    def main(self, **arg):
        raise NotImplementedError("not implemented")

    def _ConfirmExecutionPrivilege(self):
        self.cond.acquire(True)

    def _GiveAwayExecutionPrivilege(self):
        self.cond.notify_all()
        self.cond.wait()

    def _ExitExecution(self):
        self.cond.notify_all()  # no more sleep, aks sem to get up
        self.cond.release()

    def run(self, *a, **kw):
        def foo():
            try:
                self._ConfirmExecutionPrivilege()
                self.main(*a, **kw)
            except Exception as e:
                traceback.print_exc()
                if self.raiseOnErr:
                    raise e
            finally:
                self.state = self.STATE.stopped
                self._ExitExecution()

        if not self.isworking():
            self.sem.executionLock.acquire()
            self.state = self.STATE.running
            self.future = self.sem.pool.submit(foo)
            self.sem.selist.append(self)
            self.sem.executionLock.release()
        return self

    # available in main
    def sleepUntil(self, untilWhat, timeout=None):
        overduetime = self.stage.t + timeout if timeout else None

        def untilWhatOrTimeOut():
            return untilWhat() or (overduetime and self.stage.t >= overduetime)

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
