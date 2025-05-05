import collections
import concurrent.futures
import copy
import ctypes
import dataclasses
import datetime
import enum
import functools
import heapq
import inspect
import itertools
import json
import logging
import math
import multiprocessing
import os
import pprint
import queue
import random
import re
import sys
import threading
import time
import traceback
import types
import typing
import uuid
import zipfile

"""
solid
"""
EPS = 1e-10


def DictEq(a: typing.Dict, b: typing.Dict):
    if len(a) != len(b):
        return False
    for k in a.keys():
        if k not in b.keys():
            return False
        if a[k] != b[k]:
            return False
    return True


def ListEq(a: list, b: list):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def FloatEq(a, b, eps=1e-6):
    return abs(a - b) < eps


def IdentityMapping(x):
    return x


def BetterGroupBy(l: list, pred):
    # return {n: list(ll) for n, ll in itertools.groupby(sorted(l, key=pred), pred)}
    r = dict()
    for item in l:
        key = pred(item)
        if key in r:
            r[key].append(item)
        else:
            r[key] = [item]
    return r


def Deduplicate(l: list, key=None):
    key = Coalesce(key, IdentityMapping)
    m = BetterGroupBy(l, key)
    l = [v[0] for v in m.values()]
    return l


def ArrayFlatten(iterable, iterableType: tuple[type] = (list, tuple)):
    result = list()
    for item in iterable:
        if isinstance(item, iterableType):
            result.extend(ArrayFlatten(item))
        else:
            result.append(item)
    return result


def Digitsof(s: str):
    return "".join(list(filter(str.isdigit, list(s))))


def Numinstr(s: str):
    # wont consider negative
    s = Digitsof(s)
    return int(s) if len(s) > 0 else 0


def FunctionalWrapper(f: typing.Callable) -> typing.Callable:
    @functools.wraps(f)
    def f2(self, *args, **kwargs):
        f(self, *args, **kwargs)
        return self

    return f2


def EasyWrapper(wrapperLogic=None):
    """
    use like this
        @EasyWrapper()
        def yourWrapper(f, some_arg_your_wrapper_needs):
            ...
        @yourWrapper(some_arg_your_wrapper_needs)
        def foo(func_arg):
            ...
    or
        @yourWrapper # if no arg for yourWrapper
        def foo(func_arg):
            ...
    note that this is forbiden:
        @someClassInstance.methodDecorator
        def foo(...): ...
            cuz wrapper will recieve the instance as the first arg, and the foo as the second
            making easywrapper confused with wrapping a class with a method as arg
            use this instead
                @someClassInstance.methodDecorator()
                def foo(func_arg): .
    somehow buggy but works almost fine
    ###############
    note that python design is piece of shit
    ###############

    known issue:

    """

    def toGetWrapperLogic(wrapperLogic):
        def newWrapper(*arg, **kw):
            def toGetFuncLogic(funcLogic):
                return wrapperLogic(funcLogic, *arg, **kw)

            if (
                len(arg) == 1
                and len(kw) == 0
                and (inspect.isfunction(arg[0]) or inspect.isclass(arg[0]))
                and [
                    1
                    for k, v in inspect.signature(wrapperLogic).parameters.items()
                    if (
                        v.default == inspect.Parameter.empty
                        and v.kind
                        not in (
                            inspect.Parameter.VAR_POSITIONAL,  # *a
                            inspect.Parameter.VAR_KEYWORD,  # **kw
                        )
                    )
                ].__len__()
                == 1
            ):
                # to find if its possible to call without parens
                return wrapperLogic(arg[0])
            else:
                # calling without parens
                return toGetFuncLogic

        return newWrapper

    if wrapperLogic is not None:
        return toGetWrapperLogic(wrapperLogic)
    else:
        # calling without parens
        return toGetWrapperLogic


class BulletinBoard:
    @dataclasses.dataclass
    class Poster:
        content: str
        starttime: float
        timeout: float
        overduetime: float

        def __init__(self, content, timeout=10):
            self.content = content
            self.starttime = time.perf_counter()
            self.timeout = timeout
            self.overduetime = time.perf_counter() + timeout

    def __init__(self, idlecontent):
        self.idlecontent = idlecontent
        self.content: list["BulletinBoard.Poster"] = []

    def putup(self, poster: typing.Union[Poster, str]):
        if type(poster) == str:
            poster = BulletinBoard.Poster(poster)
        self.content.append(poster)

    def read(self):
        self.content = [c for c in self.content if c.overduetime > time.perf_counter()]
        rctt = list(range(len(self.content)))
        rctt.reverse()
        if len(self.content):
            return ("\n" + "-" * 10 + "\n").join(
                [self.content[c].content for c in rctt]
            )
        else:
            return self.idlecontent


def AllFileIn(
    path, includeFileInSubDir=True, path_filter: typing.Callable[[str], bool] = None
):
    ret = []
    for dirpath, dir, file in os.walk(path):
        if not includeFileInSubDir and dirpath != path:
            continue
        fullPath = [os.path.join(dirpath, f) for f in file]
        if path_filter is not None:
            fullPath = list(filter(path_filter, fullPath))
        ret.extend(fullPath)
    return ret


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
        self.pool: concurrent.futures.ThreadPoolExecutor = Coalesce(
            pool, UTS_DEFAULT_THREAD_POOL
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


def ReadFile(path):
    with open(path, "rb") as f:
        return f.read()


def EnsureDirectoryExists(directory):
    if len(directory) == 0:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)


def EnsureFileDirExists(path):
    EnsureDirectoryExists(os.path.dirname(path))


def WriteFile(path, content):
    EnsureFileDirExists(path)
    with open(path, "wb+") as f:
        f.write(content)


def AppendFile(path, content):
    EnsureFileDirExists(path)
    with open(path, "ab+") as f:
        f.write(content.encode("utf-8"))


def ReadTextFile(path: str) -> str:
    return ReadFile(path).decode("utf-8")


def WriteTextFile(path: str, text: str):
    WriteFile(path, text.encode("utf-8"))


def GetTimeString():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


class Progress:
    """
    irreversable!
    """

    def __init__(self, total: float, cur=0, printPercentageStep: float = 0.1) -> None:
        self.total = total
        self.nowStage = 0
        self.printPercentageStep = printPercentageStep
        self.cur = cur
        self.ps = perf_statistic(startnow=True)

    def update(self, current: float) -> None:
        self.cur = current
        while True:
            if current / self.total > self.nowStage * self.printPercentageStep:
                self.nowStage += 1
                if current > 1:
                    self.ps.countcycle()
                    # not the first time
                    instantSpeed = (self.printPercentageStep * self.total) / (
                        self.ps.aveTime() + EPS
                    )
                else:
                    instantSpeed = 1
                print(
                    f"{100 * current / self.total:>3.2f}% of {self.total}, {instantSpeed:.2f}it/s",
                    end="\r",
                )
            else:
                break

    def setFinish(self):
        self.ps.stop()
        self.update(self.total)
        print("")
        print(f"finished within {self.ps.time():.2f}s")


def SleepUntil(con: typing.Callable, dt=None, sleepImpl=None):
    if sleepImpl is None:
        sleepImpl = time.sleep
    if dt is None:
        dt = 0.025
    while not con():
        sleepImpl(dt)


class SingleSectionedTimer:
    """
    basic timer, allows only start and stop with one section
    one could simply use start() as restart function without clear() and start()
    """

    def timeCounter(self):
        return time.perf_counter()

    def clear(self):
        self._starttime = None
        return self

    def start(self):
        self._starttime = (self.timeCounter)()
        return self

    def isRunning(self):
        return self._starttime is not None

    def get(self) -> float:
        return (self.timeCounter)() - self._starttime if self.isRunning() else 0

    def getAndRestart(self) -> float:
        v = self.get()
        self.start()
        return v

    def __init__(self, startNow=False):
        self.clear()
        if startNow:
            self.start()


class perf_statistic:
    """
    providing rich timer functions, like multicycle counting, performance profiling
    calculate the time past between start() to now, directly by perf_counter()-starttime
    record all accumulated time before start(), but uncleared after stop()
    so start and stop are also playing roles as resume and pause
    countcycle() will increase the cycle count, helping to calculate average time in a loop-like task
    clear() will clear all accumulated time, stops counting
    """

    def __init__(self, startnow=False, enable_time_detail=False):
        self._singled = SingleSectionedTimer()
        self.enable_time_detail = enable_time_detail
        self.clear()
        if startnow:
            self.start()

    def clear(self):
        self._singled.clear()
        self._stagedTime = 0
        self._cycle = 0
        self._stagedTimeList = list()
        return self

    def start(self):
        self._singled.start()
        return self

    def countcycle(self):
        self._cycle += 1
        return self

    def stop(self):
        if self.isRunning():
            timeThisRound = self._singled.get()
            self._singled.clear()
            self._stagedTime += timeThisRound
            if self.enable_time_detail:
                self._stagedTimeList.append(timeThisRound)
        return self

    def isRunning(self):
        return self._singled.isRunning()

    def time(self):
        return self._stagedTime + self._singled.get()

    def aveTime(self):
        return self.time() / (self._cycle if self._cycle > 0 else 1)

    @dataclasses.dataclass
    class SectionCounter:
        ps: "perf_statistic"
        clearOnExit: bool = False

        def __enter__(self):
            self.ps.start()

        def __exit__(self, exc_type, exc_value, traceback):
            self.ps.stop()
            if self.clearOnExit:
                self.ps.clear()


class FpsManager:
    def __init__(self, fps=60):
        self.lt = time.perf_counter()
        self.frametime = 1 / fps

    def WaitUntilNextFrame(self, sleepImpl=None):
        SleepUntil(
            lambda: time.perf_counter() - self.lt > self.frametime,
            dt=0.5 * self.frametime,
            sleepImpl=sleepImpl,
        )
        self.SetToNextFrame()

    def CheckIfTimeToDoNextFrame(self) -> bool:
        """
        usage
        if fpsmanager.CheckIfTimeToDoNextFrame():
            fpsmanager.SetToNextFrame()
            do your task here
        used in doing stuff peroidically, but in another loop with different peroid
        so have to check if it is time to do it
        """
        result = time.perf_counter() - self.lt > self.frametime
        return result

    def SetToNextFrame(self):
        self.lt = time.perf_counter()


def CostlySleep(t):
    endtime = time.perf_counter() + t
    while True:
        if time.perf_counter() >= endtime:
            break


def PreciseSleep(t):
    # to stop oscilation in autoCali due to sleep() precise
    # to my suprise time.sleep() is quite precise
    if t > 0.0005773010000120849:
        # too rough
        time.sleep(t)
    else:
        CostlySleep(t)


@dataclasses.dataclass
class PIDController:
    @dataclasses.dataclass
    class AnalizerFrameData:
        partp: float
        parti: float
        partd: float
        error: float
        integral: float
        derivative: float

    kp: float = 0
    ki: float = 0
    kd: float = 0
    integralLimitMin: float = None
    integralLimitMax: float = None
    analizerMode: bool = False
    last_error: float = dataclasses.field(default=0, init=False)
    integral: float = dataclasses.field(default=0, init=False)
    frameData: AnalizerFrameData = dataclasses.field(default=None, init=False)

    def update(self, error, dt=1):
        self.integral += error * dt
        if self.integralLimitMax is not None and self.integral > self.integralLimitMax:
            self.integral = self.integralLimitMax
        if self.integralLimitMin is not None and self.integral < self.integralLimitMin:
            self.integral = self.integralLimitMin
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        if self.analizerMode:
            self.frameData = PIDController.AnalizerFrameData(
                partp=self.kp * error,
                parti=self.ki * self.integral,
                partd=self.kd * derivative,
                error=error,
                integral=self.integral,
                derivative=derivative,
            )
        return output


class OneOrderLinearFilter:
    def __init__(self, N, initial_val=None):
        self.a = N / (N + 1)
        self.b = 1 / (N + 1)
        self.previous_output = initial_val if initial_val else 0

    def update(self, input_sample):
        output_sample = self.a * self.previous_output + self.b * input_sample
        self.previous_output = output_sample
        return output_sample

    def get(self):
        return self.previous_output


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


class AccessibleQueue:
    def Annotation(T):
        return list[T]

    class AQException(Exception):
        pass

    def __init__(self, maxsize=1):
        self._maxsize = maxsize
        self.clear()

    def push(self, val):
        if self.isFull():
            raise AccessibleQueue.AQException("queue full")
        self._q[self._eptr] = val
        self._eptr += 1
        self._eptr %= self._maxsize
        self._cursize += 1

    def pop(self):
        if self.isEmpty():
            raise AccessibleQueue.AQException("queue empty")
        val = self._q[self._sptr]
        self._sptr += 1
        self._sptr %= self._maxsize
        self._cursize -= 1
        return val

    def push__pop_if_full(self, val):
        if self.isFull():
            self.pop()
        self.push(val)

    def clear(self):
        self._q = [None] * self._maxsize
        self._sptr = 0
        self._eptr = 0
        self._cursize = 0

    def resize(self, newsize):
        if newsize < self._cursize:
            raise AccessibleQueue.AQException("newsize too small")
        # guess resizing list will realloc anyway
        l2 = list([None] * newsize)
        for i in range(self._cursize):
            l2[i] = self[i]
        self._q = l2

        self._maxsize = newsize

    def __indexMapping(self, i):
        if i < 0:
            i = self._cursize + i
        return (i + self._sptr) % self._maxsize

    def __len__(self):
        return self._cursize

    def isFull(self):
        return self._cursize == self._maxsize

    def isEmpty(self):
        return self._cursize == 0

    def __getitem__(self, i: int | slice):
        if isinstance(i, int):
            if i >= self._cursize:
                raise AccessibleQueue.AQException("index out of range")
            return self._q[self.__indexMapping(i)]
        elif isinstance(i, slice):
            return [
                self[j]
                for j in range(
                    Coalesce(i.start, 0),
                    Coalesce(i.stop, self._cursize),
                    Coalesce(i.step, 1),
                )
            ]

    def ToList(self):
        return [self[i] for i in range(len(self))]


class BeanUtil:
    @dataclasses.dataclass
    class CopyOption:
        ignoreNoneInSrc: bool = True
        recursive: bool = True

    """
    have to deal with dict, object, and class(only on dest)
    """

    @staticmethod
    def _GetClassDeclFields(clz):
        parents = NormalizeIterableOrSingleArgToIterable(clz.__base__)
        result = dict()
        for p in parents:
            if p == object:
                continue
            result.update(BeanUtil._GetClassDeclFields(p))
        if hasattr(clz, "__annotations__"):
            # override
            result.update(clz.__annotations__)
        return result

    @staticmethod
    def _GetClassInstanceFields(inst):
        staticFields = BeanUtil._GetClassDeclFields(type(inst))
        dynamicFields = {k: type(v) for k, v in inst.__dict__.items()}
        return {**dynamicFields, **staticFields}

    @staticmethod
    def _GetEmptyInstanceOfClass(cls):
        if cls == int:
            return 0
        if cls == str:
            return ""
        if cls == bool:
            return False
        if cls == float:
            return 0.0
        if cls in (dict, list, tuple, set):
            return cls()
        if hasattr(cls, "__init__") and hasattr(cls.__init__, "__code__"):
            args = inspect.getargs(cls.__init__.__code__)
            if len(args) > 1:
                # found init with arg more than self
                inst = object.__new__(cls)
                fields = BeanUtil._GetClassDeclFields(cls)
                for name, taipe in fields.items():
                    """
                    for taipe as class, its possible to recursively call GetEmptyInstance
                    but taipe could be str, or typing.GenericAlias
                    """
                    setattr(inst, name, None)
                return inst
        return cls()

    class _TypeResolution:
        def __init__(self, taipe):
            self.taipe = taipe

        def getPlainType(self):
            # ensure not GenericAlias
            if isinstance(self.taipe, types.GenericAlias):
                return self.taipe.__origin__
            else:
                return self.taipe

        def getChild(self, key) -> "BeanUtil._TypeResolution":
            taipe = None
            if isinstance(self.taipe, dict):
                taipe = None
            elif self.taipe is None or isinstance(self.taipe, str):
                # not type annotated, or annotated like field:"some class" so i cant process
                taipe = None
            elif hasattr(self.taipe, "__annotations__"):
                taipe = self.taipe.__annotations__.get(key, None)
            elif isinstance(self.taipe, types.GenericAlias):
                # typing.GenericAlias, or <container>[<element>] like
                if BeanUtil._isFlatCollection(self.taipe.__origin__) and isinstance(
                    key, int
                ):
                    if hasattr(self.taipe, "__args__"):
                        taipe = self.taipe.__args__[0]
                elif self.taipe.__origin__ in (dict):
                    taipe = self.taipe.__args__[1]

            return BeanUtil._TypeResolution(taipe)

        def getType(self):
            return self.taipe

    @staticmethod
    def _PrimaryTypeConversionFunc(taipe, obj):
        if taipe is None:
            return obj
        try:
            return taipe(obj)
        except:
            return obj

    @staticmethod
    def ObjectSerializer(o):
        # difference between this and toMap() is
        # this should always return a serializable,
        # while toMap() may give up further conversion

        # try parse with dict
        if hasattr(o, "__dict__"):
            member = {k: v for k, v in o.__dict__.items() if not str.startswith(k, "_")}
            if len(member) != 0:
                return member
        # try repr
        if hasattr(o, "__repr__"):
            try:
                return o.__repr__()
            except:
                pass
        return str(o)

    @staticmethod
    def ReprObject(o):

        return json.dumps(
            o,
            indent=4,
            ensure_ascii=False,
            default=BeanUtil.ObjectSerializer,
            sort_keys=True,
        )

    @staticmethod
    def _isPrimaryType(t: type):
        return t in (int, float, str, bool, type, type(None)) or BeanUtil.isEnum(t)

    @staticmethod
    def isEnum(t):
        return issubclass(t, enum.Enum)

    @staticmethod
    def _isFlatCollection(t: type):
        return t in (list, tuple)

    @staticmethod
    def _isCustomStructure(t: type):
        return not BeanUtil._isPrimaryType(t) and not BeanUtil._isFlatCollection(t)

    @dataclasses.dataclass
    class CompatibleStructureOperator:
        obj: typing.Any
        objtype: type = None
        objtypeRes: "BeanUtil._TypeResolution" = None
        fieldType: dict[str, type] | type = None

        def __post_init__(self):
            self.objtype = Coalesce(self.objtype, type(self.obj))
            self.objtypeRes = BeanUtil._TypeResolution(self.objtype)

        def _init_type_info(self):
            if self.fieldType is None:
                instFields = dict()
                if self.objtypeRes.getPlainType() in (list, tuple):
                    instFields = (
                        BeanUtil._TypeResolution(self.objtype)
                        .getChild(0)
                        .getPlainType()
                    )
                elif self.objtypeRes.getPlainType() == dict:
                    instFields = (
                        BeanUtil._TypeResolution(self.objtype)
                        .getChild(None)
                        .getPlainType()
                    )
                else:
                    instFields = BeanUtil._GetClassInstanceFields(self.obj)
                self.fieldType = instFields

        def typeInfo(self, key):
            self._init_type_info()
            if isinstance(self.fieldType, dict):
                return self.fieldType.get(key, None)
            else:
                return self.fieldType

        def getter(self):
            src = self.obj
            if BeanUtil._isFlatCollection(self.objtypeRes.getPlainType()):
                ret = enumerate(src)
            elif self.objtypeRes.getPlainType() == dict:
                ret = src.items()
            else:
                ret = src.__dict__.items()
            return ret

        def isSettable(self, key):
            if self.objtypeRes.getPlainType() == list:
                return True
            elif self.objtypeRes.getPlainType() == dict:
                return True
            elif self.objtypeRes.getPlainType() == tuple:
                return False
            else:
                self._init_type_info()
                return key in self.fieldType

        def set(self, key, value):
            if self.objtypeRes.getPlainType() == list:

                def ListSetter(obj: list, k, v):
                    if k >= len(obj):
                        obj.extend([None] * (k - len(obj) + 1))
                    obj[k] = v

                ListSetter(self.obj, key, value)
            elif self.objtypeRes.getPlainType() == dict:

                def DictSetter(obj, k, v):
                    obj[k] = v

                DictSetter(self.obj, key, value)
            elif self.objtypeRes.getPlainType() == tuple:
                raise ValueError("can not copy to tuple")
            else:
                self._init_type_info()

                def ObjSetter(obj, k, v):
                    if k in self.fieldType:
                        # try convert it to proper type
                        v = BeanUtil._PrimaryTypeConversionFunc(self.typeInfo(k), v)
                        obj.__setattr__(k, v)

                ObjSetter(self.obj, key, value)

    @staticmethod
    def copyProperties(
        src,
        dst: object,
        option: "BeanUtil.CopyOption" = CopyOption(),
    ):
        srcType = BeanUtil._TypeResolution(type(src))
        if isinstance(dst, types.GenericAlias):
            dstType = BeanUtil._TypeResolution(dst)
            dst = BeanUtil._GetEmptyInstanceOfClass(dst.__origin__)
        elif inspect.isclass(dst):
            dstType = BeanUtil._TypeResolution(dst)
            dst = BeanUtil._GetEmptyInstanceOfClass(dst)
        else:
            dstType = BeanUtil._TypeResolution(type(dst))

        srcOp = BeanUtil.CompatibleStructureOperator(src, srcType.getType())
        dstOp = BeanUtil.CompatibleStructureOperator(dst, dstType.getType())
        if BeanUtil._isPrimaryType(srcType.getPlainType()) or BeanUtil._isPrimaryType(
            dstType.getPlainType()
        ):
            return BeanUtil._PrimaryTypeConversionFunc(dstType.getPlainType(), src)
        if (
            srcType.getPlainType() is not None
            and dstType.getPlainType() is not None
            and BeanUtil._isFlatCollection(srcType.getPlainType())
            != BeanUtil._isFlatCollection(dstType.getPlainType())
        ):
            raise ValueError("src and dst must be or not be array the same time")
        for k, v in srcOp.getter():
            if option.ignoreNoneInSrc and v is None:
                continue
            if not dstOp.isSettable(k):
                continue
            if option.recursive:
                # deep copy
                desiredType = Coalesce(
                    dstOp.typeInfo(k),
                    type(v),
                )
                v = BeanUtil.copyProperties(v, desiredType, option)
            dstOp.set(k, v)
        return dst

    @staticmethod
    def toJsonCompatible(src, option: "BeanUtil.CopyOption" = CopyOption()):
        if BeanUtil._isPrimaryType(type(src)):
            if BeanUtil.isEnum(type(src)):
                return src.value
            else:
                return src
        elif BeanUtil._isFlatCollection(type(src)):
            return [
                BeanUtil.toJsonCompatible(v, option)
                for v in src
                if option.ignoreNoneInSrc and v is not None
            ]
        else:
            src = BeanUtil.copyProperties(src, dict)
            return {
                k: BeanUtil.toJsonCompatible(v, option)
                for k, v in src.items()
                if option.ignoreNoneInSrc and v is not None
            }


class Container:
    v = None

    def get(self):
        return self.v

    def set(self, newContent):
        self.v = newContent

    def isEmpty(self):
        return self.v is None


@dataclasses.dataclass
class Switch:

    onSetOn: typing.Callable[[], None] = None
    onSetOff: typing.Callable[[], None] = None
    initial: bool = False
    skipRespondingOnStateUnchanged: bool = True
    """
    skipRespondingOnStateUnchanged
        if true, onSetOn/onSetOff wont be called on calling on/off if its already in that on/off state
    """

    def __post_init__(self):
        self.__value = self.initial

    def on(self):
        if self.onSetOn is not None:
            if self.__value == True and self.skipRespondingOnStateUnchanged:
                pass
            else:
                self.onSetOn()
        self.__value = True

    def off(self):
        if self.onSetOff is not None:
            if self.__value == False and self.skipRespondingOnStateUnchanged:
                pass
            else:
                self.onSetOff()
        self.__value = False

    def setTo(self, val: bool):
        if val:
            self.on()
        else:
            self.off()

    def switch(self):
        if self():
            self.off()
        else:
            self.on()

    def __call__(self) -> bool:
        return self.__value


def InProbability(p: float) -> bool:
    return random.random() < p


def FlipCoin() -> bool:
    return InProbability(0.5)


# @EasyWrapper
def Singleton(cls):
    """
    known issue
        say if u have two classes with inheritance:
            @Singleton
            class Parent:
                def __init__(self):
                    print("Parent init")

            @Singleton
            class Child(Parent):
                def __init__(self):
                    super(Parent, self).__init__()
                    print("Child init")
            the __new__ of Parent wont be working as what object.__new__ does
    """
    cls.__singleton_instance__ = None
    cls.__oldNew__ = cls.__new__
    cls.__oldInit__ = cls.__init__

    def newNew(the_calling_cls_i_dont_care, *args, **kwargs):
        if cls.__singleton_instance__ is None:
            instance = cls.__oldNew__(cls)
            cls.__oldInit__(instance, *args, **kwargs)
            cls.__singleton_instance__ = instance
        return cls.__singleton_instance__

    def newInit(self, *args, **kwargs): ...

    cls.__new__ = newNew
    cls.__init__ = newInit
    return cls


def NormalizeIterableOrSingleArgToIterable(arg):
    if not isinstance(arg, (list, tuple)):
        return [arg]
    return arg


class DictAsAnObject:
    def __init__(self, data):
        self.__dict__ = data

    def __getattr__(self, attr):
        return self.__dict__.get(attr)


class AnnotationUtil:
    @staticmethod
    def __checkAnnoNonexisted(obj):
        return not hasattr(obj, "__ExtraAnnotations__")

    @staticmethod
    @EasyWrapper
    def Annotation(obj, **kwargs):
        return AnnotationUtil.AnnotationWithAnyKeyType(kwargs)(obj)

    @staticmethod
    @EasyWrapper
    def AnnotationWithAnyKeyType(obj, annoDict: dict):
        if AnnotationUtil.__checkAnnoNonexisted(obj):
            obj.__ExtraAnnotations__ = dict()
        obj.__ExtraAnnotations__.update(annoDict)
        return obj

    @staticmethod
    def getAnnotationDict(obj):
        if AnnotationUtil.__checkAnnoNonexisted(obj):
            return dict()
        return obj.__ExtraAnnotations__

    @staticmethod
    def getAnnotation(obj, key):
        return AnnotationUtil.getAnnotationDict(obj).get(key, None)

    @staticmethod
    def isAnnotationPresentd(obj, key):
        return key in AnnotationUtil.getAnnotationDict(obj)

    @staticmethod
    @EasyWrapper
    def AnnotationClass(cls):
        def newCall(self, foo):
            AnnotationUtil.AnnotationWithAnyKeyType({cls: self})(foo)
            return foo

        cls.__call__ = newCall
        return cls


class Cache:
    class UpdateStrategey:
        class UpdateStrategeyBase:
            def test(self, cache: "Cache") -> bool: ...
            def onUpdated(self, cache: "Cache") -> None: ...

        @dataclasses.dataclass
        class Outdated(UpdateStrategeyBase):
            outdatedTime: float
            __lastUpdateTime: float = dataclasses.field(init=False, default=None)

            def test(self, cache: "Cache"):
                if self.__lastUpdateTime is None:
                    return True
                return time.perf_counter() - self.__lastUpdateTime > self.outdatedTime

            def onUpdated(self, cache: "Cache"):
                self.__lastUpdateTime = time.perf_counter()

        @dataclasses.dataclass
        class Invalid(UpdateStrategeyBase):
            isValid: typing.Callable[[typing.Any], bool]

            def test(self, cache: "Cache"):
                return self.isValid(cache.__val)

    def __init__(
        self,
        toFetch,
        updateStrategey: "Cache.UpdateStrategey.UpdateStrategeyBase | list[Cache.UpdateStrategey.UpdateStrategeyBase]",
    ):
        self.__toFetch = toFetch
        self.updateStrategey = NormalizeIterableOrSingleArgToIterable(updateStrategey)
        self.__val = None

    def update(self):
        self.__val = self.__toFetch()
        for u in self.updateStrategey:
            u.onUpdated(self)

    def get(self, newest=None):
        if newest is None:
            newest = False
        if newest or any([u.test(self) for u in self.updateStrategey]):
            self.update()
        return self.__val


def PathNormalize(path: str):
    return path.replace("\\", "/")


def Decode(*args):
    assert len(args) % 2 == 1
    i = 0
    while True:
        if i + 1 >= len(args):
            # the default
            return args[i]
        cond, val = args[i], args[i + 1]
        if isinstance(cond, bool):
            if cond:
                return val
        elif isinstance(cond, callable):
            if len(inspect.signature(cond).parameters) == 0:
                if cond():
                    return val
            else:
                if cond(val):
                    return val
        i += 2


def Coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def mlambda(s: str) -> typing.Callable:
    exp = regex.compile(
        r"^\s*def\s*(?<paraAndType>.*?):\s*?\n?(?<body>.+)$", flags=regex.DOTALL
    )
    match = exp.match(s)
    if not match:
        raise SyntaxError("function signing syntax error")
    match = match.groupdict()
    paraAndType = match["paraAndType"]
    body = match["body"]

    def fixBodyIndent(body: str):
        # force at least 1 space indent
        lines = body.splitlines()
        lines = [" " + l for l in lines]
        return "\n".join(lines)

    body = fixBodyIndent(body)

    func = None
    lambdaName = "_lambda_" + str(uuid.uuid1()).replace("-", "_")

    def _setBackFun(f):
        nonlocal func
        func = f

    code = f"""
def {lambdaName}{paraAndType}:
{body}
_setBackFun({lambdaName})
    """
    caller_frame = sys._getframe(1)
    exec(
        code,
        caller_frame.f_globals,
        {**caller_frame.f_locals, "_setBackFun": _setBackFun},
    )
    return func


def ReadFileInZip(zipf, filename: str | list[str] | tuple[str]):
    zipf = zipfile.ZipFile(zipf)
    singleFile = not isinstance(filename, (tuple, list))
    if singleFile:
        filename = [filename]
    file = [zipf.read(f) for f in filename]
    if singleFile:
        return file[0]
    return file


@EasyWrapper
def RunThis(f: typing.Callable[[], typing.Any], *a, **kw):
    f(*a, **kw)
    return f


class MaxRetry:
    def __init__(
        self, succCond: typing.Callable[[], bool], maxRetry: int = 3, errOnMaxRetry=True
    ):
        self.succCond = succCond
        self.maxRetry = maxRetry
        self.i = 0
        self.isSuccessed = False
        self.errOnMaxRetry = errOnMaxRetry

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.maxRetry:
            if self.errOnMaxRetry:
                raise RuntimeError("Max retry reached")
            else:
                raise StopIteration
        if self.succCond():
            self.isSuccessed = True
            raise StopIteration
        self.i += 1
        return self.i


@EasyWrapper
def StaticCall(cls):
    cls.__new__ = lambda cls, *a, **kw: getattr(cls, "__call__")(*a, **kw)
    return cls


@EasyWrapper
def ComparatorOverloadedByPred(cls, Pred):
    cls.__lt__ = lambda a, b: Pred(a) < Pred(b)
    cls.__le__ = lambda a, b: Pred(a) <= Pred(b)
    cls.__gt__ = lambda a, b: Pred(a) > Pred(b)
    cls.__ge__ = lambda a, b: Pred(a) >= Pred(b)
    cls.__eq__ = lambda a, b: Pred(a) == Pred(b)
    cls.__ne__ = lambda a, b: Pred(a) != Pred(b)
    return cls


def BiptrFindSection(x, section):
    """
    assumes sections are sorted
    returns index i so that section[i]<=x<section[i+1]
    if x<section[0], returns -1
    if x>=section[-1], returns len(section)-1
    """
    beg = 0
    end = len(section)
    if beg == end:
        raise ValueError("empty section")
    if x < section[0]:
        return -1
    if x >= section[-1]:
        return len(section) - 1
    while beg + 1 < end:
        mid = (beg + end) // 2
        if x < section[mid]:
            end = mid
        elif x > section[mid]:
            beg = mid
        else:  # x==section[mid]
            beg = mid
            break
    return beg


class TaskScheduler:
    @ComparatorOverloadedByPred(lambda a: a.time)
    @dataclasses.dataclass
    class Task:
        time: float
        action: typing.Callable

    tasks: list["TaskScheduler.Task"]

    def __init__(self, initialTasks=None) -> None:
        self.tasks = Coalesce(initialTasks, list())

    def add(self, action: Task):
        heapq.heappush(self.tasks, action)

    def addAll(self, actions: list[Task]):
        for a in actions:
            self.add(a)

    def clear(self):
        self.tasks = list()

    def uptate(self, t: float):
        if self.tasks.__len__() != 0:
            task = self.tasks[0]
            if task.time <= t:
                heapq.heappop(self.tasks)
                task.action()


@dataclasses.dataclass
class Section:
    start: int = None
    end: int = None

    def __len__(self):
        return self.end - self.start

    def cut(self, container):
        return container[self.start : self.end]


def AutoFunctional(clz):
    for name, func in clz.__dict__.items():
        if not callable(func):
            continue
        if inspect.isclass(func):
            continue
        # only specified to be none
        if inspect.signature(func).return_annotation is not None:
            continue
        if isinstance(func, staticmethod):
            continue
        if name == "__init__":
            continue

        def funcToFunctional(func):
            def functionalWrapped(self, *args, **kw):
                func(self, *args, **kw)
                return self

            return functionalWrapped

        setattr(clz, name, funcToFunctional(func))
    return clz


@Singleton
class GSLogger:
    loggingFormat = "%(asctime)s - %(levelname)s - %(message)s"
    bulletinLogFormat = "%(message)s"
    loggingLevel = logging.INFO

    DefaultGlobalSysLoggerName = "GLOBAL_SYS_LOGGER"

    def _InnerLoggerShorthand(name):
        def shortHand(self, msg, *a, **kw):
            getattr(self.logger, name)(msg, *a, **kw)

        return shortHand

    debug = _InnerLoggerShorthand("debug")
    info = _InnerLoggerShorthand("info")
    warning = _InnerLoggerShorthand("warning")
    error = _InnerLoggerShorthand("error")
    critical = _InnerLoggerShorthand("critical")

    class Handlers:
        @staticmethod
        def ConsoleHandler():
            return logging.StreamHandler()

        FileHandlerDefaultPath = "asset/log/"

        @staticmethod
        def FileHandler(fileName=None, logFilePath=None):
            fileName = fileName or f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
            logFilePath = logFilePath or GSLogger.Handlers.FileHandlerDefaultPath
            EnsureDirectoryExists(logFilePath)
            return logging.FileHandler(os.path.join(logFilePath, fileName))

    def __init__(self, handlers: list[logging.Handler] = None):
        # Create a logger
        if handlers is None:
            handlers = [self.Handlers.ConsoleHandler(), self.Handlers.FileHandler()]
        logger = logging.getLogger(self.DefaultGlobalSysLoggerName)
        logger.setLevel(self.loggingLevel)
        for h in handlers:
            h.setFormatter(logging.Formatter(self.loggingFormat))
            logger.addHandler(h)

        self.logger = logger

    @EasyWrapper
    @staticmethod
    def ExceptionLogged(f, execType=Exception):
        execType = tuple(NormalizeIterableOrSingleArgToIterable(execType))

        @functools.wraps(f)
        def f2(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except BaseException as err:
                if isinstance(err, execType):
                    GSLogger().error(err)
                raise err

        return f2

    @EasyWrapper
    @staticmethod
    def TimeLogged(f):

        @functools.wraps(f)
        def f2(*args, **kwargs):
            stt = SingleSectionedTimer(True)
            ret = f(*args, **kwargs)
            GSLogger().info(f"time cost of {f.__name__}: {stt.get()}")
            return ret

        return f2


class CollUtil:
    @staticmethod
    def split(iterable: typing.Iterable, size: int):
        return [
            iterable[i : min(i + size, len(iterable))]
            for i in range(0, len(iterable), size)
        ]


def NormalizeCrlf(s: str):
    return s.replace("\r\n", "\n").replace("\r", "\n")


class Profiling:
    class Persister:
        def __init__(self): ...
        def write(self, group, item, value): ...
        def flush(self): ...

    class PersisterInMemory(Persister):

        def getPath(self):
            return os.path.join(os.getcwd(), "profiling.log")

        def __init__(self):
            self.data: dict[tuple, list[float]] = dict()
            path = self.getPath()
            if os.path.exists(path):
                try:
                    self.data = json.loads(ReadTextFile(path))
                except:
                    pass

        def write(self, group, item, value):
            k = f"{group}:{item}"
            if k not in self.data:
                self.data[k] = list()
            else:
                self.data[k].append(value)

        def flush(self):
            WriteTextFile(self.getPath(), json.dumps(self.data))

    def __init__(self):
        self.persister = Profiling.PersisterInMemory()

    def __del__(self):
        self.persister.flush()

    def profiling(self, group, item):
        def toGetF(f):
            @functools.wraps(f)
            def newF(*a, **kw):
                ps = perf_statistic(True)
                f(*a, **kw)
                cost = ps.stop().time()
                self.persister.write(group, item, cost)

            return newF

        return toGetF


class ParamValueUnset:
    # placeholder for param unset, use when None is a meaningful param
    ...


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


class LazyLoading:
    """
    usage:
        class Clz(LazyLoading):
            a: int = LazyLoading.LazyField(lambda self: 1)
            b: int = LazyLoading.LazyField(lambda self: self.a + 1)
            c: int = LazyLoading.LazyField(lambda self: self.b + 1)
        clz = Clz()
    known issue:
        when using dataclass to initialize, all lazy fields will be initialized on creating
    """

    @dataclasses.dataclass
    class LazyField:
        fetcher: typing.Callable

    def _raw_get(self, name):
        return super().__getattribute__(name)

    def _raw_set(self, name, value):
        setattr(self, name, value)

    def _is_uninitialized(self, name):
        return isinstance(self._raw_get(name), LazyLoading.LazyField)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, LazyLoading.LazyField):
            value = value.fetcher(self)
            self._raw_set(name, value)
        return value


T = typing.TypeVar("T")


class Stream(typing.Generic[T]):
    # copied from superstream!
    R = typing.TypeVar("R")
    K = typing.TypeVar("K")
    U = typing.TypeVar("U")

    def __init__(self, stream: typing.Iterable[T]):
        self._stream = iter(stream)

    def __iter__(self):
        return self._stream

    @staticmethod
    def of(*args: T) -> "Stream[T]":
        return Stream(args)

    def map(self, func: typing.Callable[[T], R]) -> "Stream[R]":
        return Stream(map(func, self._stream))

    def flat_map(self, func: typing.Callable[[T], "Stream[R]"]) -> "Stream[R]":
        return Stream(itertools.chain.from_iterable(map(func, self._stream)))

    def filter(self, func: typing.Callable[[T], bool]) -> "Stream[T]":
        return Stream(filter(func, self._stream))

    def for_each(self, func: typing.Callable[[T], None]) -> None:
        for i in self._stream:
            func(i)

    def distinct(self):
        return Stream(list(dict.fromkeys(self._stream)))

    def sorted(self, key=None, reverse=False) -> "Stream[T]":
        return Stream(sorted(self._stream, key=key, reverse=reverse))

    def count(self) -> int:
        cnt = itertools.count()
        collections.deque(zip(self._stream, cnt), maxlen=0)
        return next(cnt)

    def sum(self) -> "T":
        return sum(self._stream)

    def group_by(self, classifier: typing.Callable[[T], K]) -> dict[K, list[T]]:
        groups = {}
        for i in self._stream:
            groups.setdefault(classifier(i), []).append(i)
        return groups

    def reduce(
        self, func: typing.Callable[[T, T], T], initial: T = None
    ) -> typing.Optional[T]:
        if initial is not None:
            return functools.reduce(func, self._stream, initial)
        else:
            try:
                return functools.reduce(func, self._stream)
            except TypeError:
                return None

    def limit(self, max_size: int) -> "Stream[T]":
        return Stream(itertools.islice(self._stream, max_size))

    def skip(self, n: int) -> "Stream[T]":
        return Stream(itertools.islice(self._stream, n, None))

    def min(
        self, key: typing.Callable[[T], typing.Any] = lambda x: x, default: T = None
    ) -> typing.Optional[T]:
        """
        :param default: use default value when stream is empty
        :param key: at lease supported __lt__ method
        """
        return min(self._stream, key=key, default=default)

    def max(
        self, key: typing.Callable[[T], typing.Any] = lambda x: x, default: T = None
    ) -> typing.Optional[T]:
        """
        :param default: use default value when stream is empty
        :param key: at lease supported __lt__ method
        """
        return max(self._stream, key=key, default=default)

    def find_first(self) -> typing.Optional[T]:
        try:
            return next(self._stream)
        except StopIteration:
            return None

    def any_match(self, func: typing.Callable[[T], bool]) -> bool:
        """
        this is equivalent to
            for i in self._stream:
                if func(i):
                    return True
            return False
        :param func:
        :return:
        """
        return any(map(func, self._stream))

    def all_match(self, func: typing.Callable[[T], bool]) -> bool:
        return all(map(func, self._stream))

    def none_match(self, func: typing.Callable[[T], bool]) -> bool:
        return not self.any_match(func)

    def to_list(self) -> list[T]:
        return list(self._stream)

    def to_set(self) -> set[T]:
        return set(self._stream)

    def to_dict(
        self, k: typing.Callable[[T], K], v: typing.Callable[[T], U]
    ) -> dict[K, U]:
        return {k(i): v(i) for i in self._stream}

    def to_map(
        self, k: typing.Callable[[T], K], v: typing.Callable[[T], U]
    ) -> dict[K, U]:
        return self.to_dict(k, v)

    def collect(self, func: typing.Callable[[typing.Iterable[T]], R]) -> R:
        return func(self._stream)


class PositionalArgsResolvedAsNamedKwargs:
    def __init__(
        self,
        func: typing.Callable,
    ):
        self.func = func
        self.parameters_of_signature = list(inspect.signature(func).parameters.items())

    def apply(
        self,
        modifier: typing.Callable[[str, typing.Any], typing.Any],
        args: list[typing.Any],
        kwargs: dict[str, typing.Any],
    ):
        """
        modifier:
            param:
                name of param
                value of param
            return:
                value result of modified param
        """
        args = [
            modifier(name, val_arg)
            for val_arg, (name, param) in zip(args, self.parameters_of_signature)
        ]
        kwargs = {k: modifier(k, v) for k, v in kwargs.items()}
        return args, kwargs


################################################
################# not so solid #################
################################################

try:
    import regex

    class FSMUtil:

        class ParseError(Exception): ...

        class TokenTypeLike(enum.Enum): ...

        class TokenMatcher:
            # s here is actually a substring of the original string[i:]
            # i is not used to cut s again here
            def tryMatch(
                self, s: str, i: int
            ) -> typing.Union[None, "FSMUtil.Token"]: ...

        @dataclasses.dataclass
        class RegexpTokenMatcher(TokenMatcher):
            exp: str | regex.Pattern
            type: "FSMUtil.TokenTypeLike"

            def __post_init__(self):
                if isinstance(self.exp, str):
                    self.exp = regex.compile(self.exp, flags=regex.DOTALL)

            def tryMatch(self, s: str, i: int) -> "None | FSMUtil.Token":
                match = regex.match(self.exp, s[i:])
                if match is not None:
                    return FSMUtil.Token(
                        self.type,
                        match.group(0),
                        i,
                        i + len(match.group(0)),
                        source=s,
                    )
                return None

        @dataclasses.dataclass(repr=True)
        class Token:
            type: "FSMUtil.TokenTypeLike"
            value: typing.Any
            start: int
            end: int
            source: str = None

            def Unexpected(self):
                msg = ""
                msg += f'unexpected token "{self.value}":{self.type}\n'
                if self.source is not None:
                    # provide line No information
                    lineStartPos = [
                        m.end()
                        for m in regex.finditer(r"^", self.source, regex.MULTILINE)
                    ]
                    lineNo = Section(
                        BiptrFindSection(self.start, lineStartPos),
                        BiptrFindSection(self.end, lineStartPos),
                    )
                    columnNo = Section(
                        self.start - lineStartPos[lineNo.start],
                        self.end - lineStartPos[lineNo.end],
                    )
                    msg += f"At {self.start}-{self.end}, Ln {lineNo.start+1} Col {columnNo.start+1} ~ Ln {lineNo.end+1} Col {columnNo.end+1}"
                else:
                    msg += f"At {self.start}-{self.end}"
                raise FSMUtil.ParseError(msg)

            def toSection(self):
                return Section(self.start, self.end)

            def viewSection(self, s: str):
                lineStart = s.rfind("\n", 0, self.start)
                if lineStart == -1:
                    lineStart = 0
                lineEnd = s.find("\n", self.end)
                if lineEnd == -1:
                    lineEnd = len(s)
                return "{}>>>>{}<<<<{}".format(
                    s[lineStart : self.start],
                    s[self.start : self.end],
                    s[self.end : lineEnd],
                )

        @dataclasses.dataclass
        class GetTokenParam:
            # for rich function support of getToken()
            # unused for now
            s: str
            matcher: list["FSMUtil.TokenMatcher"]
            redirectedToTokenWhenUnparsable: "FSMUtil.TokenTypeLike" = None

        @staticmethod
        def getToken(
            s: str,
            i: int,
            matchers: list["FSMUtil.TokenMatcher"],
        ) -> "FSMUtil.Token":
            for m in matchers:
                token = m.tryMatch(s, i)
                if token is not None:
                    return token
            sectionEnd = min(i + 10, len(s))
            raise FSMUtil.ParseError(f"unparsable token at {i}: {s[i:sectionEnd]}")

        @staticmethod
        def getAllToken(
            s: str,
            matchers: list["FSMUtil.TokenMatcher"],
            endTokenType: "FSMUtil.TokenTypeLike",
            start=0,
        ) -> list["FSMUtil.Token"]:
            i = start
            tokenList: list[FSMUtil.Token] = []
            while True:
                token = FSMUtil.getToken(s, i, matchers)
                tokenList.append(token)
                i = token.end
                if token.type == endTokenType:
                    break
            return tokenList

        class PeekableLazyTokenizer:
            class Iterator:
                pltk: "FSMUtil.PeekableLazyTokenizer"
                pos: int

                def __init__(
                    self,
                    parent: "FSMUtil.PeekableLazyTokenizer | FSMUtil.PeekableLazyTokenizer.Iterator" = None,
                ):
                    self._init(parent)

                def _init(
                    self,
                    parent: "FSMUtil.PeekableLazyTokenizer | FSMUtil.PeekableLazyTokenizer.Iterator" = None,
                ):
                    if parent is None:
                        self.pltk = None
                        self.pos = None
                    elif isinstance(parent, FSMUtil.PeekableLazyTokenizer.Iterator):
                        self.pltk = parent.pltk
                        self.pos = parent.pos
                    elif isinstance(parent, FSMUtil.PeekableLazyTokenizer):
                        self._init(parent._it)

                def next(self):
                    ret = self.pltk.getByTokenAbsIndex(self.pos)
                    self.movNext()
                    return ret

                def movNext(self):
                    self.pos += 1

                def movPrev(self):
                    self.pos -= 1

            Peeker = Iterator

            s: str
            matchers: list["FSMUtil.TokenMatcher"]
            _tokenList: list["FSMUtil.Token"]
            _indexTextTokenizing: int
            _it: Iterator

            def __init__(
                self,
                s: str,
                matchers: list["FSMUtil.TokenMatcher"],
                start=0,
            ):
                self.s = s
                self.matchers = matchers
                self._tokenList = list()
                self._indexTextTokenizing = start
                self._it = self.Iterator(None)
                self._it.pltk = self
                self._it.pos = 0

            def _tokenizeNext(self):
                token = FSMUtil.getToken(
                    self.s, self._indexTextTokenizing, self.matchers
                )
                self._indexTextTokenizing = token.end
                self._tokenList.append(token)

            def getByTokenAbsIndex(self, index):
                while True:
                    if index < len(self._tokenList):
                        return self._tokenList[index]
                    self._tokenizeNext()

            def next(self):
                return self._it.next()

            def movPrev(self):
                return self._it.movPrev()

    class UrlFullResolution(LazyLoading):

        class _Scopes:
            protocol = "protocol"
            host = "host"
            path = "path"
            param = "param"
            secondaryHost = "secondaryHost"
            baseHost = "baseHost"
            domain = "domain"
            port = "port"
            folder = "folder"
            fileName = "fileName"
            fileBaseName = "fileBaseName"
            extName = "extName"

        url: str | None
        protocol = LazyLoading.LazyField(
            lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.protocol)
        )
        host = LazyLoading.LazyField(
            lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.host)
        )
        path = LazyLoading.LazyField(
            lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.path)
        )
        param = LazyLoading.LazyField(
            lambda self: self._parseStepGlobally(UrlFullResolution._Scopes.param)
        )
        secondaryHost = LazyLoading.LazyField(
            lambda self: self._parseStepHost(UrlFullResolution._Scopes.secondaryHost)
        )
        baseHost = LazyLoading.LazyField(
            lambda self: self._parseStepHost(UrlFullResolution._Scopes.baseHost)
        )
        domain = LazyLoading.LazyField(
            lambda self: self._parseStepHost(UrlFullResolution._Scopes.domain)
        )
        port = LazyLoading.LazyField(
            lambda self: self._parseStepHost(UrlFullResolution._Scopes.port)
        )
        folder = LazyLoading.LazyField(
            lambda self: self._parseStepPath(UrlFullResolution._Scopes.folder)
        )
        fileName = LazyLoading.LazyField(
            lambda self: self._parseStepPath(UrlFullResolution._Scopes.fileName)
        )
        fileBaseName = LazyLoading.LazyField(
            lambda self: self._parseStepPath(UrlFullResolution._Scopes.fileBaseName)
        )
        extName = LazyLoading.LazyField(
            lambda self: self._parseStepPath(UrlFullResolution._Scopes.extName)
        )

        def _SetScope(self, name, val):
            self._raw_set(name, val)

        class RegPool:
            globally = regex.compile(
                r"^((?<protcol>[A-Za-z]+)://)?(?<host>[^/]+\.[^/.]+)?(?<path>[^?]*)?(\?(?<param>.*))?$"
            )
            host = regex.compile(r"^(?<host>[^:]+)(:(?<port>\d+))?$")
            path = regex.compile(
                r"^(?<folder>/?(?:[^/]+/)+)(?:(?<fileName>(?<fileBaseName>.+?)(?:\.(?<extName>.*))?))?$"
            )

        class UnexpectedException(Exception): ...

        @staticmethod
        def _parse_and_return_specified_field(func: typing.Callable):
            def f2(self: "UrlFullResolution", ret_field=None):
                func(self)
                if ret_field is not None:
                    return self._raw_get(ret_field)

            return f2

        @_parse_and_return_specified_field
        def _parseStepGlobally(self):
            if any(
                (
                    self._is_uninitialized(n)
                    for n in [
                        UrlFullResolution._Scopes.protocol,
                        UrlFullResolution._Scopes.host,
                        UrlFullResolution._Scopes.path,
                        UrlFullResolution._Scopes.param,
                    ]
                )
            ):
                protocol, host, path, param = [None] * 4
                if self.url is not None:
                    matchGlobally = UrlFullResolution.RegPool.globally.match(self.url)
                    if matchGlobally is not None:
                        protocol, host, path, param = matchGlobally.group(
                            "protcol", "host", "path", "param"
                        )
                self._SetScope(UrlFullResolution._Scopes.protocol, protocol)
                self._SetScope(UrlFullResolution._Scopes.host, host)
                self._SetScope(UrlFullResolution._Scopes.path, path)
                self._SetScope(UrlFullResolution._Scopes.param, param)

        @_parse_and_return_specified_field
        def _parseStepHost(self):
            self._parseStepGlobally()
            if any(
                (
                    self._is_uninitialized(n)
                    for n in [
                        UrlFullResolution._Scopes.port,
                        UrlFullResolution._Scopes.secondaryHost,
                        UrlFullResolution._Scopes.baseHost,
                        UrlFullResolution._Scopes.domain,
                    ]
                )
            ):
                secondaryHost, baseHost, domain, port = [None] * 4
                if self.host is not None:
                    matchHost = UrlFullResolution.RegPool.host.match(self.host)
                    if matchHost is not None:
                        hostNoPort, port = matchHost.group("host", "port")
                        lHost = hostNoPort.split(".")
                        if len(lHost) < 2:
                            raise UrlFullResolution.UnexpectedException()
                        if not (
                            len(lHost) == 4
                            and all(
                                str.isdigit(i) and 255 >= int(i) >= 0 for i in lHost
                            )
                        ):
                            secondaryHost = ".".join(lHost[0:-2])
                            baseHost = ".".join(lHost[-2:])
                            domain = lHost[-1]
                self._SetScope(UrlFullResolution._Scopes.port, port)
                self._SetScope(UrlFullResolution._Scopes.secondaryHost, secondaryHost)
                self._SetScope(UrlFullResolution._Scopes.baseHost, baseHost)
                self._SetScope(UrlFullResolution._Scopes.domain, domain)

        @_parse_and_return_specified_field
        def _parseStepPath(self):
            self._parseStepGlobally()
            if any(
                (
                    self._is_uninitialized(n)
                    for n in [
                        UrlFullResolution._Scopes.folder,
                        UrlFullResolution._Scopes.fileName,
                        UrlFullResolution._Scopes.extName,
                    ]
                )
            ):
                folder, fileName, fileBaseName, extName = [None] * 4
                if self.path is not None:
                    matchPath = UrlFullResolution.RegPool.path.match(self.path)
                    if matchPath is not None:
                        folder, fileName, fileBaseName, extName = matchPath.group(
                            "folder", "fileName", "fileBaseName", "extName"
                        )
                self._SetScope(UrlFullResolution._Scopes.folder, folder)
                self._SetScope(UrlFullResolution._Scopes.fileName, fileName)
                self._SetScope(UrlFullResolution._Scopes.fileBaseName, fileBaseName)
                self._SetScope(UrlFullResolution._Scopes.extName, extName)

        def calcAll(self):
            self._parseStepGlobally()
            self._parseStepHost()
            self._parseStepPath()

        def __init__(self, url: str):
            self.url = PathNormalize(url)

        @staticmethod
        def of(url: str):
            return UrlFullResolution(url)

except ImportError:
    pass

try:

    import aenum

    def ExtendEnum(src):
        def deco_inner(cls):
            nonlocal src
            if (
                issubclass(src, aenum.Enum)
                or aenum.stdlib_enums
                and issubclass(src, aenum.stdlib_enums)
            ):
                src = src.__members__.items()
            for name, value in src:
                aenum.extend_enum(cls, name, value)
            return cls

        return deco_inner

    class MessagedThread:
        class MessageType(aenum.Enum):
            stop = 0

        @dataclasses.dataclass
        class Message:
            type: "MessagedThread.MessageType"
            body: typing.Any = None

        mq: queue.Queue["MessagedThread.Message"] = queue.Queue()

        def run(self): ...

except ImportError:
    pass
