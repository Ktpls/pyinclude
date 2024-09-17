import concurrent.futures as futures
from datetime import datetime
import copy
import ctypes
import dataclasses
import enum
import functools
import inspect
import itertools
import math
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import traceback
import typing
import uuid
import json
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


IdentityMapping = lambda x: x


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


def EasyWrapper(wrappedLogic=None):
    """
    use like this
        @WrapperAsMyTaste()
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
        @yourWrapper(some_callable)
        def foo(func_arg): ...
            cuz wrapper is confused with if its processing the wrapped function or callable in arg
            use this instead if wrapper needs another callable more than the one you are wrapping
                @yourWrapper(keyword=some_callable)
                def foo(func_arg): ...
        @someClassInstance.methodDecorator
        def foo(...): ...
            cuz wrapper will recieve the instance as the first arg, and the foo as the second
            making easywrapper confused with wrapping a class with a method as arg
            use this instead
                @someClassInstance.methodDecorator()
                def foo(func_arg): .
    somehow buggy but works almost fine
    ###############
    note that python design is piece of shlt
    ###############

    known issue

    """

    def toGetWrapperLogic(wrappedLogic):
        def newWrapper(*arg, **kw):
            def toGetFLogic(fLogic):
                return wrappedLogic(fLogic, *arg, **kw)

            if (
                len(arg) == 1
                and (inspect.isfunction(arg[0]) or inspect.isclass(arg[0]))
                and len(kw) == 0
            ):
                # calling without parens
                return wrappedLogic(arg[0])
            else:
                return toGetFLogic

        return newWrapper

    if wrappedLogic is None:
        # calling without parens
        return toGetWrapperLogic
    else:
        return toGetWrapperLogic(wrappedLogic)


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


def AllFileIn(path, includeFileInSubDir=True):
    ret = []
    for dirpath, dir, file in os.walk(path):
        if not includeFileInSubDir and dirpath != path:
            continue
        ret.extend([os.path.join(dirpath, f) for f in file])
    return ret


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
        if not issubclass(implType, StoppableSomewhat):
            raise NotImplementedError("doesn't work")
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
    """
    derivate from it and override foo()
    """

    # TODO give option to handle error by user. thats for wtutily system to log error
    def __init__(
        self,
        strategy_runonrunning: "StoppableSomewhat.StrategyRunOnRunning" = None,
        strategy_error: "StoppableSomewhat.StrategyError" = None,
        pool: futures.ThreadPoolExecutor = None,
    ) -> None:
        super().__init__(strategy_runonrunning, strategy_error)
        self.running: bool = False
        self.stopsignal: bool = True
        self.pool: futures.ThreadPoolExecutor = Coalesce(
            pool, futures.ThreadPoolExecutor()
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
            self.running = False

        self.submit = self.pool.submit(call)

    @FunctionalWrapper
    def stop(self) -> None:
        if self.submit is None:
            return
        self.stopsignal = True
        self.submit.result()
        self.running = False
        self.submit = None

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


def WriteFile(path, content):
    EnsureDirectoryExists(os.path.dirname(path))
    with open(path, "wb+") as f:
        f.write(content)


def AppendFile(path, content):
    EnsureDirectoryExists(os.path.dirname(path))
    with open(path, "ab+") as f:
        f.write(content.encode("utf-8"))


def ReadTextFile(path: str) -> str:
    return ReadFile(path).decode("utf-8")


def WriteTextFile(path: str, text: str):
    WriteFile(path, text.encode("utf-8"))


def GetTimeString():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


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


class perf_statistic:
    """
    calculate the time past between start() to now, directly by perf_counter()-starttime
    record all accumulated time before start(), but uncleared after stop()
    so start and stop are also playing roles as resume and pause
    countcycle() will increase the cycle count, helping to calculate average time in a loop-like task
    clear() will clear all accumulated time, stops counting
    """

    def __init__(self, startnow=False):
        self.clear()
        if startnow:
            self.start()

    def clear(self):
        self._starttime = None
        self._stagedTime = 0
        self._cycle = 0
        self._stagedTimeList = list()
        return self

    def start(self):
        self._starttime = time.perf_counter()
        return self

    def countcycle(self):
        self._cycle += 1
        return self

    def stop(self):
        if not self.isRunning():
            return self
        timeThisRound = self._timeCurrentlyCounting()
        self._stagedTime += timeThisRound
        self._stagedTimeList.append(timeThisRound)
        self._starttime = None
        return self

    def isRunning(self):
        return self._starttime is not None

    def time(self):
        return self._stagedTime + self._timeCurrentlyCounting()

    def aveTime(self):
        return self.time() / (self._cycle if self._cycle > 0 else 1)

    def _timeCurrentlyCounting(self):
        return time.perf_counter() - self._starttime if self.isRunning() else 0

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
            return output, self.AnalizerFrameData(
                partp=self.kp * error,
                parti=self.ki * self.integral,
                partd=self.kd * derivative,
                error=error,
                integral=self.integral,
                derivative=derivative,
            )
        else:
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

    t = property(fget=lambda self: None)


class TimeNonconcernedStage(Stage):
    def step(self, dt):
        pass

    t = property(fget=lambda self: 0)


class SyncExecutableManager:
    def __init__(self, pool: futures.ThreadPoolExecutor) -> None:
        self.pool = pool
        self.selist: list[SyncExecutable] = []

    def step(self):
        # call this on wolf update
        # make sure set running before submitting. or would be possibly kicked out here
        self.selist = [
            e for e in self.selist if e.state != SyncExecutable.STATE.stopped
        ]
        for se in self.selist:
            se.cond.acquire(True)
            if se.state == SyncExecutable.STATE.suspended:
                # knowing not satisfied, skip waking up
                if se.waitWhat():
                    se.cond.notify_all()
                    se.cond.wait()  # consider wait asyncly here and below
            elif se.state == SyncExecutable.STATE.running:
                se.cond.notify_all()
                se.cond.wait()
            elif se.state == SyncExecutable.STATE.stopped:
                pass
            else:
                pass
            se.cond.release()

    def submit(self, se: "SyncExecutable", foo: typing.Callable):
        self.selist.append(se)
        return self.pool.submit(foo)


class SyncExecutable:
    # for impl serialized but sync mechanization in async foo
    # stage is something with t readable
    class STATE(enum.Enum):
        stopped = 0
        running = 1
        suspended = 2

    def __init__(
        self, stage: Stage, sem: SyncExecutableManager, raiseOnErr=True
    ) -> None:
        self.stage = stage
        self.sem = sem
        self.cond = threading.Condition()
        self.state = self.STATE.stopped
        self.future = None
        self.raiseOnErr = raiseOnErr
        self.waitWhat = None

    # override
    def main(self, **arg):
        raise BaseException("not implemented")

    def run(self, **arg):
        def foo():
            self.cond.acquire(True)
            try:
                self.main(**arg)
            except BaseException as e:
                if self.raiseOnErr:
                    traceback.print_exc()
                    raise e
                else:
                    traceback.print_exc()
            self.state = self.STATE.stopped
            self.cond.notify_all()  # no more sleep, aks sem to get up
            self.cond.release()

        if not self.isworking():
            self.state = self.STATE.running
            self.future = self.sem.submit(self, foo)
        return self

    # available in main
    def sleepUntil(self, untilWhat, timeout=None):
        overduetime = self.stage.t + timeout if timeout else None

        def untilWhatOrTimeOut():
            return untilWhat() or (overduetime and self.stage.t >= overduetime)

        # give right of check to manager, so can i save cost of thread switching
        self.waitWhat = untilWhatOrTimeOut
        self.state = self.STATE.suspended
        while True:
            """
            do this in main thread so cancelling thread switching cost
            """
            if untilWhatOrTimeOut():
                break
            # register
            self.cond.notify_all()
            self.cond.wait()
        self.waitWhat = None
        self.state = self.STATE.running

    # available in main
    def sleep(self, delaytime):
        self.sleepUntil(lambda: False, delaytime)

    # available in main
    def stepOneFrame(self):
        self.cond.notify_all()
        self.cond.wait()

    def isworking(self):
        return self.state != self.STATE.stopped


class AccessibleQueue:
    Annotation = lambda T: list[T]

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
    class Option:
        ignoreNoneInSrc: bool = True

    """
    have to deal with dict, object, and class(only on dest)
    """

    @staticmethod
    def __GetFields(clz):
        parents = NormalizeIterableOrSingleArgToIterable(clz.__base__)
        result = dict()
        for p in parents:
            if p == object:
                continue
            result.update(BeanUtil.__GetFields(p))
        if hasattr(clz, "__annotations__"):
            # override
            result.update(clz.__annotations__)
        return result

    @staticmethod
    def GetEmptyInstance(cls):
        args = inspect.getargs(cls.__init__.__code__)
        if len(args) > 1:
            # found init with arg more than self
            inst = object.__new__(cls)
            fields = BeanUtil.__GetFields(cls)
            for name, taipe in fields.items():
                """
                for taipe as class, its possible to recursively call GetEmptyInstance
                but taipe could be str, or typing.GenericAlias
                """
                setattr(inst, name, None)
            return inst
        else:
            return cls()

    @staticmethod
    def __FieldConversionFunc(obj, field):
        taipe = obj.__annotations__.get(field, None)
        if taipe is None or isinstance(taipe, str):
            # not type annotated, or annotated like field:"some class"
            return IdentityMapping
        if hasattr(taipe, "__origin__"):
            # typing.GenericAlias
            taipe = taipe.__origin__
        return taipe

    @staticmethod
    def __GetterOf(obj):
        if isinstance(obj, dict):
            return lambda: obj.items()
        return lambda: obj.__dict__.items()

    @staticmethod
    def __SetterOf(obj):
        if isinstance(obj, dict):

            def DictSetter(obj, k, v):
                if k in obj:
                    obj[k] = v

            return DictSetter

        def ObjSetter(obj, k, v):
            if k in obj.__dict__:
                try:
                    # try convert it to proper type
                    v = BeanUtil.__FieldConversionFunc(obj, k)(v)
                except:
                    pass
                obj.__setattr__(k, v)

        return ObjSetter

    @staticmethod
    def __DictOrObj2DictOrObjCopy(src: object, dst: object, option: "BeanUtil.Option"):
        Getter = BeanUtil.__GetterOf(src)
        Setter = BeanUtil.__SetterOf(dst)
        for k, v in Getter():
            if option.ignoreNoneInSrc and v is None:
                continue
            Setter(dst, k, v)

    @staticmethod
    def __DictOrObj2ClassCopy(
        src: object, dst: typing.Callable, option: "BeanUtil.Option"
    ):
        dstobj = BeanUtil.GetEmptyInstance(dst)
        BeanUtil.__DictOrObj2DictOrObjCopy(src, dstobj, option)
        return dstobj

    @staticmethod
    def copyProperties(src, dst: object, option: "BeanUtil.Option" = Option()):
        if inspect.isclass(dst):
            return BeanUtil.__DictOrObj2ClassCopy(src, dst, option)
        else:
            BeanUtil.__DictOrObj2DictOrObjCopy(src, dst, option)


class Container:
    __content = None

    def get(self):
        return self.__content

    def set(self, newContent):
        self.__content = newContent

    def isEmpty(self):
        return self.__content is None


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

    def setTo(self, val):
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


@EasyWrapper
def Singleton(cls):
    cls.__singleton_instance__ = None
    cls.__oldNew__ = cls.__new__
    cls.__oldInit__ = cls.__init__

    def newNew(cls, *args, **kwargs):
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


def mlambda(s: str, _globals=None, _locals=None) -> typing.Callable:
    exp = regex.compile(
        r"^\s*def\s*(?<paraAndType>.*?):\s*\n?(?<body>.+)$", flags=regex.DOTALL
    )
    match = exp.match(s)
    if not match:
        raise SyntaxError("function signing syntax error")
    match = match.groupdict()
    paraAndType = match["paraAndType"]
    body = match["body"]

    emptyLine = regex.compile("^\s*(#.*)?$", flags=regex.MULTILINE)

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

    exec(code, _globals, {**(_locals or {}), "_setBackFun": _setBackFun})
    return func


def ReprObject(o):
    def serialize(o):
        if hasattr(o, "__repr__"):
            return o.__repr__()
        else:
            return {k: v for k, v in o.__dict__.items() if not str.startswith(k, "_")}

    return json.dumps(
        o,
        indent=4,
        ensure_ascii=False,
        default=serialize,
        sort_keys=True,
    )


def ReadFileInZip(zipf, filename: str | list[str] | tuple[str]):
    zipf = zipfile.ZipFile(zipf)
    singleFile = not isinstance(filename, (tuple, list))
    if singleFile:
        filename = [filename]
    file = [zipf.read(f) for f in filename]
    if singleFile:
        return file[0]
    return file


################################################
################# not so solid #################
################################################

try:
    import regex

    class FSMUtil:

        class ParseError(Exception): ...

        class TokenTypeLike(enum.Enum): ...

        class TokenMatcher:
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
                        self.type, match.group(0), i, i + len(match.group(0))
                    )
                return None

        @dataclasses.dataclass(repr=True)
        class Token:
            type: "FSMUtil.TokenTypeLike"
            value: typing.Any
            start: int
            end: int

            def Unexpected(self):
                raise FSMUtil.ParseError(
                    f"unexpected token {self.value}:{self.type} at {self.start}-{self.end}"
                )

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

        @staticmethod
        def getToken(
            s: str, i: int, matchers: list["FSMUtil.TokenMatcher"]
        ) -> "FSMUtil.Token":
            for m in matchers:
                token = m.tryMatch(s, i)
                if token is not None:
                    return token
            sectionEnd = min(i + 10, len(s))
            raise FSMUtil.ParseError(f"unparseable token at {i}: {s[i:sectionEnd]}")

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
            s: str
            matchers: list["FSMUtil.TokenMatcher"]
            _tokenList: list["FSMUtil.Token"] = list()
            _indexTextTokenizing = 0
            _indexTokenListCurr = -1

            def __init__(
                self,
                s: str,
                matchers: list["FSMUtil.TokenMatcher"],
                start=0,
            ):
                self.s = s
                self.matchers = matchers
                self._indexTextTokenizing = start

            def _tokenizeNext(self):
                token = FSMUtil.getToken(
                    self.s, self._indexTextTokenizing, self.matchers
                )
                self._indexTextTokenizing = token.end
                self._tokenList.append(token)

            def getByTokenIndex(self, index):
                while True:
                    if index < len(self._tokenList):
                        return self._tokenList[index]
                    self._tokenizeNext()

            def curr(self):
                return self.peek(0)

            def peek(self, distance=1):
                return self.getByTokenIndex(self._indexTokenListCurr + distance)

            def next(self):
                self._indexTokenListCurr += 1
                return self.curr()

    @dataclasses.dataclass
    class UrlFullResolution:

        url: str | None
        protocol: str | None
        host: str | None
        path: str | None
        param: str | None
        secondaryHost: str | None
        baseHost: str | None
        domain: str | None
        port: str | None
        folder: str | None
        fileName: str | None
        extName: str | None

        class RegPool:
            globally = regex.compile(
                r"^(?<protcol>[A-Za-z]+://)?(?<host>[^/]+\.[^/.]+)?(?<path>[^?]*)?(?<param>\?.*)?$"
            )
            host = regex.compile(r"^(?<host>[^:]+)(?<port>:\d+)?$")
            path = regex.compile(
                r"^(?<folder>.+?)(?:/(?<fileName>[^/]+(?<extName>\..*)))?$"
            )

        class UnexpectedException(Exception): ...

        @staticmethod
        def of(url: str):
            url = PathNormalize(url)
            (
                protocol,
                host,
                path,
                param,
                secondaryHost,
                baseHost,
                domain,
                port,
                folder,
                fileName,
                extName,
            ) = [None] * 11
            matchGlobally = UrlFullResolution.RegPool.globally.match(url)
            if matchGlobally is not None:
                protocol, host, path, param = matchGlobally.group(
                    "protcol", "host", "path", "param"
                )
                if host is not None:
                    matchHost = UrlFullResolution.RegPool.host.match(host)
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
                if path is not None:
                    matchPath = UrlFullResolution.RegPool.path.match(
                        path,
                    )
                    if matchPath is not None:
                        folder, fileName, extName = matchPath.group(
                            "folder", "fileName", "extName"
                        )

            return UrlFullResolution(
                url=url,
                protocol=protocol,
                host=host,
                path=path,
                param=param,
                secondaryHost=secondaryHost,
                baseHost=baseHost,
                domain=domain,
                port=port,
                folder=folder,
                fileName=fileName,
                extName=extName,
            )

except ImportError:
    pass
