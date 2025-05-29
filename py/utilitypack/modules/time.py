import datetime
import typing
import time
import dataclasses
import os
import json
import functools
from .io import ReadTextFile, WriteTextFile


def GetTimeString():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


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
        self._starttime = self.timeCounter()
        return self

    def isRunning(self):
        return self._starttime is not None

    def get(self) -> float:
        return self.timeCounter() - self._starttime if self.isRunning() else 0

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
        # do it earlier to reduce precision error
        timeThisRound = self._singled.get()
        if self.isRunning():
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

    def count_generator_consumption(self, it):
        while True:
            try:
                self.start()
                ret = next(it)
                self.stop().countcycle()
                yield ret
            except StopIteration:
                self.stop().countcycle()
                break


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
