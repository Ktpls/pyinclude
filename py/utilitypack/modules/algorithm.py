import dataclasses
import typing
import heapq
import random
import itertools
import collections
from .misc import ComparatorOverloadedByPred


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
    integralLimitMin: typing.Optional[float] = None
    integralLimitMax: typing.Optional[float] = None
    analizerMode: bool = False
    last_error: float = dataclasses.field(default=0, init=False)
    integral: float = dataclasses.field(default=0, init=False)
    frameData: typing.Optional[AnalizerFrameData] = dataclasses.field(
        default=None, init=False
    )

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
                    i.start or 0,
                    i.stop or self._cursize,
                    i.step or 1,
                )
            ]

    def ToList(self):
        return [self[i] for i in range(len(self))]


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
        self.tasks = initialTasks or list()

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
class ChannelSniffer:
    capacity: float = 0
    usage: float = 0
    capacity_follow_lambda: float = 5
    explore_epsilon: float = 0.1

    class ChannelOverloadError(Exception): ...

    # inherit to override this
    def failable(self, *a, **kw): ...

    def acquire(self, usage: float, *a, **kw):
        if not self.expolitable(usage):
            raise ChannelSniffer.ChannelOverloadError
        try:
            r = self.failable(*a, **kw)
            self.usage += usage
            if self.usage > self.capacity:
                self.follow_usage()
            return r
        except ChannelSniffer.ChannelOverloadError:
            self.follow_usage()
            raise

    def release(self, usage: float):
        self.usage -= usage

    def expolitable(self, usage: float):
        target_usage = self.usage + usage
        if target_usage <= self.capacity:
            return True
        if random.random() < self.explore_epsilon:
            return True
        return False

    def follow_usage(self):
        # update capacity estimate on usage touching the limit
        self.capacity += (self.usage - self.capacity) / self.capacity_follow_lambda
