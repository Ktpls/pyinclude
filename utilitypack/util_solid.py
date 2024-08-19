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
import regex
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


def UnfinishedWrapper(msg=None) -> typing.Callable[..., typing.Any]:
    if callable(msg):
        # calling without parens, works both on a class and a function
        foo = msg
        return UnfinishedWrapper()(foo)

    default_msg = "Unfinished"
    if msg is None:
        msg = default_msg

    def f2(foo):
        def f3(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise NotImplementedError(msg)

        return f3

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


class Logger:
    def __init__(self, path):
        self.path = path
        # wont fail
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.f = open(path, "wb+")

    def log(self, content):
        self.f.write((content + "\n").encode("utf8"))
        # self.f.flush()

    def __del__(self):
        self.f.close()

    def __call__(self, content):
        self.log(content)


def QuickSummonCard(inteprob):
    # faster summon using division
    pos = random.random() * inteprob[-1]
    section = [0, len(inteprob)]

    def compare(n):
        # compare pos with section[n]
        if pos > inteprob[n]:
            return 1
        else:  # pos<=section[n]
            if pos > (inteprob[n - 1] if n >= 1 else 0):
                return 0
            else:
                return -1

    while True:
        mid = int((section[1] + section[0]) * 0.5)
        compresult = compare(mid)
        if compresult == 1:
            section[0] = mid + 1
        elif compresult == -1:
            section[1] = mid
        else:  # compresult==0
            return mid


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
        if implType == StoppableProcess:
            raise NotImplementedError("doesn't work")

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


class StoppableProcess(StoppableSomewhat):
    class StoppableOnlyOnceProcess(multiprocessing.Process):
        def __init__(self, sp: "StoppableProcess", args, kwargs) -> None:
            multiprocessing.Process.__init__(self, args=args, kwargs=kwargs)
            self.sp = sp  # read only. cuz cant write

        # override
        def run(self, *args, **kwargs):
            try:
                result = self.sp.foo(*args, **kwargs)
            except Exception as e:
                if self.sp.strategy_error == StoppableThread.StrategyError.raise_error:
                    raise e
                elif (
                    self.sp.strategy_error == StoppableThread.StrategyError.print_error
                ):
                    traceback.print_exc()
                elif self.sp.strategy_error == StoppableThread.StrategyError.ignore:
                    pass
            # cant return result now
            # return result

    def __init__(
        self,
        strategy_runonrunning: "StoppableThread.StrategyRunOnRunning" = None,
        strategy_error: "StoppableThread.StrategyError" = None,
    ):
        StoppableSomewhat.__init__(self, strategy_runonrunning, strategy_error)
        self._stop_event = multiprocessing.Event()
        self.result = None
        self.submit: "StoppableProcess.StoppableOnlyOnceProcess" = None

    def foo(self):
        # This is a placeholder for the method to be overridden by the inherited class
        pass

    def isRunning(self) -> bool:
        return self.submit is not None and self.submit.is_alive()

    @FunctionalWrapper
    def go(self, *args, **kwargs):
        if self.isRunning():
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
        self._stop_event.clear()
        self.submit = self.StoppableOnlyOnceProcess(self, args=args, kwargs=kwargs)
        self.submit.start()

    def stop(self):
        if not self.isRunning():
            return
        self._stop_event.set()
        self.submit.join()
        # self.result=...
        self.submit = None

    def timeToStop(self) -> bool:
        return self._stop_event.is_set()


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


class Pipe:
    value: typing.Any = None

    def __init__(self, initValue: typing.Any = None, printStep: bool = False) -> None:
        self.printStep = printStep
        self.set(initValue)

    def get(self) -> typing.Any:
        return self.value

    def set(self, val: typing.Any) -> None:
        self.value = val
        if self.printStep:
            print(self.value)
        return self

    @FunctionalWrapper
    def do(self, foo: typing.Callable[[typing.Any], typing.Any]) -> "Pipe":
        self.set(foo(self.get()))

    def __repr__(self) -> str:
        return self.get().__repr__()


class Stream:
    content: list
    actions: list

    def __init__(self, iter: list | tuple | dict) -> None:
        if isinstance(iter, (list, tuple)):
            self.content = iter
        elif isinstance(iter, dict):
            self.content = iter.items()
        else:
            raise TypeError("iter must be list|tuple|dict")

    def sort(self, pred: typing.Callable[[typing.Any, typing.Any], int]):
        self.content.sort(key=functools.cmp_to_key(pred))
        return self

    def peek(self, pred: typing.Callable[[typing.Any], None]):
        for i in self.content:
            pred(i)
        return self

    def filter(self, pred: typing.Callable[[typing.Any], bool]):
        self.content = list(filter(pred, self.content))
        return self

    def map(self, pred: typing.Callable[[typing.Any], typing.Any]):
        self.content = list(map(pred, self.content))
        return self

    def flatMap(self, pred: "typing.Callable[[typing.Any],Stream]"):
        self.content = list(
            itertools.chain.from_iterable([s.content for s in map(pred, self.content)])
        )
        return self

    def distinct(self):
        self.content = Deduplicate(self.content)
        return self

    class Collector:
        def __init__(self, collectImpl):
            self.collectImpl = collectImpl

        def do(self, stream):
            return self.collectImpl(stream)

        @staticmethod
        def toList():
            return Stream.Collector(lambda stream: list(stream.content))

        @staticmethod
        def toDict(keyPred, valuePred):
            return Stream.Collector(
                lambda stream: {keyPred(i): valuePred(i) for i in stream.content}
            )

        @staticmethod
        def groupBy(keyPred):
            return Stream.Collector(
                lambda stream: {
                    key: list(group)
                    for key, group in itertools.groupby(
                        sorted(stream.content, key=keyPred), key=keyPred
                    )
                }
            )

    def collect(self, collector: "Stream.Collector"):
        return collector.do(self)


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


class FSMUtil:

    class ParseError(Exception): ...

    class TokenTypeLike(enum.Enum): ...

    class TokenMatcher:
        def tryMatch(self, s: str, i: int) -> typing.Union[None, "FSMUtil.Token"]: ...

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
            token = FSMUtil.getToken(self.s, self._indexTextTokenizing, self.matchers)
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


class expparser:
    """
    TODO:
    numlike
        tensor operator
        string operator
        named parameter in function call
            foo(1, 2, 3, a=1, b=2)
        delayed evaluation optimization
    """

    @dataclasses.dataclass
    class evaluator:
        class EvalType(enum.Enum):
            literal = 0
            operator = 1
            func = 2
            var = 3
            lyst = 4

        type: EvalType
        value: typing.Any
        para: typing.Any

        @staticmethod
        def ofOpr(opr: "expparser._OprType", para):
            return expparser.evaluator(expparser.evaluator.EvalType.operator, opr, para)

        @staticmethod
        def ofLiteral(literal):
            return expparser.evaluator(
                expparser.evaluator.EvalType.literal, literal, None
            )

        @staticmethod
        def ofFunc(func, para):
            return expparser.evaluator(expparser.evaluator.EvalType.func, func, para)

        @staticmethod
        def ofVar(var):
            return expparser.evaluator(expparser.evaluator.EvalType.var, var, None)

        @staticmethod
        def ofList(var):
            return expparser.evaluator(expparser.evaluator.EvalType.lyst, var, None)

        def eval(self, var=dict(), func=dict()):
            if self.type == expparser.evaluator.EvalType.literal:
                return self.value
            elif self.type == expparser.evaluator.EvalType.operator:
                para = [p.eval(var, func) for p in self.para]
                return self.value.do(para)
            elif self.type == expparser.evaluator.EvalType.func:
                assert self.value in func
                para = [p.eval(var, func) for p in self.para]
                return func[self.value](*para)
            elif self.type == expparser.evaluator.EvalType.var:
                assert self.value in var
                return var[self.value]
            elif self.type == expparser.evaluator.EvalType.lyst:
                value = [p.eval(var, func) for p in self.value]
                return value

        def __repr__(self, indentLvl: int = 0) -> str:
            indent = " " * 4 * indentLvl
            tipe = f"{indent}{self.type}, "
            if self.type == expparser.evaluator.EvalType.lyst:
                val = f"list\n"
                child = "".join(
                    [p.__repr__(indentLvl=indentLvl + 1) for p in self.value]
                )
            else:
                val = f"{self.value}\n"
                child = ""
                if self.para is not None:
                    child = "".join(
                        [p.__repr__(indentLvl=indentLvl + 1) for p in self.para]
                    )
                else:
                    child = ""
            return tipe + val + child

    class _TokenType(enum.Enum):
        NUMLIKE = 1
        OPR = 2
        BRA = 3
        KET = 4
        EOF = 5
        IDR = 6
        SPACE = 7
        COMMA = 8
        COMMENT = 9

    class _State(enum.Enum):
        START = 1
        NUM = 2
        NEG = 3
        OPR = 4
        END = 5
        IDR = 6

    @dataclasses.dataclass
    class _OprPriorityLeap:
        pos: int
        pribefore: int
        priafter: int

    class _NumLikeUnionUtil:
        class NumLikeException(Exception):
            pass

        class NumLikeType(enum.Enum):
            NUM = 0
            STR = 1
            LIST = 2
            BOOL = 3
            NONE = 4

        @staticmethod
        def TypeOf(nl):
            if isinstance(nl, str):
                return expparser._NumLikeUnionUtil.NumLikeType.STR
            elif isinstance(nl, typing.Iterable):
                return expparser._NumLikeUnionUtil.NumLikeType.LIST
            elif isinstance(nl, float):
                return expparser._NumLikeUnionUtil.NumLikeType.NUM
            elif isinstance(nl, bool):
                return expparser._NumLikeUnionUtil.NumLikeType.BOOL
            elif nl is None:
                return expparser._NumLikeUnionUtil.NumLikeType.NONE
            else:
                raise expparser._NumLikeUnionUtil.NumLikeException()

        # imexplicit conversion
        @staticmethod
        def ToNum(nl):
            t = expparser._NumLikeUnionUtil.TypeOf(nl)
            if t == expparser._NumLikeUnionUtil.NumLikeType.NUM:
                return nl
            elif t == expparser._NumLikeUnionUtil.NumLikeType.BOOL:
                return 1.0 if nl else 0.0
            else:
                raise expparser._NumLikeUnionUtil.NumLikeException()

        @staticmethod
        def ToList(nl):
            t = expparser._NumLikeUnionUtil.TypeOf(nl)
            if t == expparser._NumLikeUnionUtil.NumLikeType.LIST:
                return list(nl)
            elif t == expparser._NumLikeUnionUtil.NumLikeType.STR:
                return [nl]
            elif t == expparser._NumLikeUnionUtil.NumLikeType.NUM:
                return [nl]
            elif t == expparser._NumLikeUnionUtil.NumLikeType.BOOL:
                return [nl]
            elif t == expparser._NumLikeUnionUtil.NumLikeType.NONE:
                return [nl]
            else:
                raise expparser._NumLikeUnionUtil.NumLikeException()

        @staticmethod
        def ToBool(nl):
            t = expparser._NumLikeUnionUtil.TypeOf(nl)
            if t == expparser._NumLikeUnionUtil.NumLikeType.NUM:
                return nl > 0
            elif t == expparser._NumLikeUnionUtil.NumLikeType.BOOL:
                return nl
            else:
                raise expparser._NumLikeUnionUtil.NumLikeException()

        @staticmethod
        def ToProperFormFromAny(nl):
            # the adaptor for data to be proper
            # after which there will be no int form, they all stored as float
            # this will deal with more situations, so TypeOf is not proper here
            if isinstance(nl, str):
                return nl
            elif isinstance(nl, typing.Iterable):
                return list(nl)
            elif isinstance(nl, float):
                return nl
            elif isinstance(nl, bool):
                # isinstance(True, int)==True
                return nl
            elif isinstance(nl, int):
                return float(nl)
            elif nl is None:
                return nl
            else:
                raise expparser._NumLikeUnionUtil.NumLikeException()

    class _ParseException(Exception):
        pass

    @staticmethod
    def unpackParaArray(f):
        def f2(a):
            if (
                expparser._NumLikeUnionUtil.TypeOf(a)
                == expparser._NumLikeUnionUtil.NumLikeType.LIST
            ):
                return f(*a)
            else:
                return f(a)

        return f2

    @staticmethod
    def CList(a):
        if (
            expparser._NumLikeUnionUtil.TypeOf(a)
            == expparser._NumLikeUnionUtil.NumLikeType.LIST
        ):
            return a
        else:
            return [a]

    class _OprException(Exception):
        pass

    class _OprType(enum.Enum):
        UNSPECIFIED = 0
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        POW = 5
        NEG = 7  # available by manual transfering from sub
        NEQ = 8
        EQ = 9
        GT = 10
        GE = 11
        LT = 12
        LE = 13
        NOT = 14
        AND = 15
        OR = 16
        XOR = 17

        @staticmethod
        def throw_opr_exception(s):
            raise expparser._OprException(f"bad opr {s}")

        def getPriority(self):
            if self in [
                expparser._OprType.OR,
                expparser._OprType.AND,
                expparser._OprType.XOR,
            ]:
                return 1
            elif self in [
                expparser._OprType.GT,
                expparser._OprType.GE,
                expparser._OprType.LT,
                expparser._OprType.LE,
                expparser._OprType.EQ,
                expparser._OprType.NEQ,
            ]:
                return 2
            elif self in [expparser._OprType.ADD, expparser._OprType.SUB]:
                return 3
            elif self in [expparser._OprType.MUL, expparser._OprType.DIV]:
                return 4
            elif self == expparser._OprType.POW:
                return 5
            elif self in [expparser._OprType.NEG, expparser._OprType.NOT]:
                # unaries
                return 99
            else:
                expparser._OprType.throw_opr_exception(self)

        @staticmethod
        def fromStr(s):
            dict = {
                "+": expparser._OprType.ADD,
                "-": expparser._OprType.SUB,
                "*": expparser._OprType.MUL,
                "/": expparser._OprType.DIV,
                "^": expparser._OprType.POW,
                "=": expparser._OprType.EQ,
                "!=": expparser._OprType.NEQ,
                ">": expparser._OprType.GT,
                ">=": expparser._OprType.GE,
                "<": expparser._OprType.LT,
                "<=": expparser._OprType.LE,
                "!": expparser._OprType.NOT,
                "&": expparser._OprType.AND,
                "|": expparser._OprType.OR,
                "^^": expparser._OprType.XOR,
            }
            if s not in dict:
                expparser._OprType.throw_opr_exception(s)
            return dict[s]

        def do(self, arg):
            if self == expparser._OprType.ADD:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) + expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.SUB:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) - expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.MUL:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) * expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.DIV:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) / expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.POW:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) ** expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.NEG:
                return -expparser._NumLikeUnionUtil.ToNum(arg[0])
            elif self == expparser._OprType.NEQ:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) != expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.EQ:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) == expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.GT:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) > expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.GE:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) >= expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.LT:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) < expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.LE:
                return expparser._NumLikeUnionUtil.ToNum(
                    arg[0]
                ) <= expparser._NumLikeUnionUtil.ToNum(arg[1])
            elif self == expparser._OprType.NOT:
                return not expparser._NumLikeUnionUtil.ToBool(arg[0])
            elif self == expparser._OprType.AND:
                return expparser._NumLikeUnionUtil.ToBool(
                    arg[0]
                ) and expparser._NumLikeUnionUtil.ToBool(arg[1])
            elif self == expparser._OprType.OR:
                return expparser._NumLikeUnionUtil.ToBool(
                    arg[0]
                ) or expparser._NumLikeUnionUtil.ToBool(arg[1])
            elif self == expparser._OprType.XOR:
                return expparser._NumLikeUnionUtil.ToBool(
                    arg[0]
                ) ^ expparser._NumLikeUnionUtil.ToBool(arg[1])
            else:
                expparser._OprType.throw_opr_exception(self)

        def isUnary(self):
            return self in [expparser._OprType.NEG, expparser._OprType.NOT]

    _matcherList = [
        # comment "/" out priored the operator "/"
        FSMUtil.RegexpTokenMatcher(exp=r"^//.+?\n", type=_TokenType.COMMENT),
        FSMUtil.RegexpTokenMatcher(exp=r"^/\*.+?\*/", type=_TokenType.COMMENT),
        FSMUtil.RegexpTokenMatcher(
            exp=r"^(<=)|(>=)|(\^\^)|(!=)", type=_TokenType.OPR
        ),  # two width operator, match before single widthed ones to get priority
        FSMUtil.RegexpTokenMatcher(
            exp=r"^[*/+\-^=<>&|]", type=_TokenType.OPR
        ),  # single width operator
        FSMUtil.RegexpTokenMatcher(exp=r"^[0-9]+(\.[0-9]+)?", type=_TokenType.NUMLIKE),
        # cant process r'"\\"' properly, but simply ignore it
        FSMUtil.RegexpTokenMatcher(exp=r'^".+?(?<!\\)"', type=_TokenType.NUMLIKE),
        FSMUtil.RegexpTokenMatcher(exp=r"^[A-Za-z_][A-Za-z0-9_]*", type=_TokenType.IDR),
        FSMUtil.RegexpTokenMatcher(exp=r"^\(", type=_TokenType.BRA),
        FSMUtil.RegexpTokenMatcher(exp=r"^\)", type=_TokenType.KET),
        FSMUtil.RegexpTokenMatcher(exp=r"^,", type=_TokenType.COMMA),
        FSMUtil.RegexpTokenMatcher(exp=r"^$", type=_TokenType.EOF),
        FSMUtil.RegexpTokenMatcher(exp=r"^[\s\r\n\t]+", type=_TokenType.SPACE),
    ]

    @staticmethod
    def _NextToken(s, i=0):
        def getNextToken(s, i):
            while True:
                ret = FSMUtil.getToken(s, i, expparser._matcherList)
                if ret.type not in [
                    expparser._TokenType.SPACE,
                    expparser._TokenType.COMMENT,
                ]:
                    break
                else:
                    i = ret.end
            if ret.type == expparser._TokenType.NUMLIKE:
                # here is only num and str without other numlike types
                if len(ret.value) >= 2 and ret.value[0] == '"':
                    # is str
                    ret.start = ret.start
                    ret.value = ret.value[1:-1].replace(r"\"", '"')
                else:
                    ret.value = float(ret.value)
            elif ret.type == expparser._TokenType.OPR:
                ret.value = expparser._OprType.fromStr(ret.value)
            return ret

        return getNextToken(s, i)

    @dataclasses.dataclass
    class _ExpParserResult:
        val: typing.Any
        end: int
        endedby: "expparser._TokenType"

    @staticmethod
    def _expparse_recursive__comma_collector_wrapper(s, i=0):
        nextval = expparser._expparse_recursive(s, i)
        if nextval.endedby == expparser._TokenType.COMMA:
            vallist = [nextval.val]
            while True:
                nextval = expparser._expparse_recursive(s, nextval.end)
                vallist.append(nextval.val)
                if nextval.endedby != expparser._TokenType.COMMA:
                    break
            retval = expparser.evaluator.ofList(vallist)
        else:
            retval = (
                nextval.val
            )  # should be converted into evaluator by expparse_recursive
        return expparser._ExpParserResult(
            val=retval, end=nextval.end, endedby=nextval.endedby
        )

    @staticmethod
    def _expparse_recursive(
        s,
        startPos=0,
    ):
        # fsm fields
        state = expparser._State.START
        token: FSMUtil.Token = None
        # never modify peekToken
        peekToken = expparser._NextToken(s, startPos)

        # buffer
        tokenList: list[FSMUtil.Token] = list()

        # for operator priority
        oprRisingBeginPosList: list[expparser._OprPriorityLeap] = list()

        def RaiseTokenException(token: FSMUtil.Token):
            raise expparser._ParseException(
                f'unexpected {token.type}("{token.value}") at {token.start}'
            )

        def ClearOprSectionAssumingPeer(begin, end):
            nonlocal tokenList, oprRisingBeginPosList
            # cache the section to use easily pop and push
            section = tokenList[begin:end]
            token1st = section[0]
            if token1st.type == expparser._TokenType.OPR:
                assert token1st.value.isUnary()
                # assert unary operators are never at the same priority as other multioperand operators,
                # so i can deal them seperately
                val = section[-1]
                assert val.type == expparser._TokenType.NUMLIKE
                i = len(section) - 2
                while True:
                    # cleaning backwards, which makes it right-nested as a tree
                    if i < 0:
                        break
                    opr = section[i]
                    i -= 1
                    assert opr.type == expparser._TokenType.OPR
                    assert opr.value.isUnary()
                    val.value = expparser.evaluator.ofOpr(opr.value, [val.value])
            else:
                i = 0
                val = section[0]
                i += 1
                while True:
                    if i >= len(section):
                        break
                    opr = section[i]
                    i += 1
                    assert i < len(section)
                    val2 = section[i]
                    i += 1
                    val.value = expparser.evaluator.ofOpr(
                        opr.value, [val.value, val2.value]
                    )
            tokenList = tokenList[:begin] + [val] + tokenList[end:]
            RemapToken()
            oprRisingBeginPosList.pop()

        def AddNewVirtualTokenValuedByCalculation(
            subresult: expparser._ExpParserResult,
        ):
            nonlocal token, peekToken, state, tokenList
            tokenList.append(
                FSMUtil.Token(
                    expparser._TokenType.NUMLIKE,
                    subresult.val,
                    token.start,
                    subresult.end,
                )
            )
            RemapToken()
            peekToken = expparser._NextToken(s, subresult.end)

        def DealWithBra():
            nonlocal token, peekToken, state, tokenList
            if peekToken.type == expparser._TokenType.KET:
                """
                one empty list
                cant eval with expparse recursive,
                cuz it returns with None if start follows by eof instantly
                in this case () can be confused with List(None)
                """
                subresult = expparser._ExpParserResult(
                    expparser.evaluator.ofList([]),
                    peekToken.end,
                    expparser._TokenType.KET,
                )
            else:
                subresult = expparser._expparse_recursive__comma_collector_wrapper(
                    s, token.end
                )
            tokenList.pop()  # remove the bra
            AddNewVirtualTokenValuedByCalculation(subresult)
            state = expparser._State.NUM

        def DealWithIdentifier():
            nonlocal token, peekToken, state, tokenList
            if peekToken.type == expparser._TokenType.BRA:
                # its a call
                fooName = token.value
                # manually move on
                MoveForwardToNextToken()
                DealWithBra()
                para = tokenList[-1].value
                if para.type == expparser.evaluator.EvalType.lyst:
                    # unpack list evaluator to param
                    # it could possibly be the single param, but which is actually a list, like f(list(1,2,3))
                    # and it will be unpacked weirdly here
                    para = para.value
                else:
                    # single param call, to standard param list form
                    para = [para]
                value = expparser.evaluator.ofFunc(fooName, para)
                tokenList[-1].value = value
                tokenList = tokenList[:-2] + tokenList[-1:]  # remove the func name
                RemapToken()
            else:
                # its a var
                # overwrite the identifier with num
                token.value = expparser.evaluator.ofVar(token.value)
                token.type = expparser._TokenType.NUMLIKE
            state = expparser._State.NUM

        def MoveForwardToNextToken():
            nonlocal token, peekToken, tokenList
            token = peekToken
            peekToken = expparser._NextToken(s, token.end)
            tokenList.append(token)
            # RemapToken() # not essential here
            # modifying token also applies on tokenList[-1]

        def RemapToken():
            nonlocal tokenList, token
            # reset token to last of token list after modifying tokenList structure
            if len(tokenList) != 0:
                token = tokenList[-1]
            else:
                # calling any member on this will cause exception
                token = None

        def doWhenReadNewOpr():
            nonlocal state, oprRisingBeginPosList, token, tokenList
            state = expparser._State.OPR
            lastOprPrior = (
                oprRisingBeginPosList[-1].priafter
                if len(oprRisingBeginPosList) > 0
                else 0
            )
            opr = token.value.getPriority()
            if opr > lastOprPrior:
                oprRisingBeginPosList.append(
                    expparser._OprPriorityLeap(len(tokenList) - 1, lastOprPrior, opr)
                )  # len(tokenList) - 1 for opr
            elif opr < lastOprPrior:
                while True:
                    lastOprPrior = (
                        oprRisingBeginPosList[-1].priafter
                        if len(oprRisingBeginPosList) > 0
                        else 0
                    )
                    if opr >= lastOprPrior:
                        break
                        # clear since last rising, until flat
                        # len(tokenList)-1 since the lowering opr has been appended
                    """
                    for unary, the token before first opr is unwanted to calc
                    for binary, its necessary
                    """
                    ClearOprSectionAssumingPeer(
                        (
                            oprRisingBeginPosList[-1].pos
                            if tokenList[oprRisingBeginPosList[-1].pos].value.isUnary()
                            else oprRisingBeginPosList[-1].pos - 1
                        ),
                        len(tokenList) - 1,  # the new opr
                    )
                if opr > lastOprPrior:
                    # new opr is the new rising
                    oprRisingBeginPosList.append(
                        expparser._OprPriorityLeap(
                            len(tokenList) - 1,
                            lastOprPrior,
                            opr,
                        )
                    )

        def dealWithExpressionEndSign():
            nonlocal state, oprRisingBeginPosList, token, tokenList
            expendpos = token.end
            endtype = token.type
            tokenList.pop()  # remove the eof or ket or comma, end sign anyway
            RemapToken()
            if len(tokenList) == 0:
                # empty
                val = expparser.evaluator.ofLiteral(None)
            else:
                # clear all
                while True:
                    if len(oprRisingBeginPosList) == 0:
                        break
                    ClearOprSectionAssumingPeer(
                        (
                            oprRisingBeginPosList[-1].pos
                            if tokenList[oprRisingBeginPosList[-1].pos].value.isUnary()
                            else oprRisingBeginPosList[-1].pos - 1
                        ),
                        len(tokenList),
                    )
                val = tokenList[-1].value
            return expparser._ExpParserResult(val, expendpos, endtype)

        # the fsm illustrated in notebook
        while True:
            MoveForwardToNextToken()
            if state == expparser._State.START or state == expparser._State.OPR:
                # expecting numlike, but possible to meet bra, identifier, or unary operator, eof, comma, ket
                if token.type == expparser._TokenType.BRA:
                    DealWithBra()
                elif token.type == expparser._TokenType.NUMLIKE:
                    token.value = expparser.evaluator.ofLiteral(token.value)
                    state = expparser._State.NUM
                elif token.type == expparser._TokenType.IDR:
                    DealWithIdentifier()
                elif token.type == expparser._TokenType.OPR:
                    # maybe unary operator
                    if token.value == expparser._OprType.SUB:
                        # to neg
                        tokenList[-1].value = expparser._OprType.NEG
                    if not token.value.isUnary():
                        RaiseTokenException(token)
                    doWhenReadNewOpr()
                elif token.type in [
                    expparser._TokenType.EOF,
                    expparser._TokenType.KET,
                    expparser._TokenType.COMMA,
                ]:
                    # return
                    state = expparser._State.END
                    return dealWithExpressionEndSign()
                else:
                    RaiseTokenException(token)
            elif state == expparser._State.NUM:
                if token.type == expparser._TokenType.OPR:
                    doWhenReadNewOpr()
                elif token.type in [
                    expparser._TokenType.EOF,
                    expparser._TokenType.KET,
                    expparser._TokenType.COMMA,
                ]:
                    # return
                    state = expparser._State.END
                    return dealWithExpressionEndSign()
                else:
                    RaiseTokenException(token)
            elif state == expparser._State.END:
                RaiseTokenException(token)

    @staticmethod
    def elementparse(s):
        i = 0
        tokenList: list[FSMUtil.Token] = []
        while True:
            token = expparser._NextToken(s, i)
            tokenList.append(token)
            i = token.end
            if token.type == expparser._TokenType.EOF:
                break
        var = []
        func = []
        for i, tk in enumerate(tokenList):
            if tk.type == expparser._TokenType.IDR:
                # peek
                # wont index overflow cuz there is eof and eof is not identifier
                if tokenList[i + 1].type == expparser._TokenType.BRA:
                    func.append(tk.value)
                else:
                    var.append(tk.value)
        return var, func

    @staticmethod
    def expparse(s, var=dict(), func=dict()):
        return expparser._expparse_recursive__comma_collector_wrapper(s).val.eval(
            var, func
        )

    @staticmethod
    def compile(s) -> evaluator:
        return expparser._expparse_recursive__comma_collector_wrapper(s).val

    class Utils:
        class NonOptionalException(Exception):
            pass

        class NonOptional:
            @staticmethod
            def checkParamListIfNonOptional(paramList):
                for i, p in enumerate(paramList):
                    if isinstance(p, expparser.Utils.NonOptional):
                        raise expparser.Utils.NonOptionalException(
                            f"Nonoptional parameter {i} unspecified"
                        )
                return paramList

        @staticmethod
        def OptionalFunc(defaultParam: list, func: typing.Callable):
            def newFunc(*param):
                newParam = [Coalesce(a, d) for a, d in zip(param, defaultParam)]
                if len(param) < len(defaultParam):
                    newParam.extend(defaultParam[len(param) :])
                expparser.Utils.NonOptional.checkParamListIfNonOptional(newParam)
                return func(*newParam)

            return newFunc

    BasicFunctionLib = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "abs": abs,
        "sign": lambda x: 1 if x > 0 else -1 if x < 0 else 0,
        "floor": math.floor,
        "ceil": math.ceil,
        "neg": lambda x: -x,
        "iif": lambda cond, x, y: x if expparser._NumLikeUnionUtil.ToBool(cond) else y,
        "eq": lambda x, y, ep=0.001: abs(x - y) < ep,
        "StrEq": lambda x, y: x == y,
        "CStr": str,
        "CNum": float,
        "CBool": bool,
        "CList": CList,
    }

    BasicConstantLib = {
        "e": math.e,
        "pi": math.pi,
        "true": True,
        "false": False,
        "none": None,
    }


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
        self._stagedtime = 0
        self._cycle = 0
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
        self._stagedtime += self._timeCurrentlyCounting()
        self._starttime = None
        return self

    def isRunning(self):
        return self._starttime is not None

    def time(self):
        return self._stagedtime + self._timeCurrentlyCounting()

    def aveTime(self):
        return self.time() / (self._cycle if self._cycle > 0 else 1)

    def _timeCurrentlyCounting(self):
        return time.perf_counter() - self._starttime if self.isRunning() else 0


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


def LongDelay(t, interval=0.5):
    round = math.ceil(t / interval)
    for i in range(round):
        time.sleep(interval)


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


def WrapperOfMultiLineText(s):
    """
        to process text like this
        var=WrapperOfMultilLineText(${threeQuotes}
    your
    multiline
    content
    here
    ${threeQuotes})
    """
    return s[1:-1]


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
    def __GetEmptyInstance(cls):
        args = inspect.getargs(cls.__init__.__code__)
        if len(args) > 1:
            # found init with arg more than self
            inst = object.__new__(cls)
            fields = BeanUtil.__GetFields(cls)
            for name, taipe in fields.items():
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
        dstobj = BeanUtil.__GetEmptyInstance(dst)
        BeanUtil.__DictOrObj2DictOrObjCopy(src, dstobj, option)
        return dstobj

    @staticmethod
    def copyProperties(src, dst: object, option: "BeanUtil.Option" = Option()):
        if inspect.isclass(dst):
            return BeanUtil.__DictOrObj2ClassCopy(src, dst, option)
        else:
            BeanUtil.__DictOrObj2DictOrObjCopy(src, dst, option)


@EasyWrapper
def AllOptionalInit(clz):
    oldInit = clz.__init__
    kws = [k for k in oldInit.__annotations__.keys() if k != "return"]

    def initNone(self, **kwargs):
        nonlocal oldInit, kws
        for k in kws:
            if k not in kwargs:
                kwargs[k] = None
        oldInit(self, **kwargs)

    clz.__init__ = initNone
    return clz


class Container:
    __content = None

    def get(self):
        return self.__content

    def set(self, newContent):
        self.__content = newContent

    def isEmpty(self):
        return self.__content is None


class Switch:
    def __init__(self, onSetOn=None, onSetOff=None, initial=False):
        self.__value = initial
        self.onSetOn = onSetOn
        self.onSetOff = onSetOff

    def on(self):
        self.__value = True
        if self.onSetOn is not None:
            self.onSetOn()

    def off(self):
        self.__value = False
        if self.onSetOff is not None:
            self.onSetOff()

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


def printAndRet(val):
    print(val)
    return val


def PathNormalize(path: str):
    return path.replace("\\", "/")


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
                        and all(str.isdigit(i) and 255 >= int(i) >= 0 for i in lHost)
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


@dataclasses.dataclass
class UrlFullResolutionLazy:
    class Uncalculated: ...

    url: str | None
    ensureAllCalculated: bool = False
    protocol: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    host: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    path: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    param: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    secondaryHost: "str | None | UrlFullResolutionLazy.Uncalculated" = (
        dataclasses.field(init=False, default=Uncalculated)
    )
    baseHost: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    domain: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    port: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    folder: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    fileName: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )
    extName: "str | None | UrlFullResolutionLazy.Uncalculated" = dataclasses.field(
        init=False, default=Uncalculated
    )

    def __getattribute__(self, name: str):
        if name in [
            "protocol",
            "host",
            "path",
            "param",
            "secondaryHost",
            "baseHost",
            "domain",
            "port",
            "folder",
            "fileName",
            "extName",
        ]:
            if (
                UrlFullResolutionLazy._rawGet(self, name)
                == UrlFullResolutionLazy.Uncalculated
            ):
                self._parseStepGlobally()
                if name in ["protocol", "host", "path", "param"]:
                    return UrlFullResolutionLazy._rawGet(self, name)
                if name in ["secondaryHost", "baseHost", "domain", "port"]:
                    self._parseStepHost()
                    return UrlFullResolutionLazy._rawGet(self, name)
                if name in ["folder", "fileName", "extName"]:
                    self._parseStepPath()
                    return UrlFullResolutionLazy._rawGet(self, name)
        return UrlFullResolutionLazy._rawGet(self, name)

    @staticmethod
    def _rawGet(self, name: str):
        return object.__getattribute__(self, name)

    class RegPool:
        globally = regex.compile(
            r"^(?<protcol>[A-Za-z]+://)?(?<host>[^/]+\.[^/.]+)?(?<path>[^?]*)?(?<param>\?.*)?$"
        )
        host = regex.compile(r"^(?<host>[^:]+)(?<port>:\d+)?$")
        path = regex.compile(
            r"^(?<folder>.+?)(?:/(?<fileName>[^/]+(?<extName>\..*)))?$"
        )

    class UnexpectedException(Exception): ...

    def _parseStepGlobally(self):
        if (
            UrlFullResolutionLazy._rawGet(self, "protocol")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "host")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "path")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "param")
            == UrlFullResolutionLazy.Uncalculated
        ):
            protocol, host, path, param = [None] * 4
            if UrlFullResolutionLazy._rawGet(self, "url") is not None:
                matchGlobally = UrlFullResolutionLazy.RegPool.globally.match(self.url)
                if matchGlobally is not None:
                    protocol, host, path, param = matchGlobally.group(
                        "protcol", "host", "path", "param"
                    )
            self.protocol = protocol
            self.host = host
            self.path = path
            self.param = param

    def _parseStepHost(self):
        if (
            UrlFullResolutionLazy._rawGet(self, "port")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "secondaryHost")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "baseHost")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "domain")
            == UrlFullResolutionLazy.Uncalculated
        ):
            secondaryHost, baseHost, domain, port = [None] * 4
            if UrlFullResolutionLazy._rawGet(self, "host") is not None:
                matchHost = UrlFullResolutionLazy.RegPool.host.match(self.host)
                if matchHost is not None:
                    hostNoPort, port = matchHost.group("host", "port")
                    lHost = hostNoPort.split(".")
                    if len(lHost) < 2:
                        raise UrlFullResolutionLazy.UnexpectedException()
                    if not (
                        len(lHost) == 4
                        and all(str.isdigit(i) and 255 >= int(i) >= 0 for i in lHost)
                    ):
                        secondaryHost = ".".join(lHost[0:-2])
                        baseHost = ".".join(lHost[-2:])
                        domain = lHost[-1]
            self.port = port
            self.secondaryHost = secondaryHost
            self.baseHost = baseHost
            self.domain = domain

    def _parseStepPath(self):
        if (
            UrlFullResolutionLazy._rawGet(self, "folder")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "fileName")
            == UrlFullResolutionLazy.Uncalculated
            or UrlFullResolutionLazy._rawGet(self, "extName")
            == UrlFullResolutionLazy.Uncalculated
        ):
            folder, fileName, extName = [None] * 3
            if UrlFullResolutionLazy._rawGet(self, "path") is not None:
                matchPath = UrlFullResolutionLazy.RegPool.path.match(self.path)
                if matchPath is not None:
                    folder, fileName, extName = matchPath.group(
                        "folder", "fileName", "extName"
                    )
            self.folder = folder
            self.fileName = fileName
            self.extName = extName

    def calcAll(self):
        self._parseStepGlobally()
        self._parseStepHost()
        self._parseStepPath()

    def __post_init__(self):
        self.url = PathNormalize(self.url)
        if self.ensureAllCalculated:
            self.calcAll()


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


def mlambda(s: str, _globals=None, _locals=None):
    exp = regex.compile(
        r"^\s*def\s*(?<paraAndType>.*?):\s*\n(?<body>.+)$", flags=regex.DOTALL
    )
    match = exp.match(s)
    if not match:
        raise SyntaxError("function signing syntax error")
    match = match.groupdict()
    paraAndType = match["paraAndType"]
    body = match["body"]

    emptyLine = regex.compile("^\s*(#.*)?$", flags=regex.MULTILINE)

    def fixBodyIndent(body: str):
        lines = body.splitlines()
        originalBaseIndent = None
        for l in lines:
            if emptyLine.match(l):
                continue
            originalBaseIndent = len(l) - len(l.lstrip())
        assert originalBaseIndent is not None, "bad indent"
        newBodyIndent = 2
        for i in range(len(lines)):
            l = lines[i]
            if emptyLine.match(l):
                continue
            for realIndentToTrim in range(0, originalBaseIndent):
                if l[realIndentToTrim] not in (" ", "\t"):
                    break
            lines[i] = " " * newBodyIndent + l[realIndentToTrim:]
        return "\n".join(lines)

    # body = fixBodyIndent(body)

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
    return json.dumps(
        o,
        indent=4,
        ensure_ascii=False,
        default=lambda x: x.__dict__,
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
