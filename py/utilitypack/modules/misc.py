import asyncio
import collections
import concurrent.futures
import dataclasses
import datetime
import enum
import functools
import inspect
import itertools
import json
import logging
import os
import re
import subprocess
import time
import types
import typing

from .io import EnsureDirectoryExists
from .time import SingleSectionedTimer

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


def ArrayFlatten(iterable, iterableType: tuple[type] = (list, tuple)):
    result = list()
    for item in iterable:
        if isinstance(item, iterableType):
            result.extend(ArrayFlatten(item))
        else:
            result.append(item)
    return result


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
            self.objtype = self.objtype or type(self.obj)
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
                desiredType = dstOp.typeInfo(k) or type(v)
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


def NormalizeIterableOrSingleArgToIterable(arg) -> list:
    if not isinstance(arg, (list, tuple)):
        return [arg]
    return arg


T = typing.TypeVar("T")


class IterableOrSingle(list):
    Annotation = typing.Iterable[T] | T

    @staticmethod
    def adapt(data: "IterableOrSingle.Annotation"):
        return NormalizeIterableOrSingleArgToIterable(data)


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


@EasyWrapper
def RunThis(f: typing.Callable[[], typing.Any], *a, **kw):
    f(*a, **kw)
    return f


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


class MaxRetry:
    def __init__(
        self,
        maxRetry: int = 3,
        succCond: typing.Callable[[], bool] = None,
        errOnMaxRetry=True,
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
        if self.succCond and self.succCond():
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


def NormalizeCrlf(s: str):
    return s.replace("\r\n", "\n").replace("\r", "\n")


class ParamValueUnset:
    # placeholder for param unset, use when None is a meaningful param
    ...


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


class BashProcess:
    # interactive!
    proc: subprocess.Popen = None
    END_OF_COMMAND = "@@@@END_OF_COMMAND@@@@"

    def __init__(self): ...

    def enter(self):
        self.proc = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def exit(self):
        self.proc.stdin.close()
        self.proc.terminate()
        self.proc.wait()

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
        return False

    def send_command(self, command):
        command = f"{command}\necho '{self.END_OF_COMMAND}'\n"
        self.proc.stdin.write(command)
        self.proc.stdin.flush()

        output = []
        while True:
            line = self.proc.stdout.readline()
            if self.END_OF_COMMAND + "\n" == line:
                break
            if not line:
                break
            output.append(line.strip())
        return output


T = typing.TypeVar("T")
F = typing.TypeVar("F")
R = typing.TypeVar("R")
K = typing.TypeVar("K")
V = typing.TypeVar("V")


class Stream(typing.Generic[T], typing.Iterable[T]):
    # copied from superstream 0.2.6 !
    # but with some improvements
    class Collectors:
        @staticmethod
        def join(separator: str = ""):
            return lambda it: separator.join(it)

        @staticmethod
        def ndarray():
            import numpy as np

            return lambda it: np.array(it)

        list = list
        set = set
        set_union = lambda x: set().union(x)

        @staticmethod
        def dict_union():
            def _(x: typing.Iterable[dict[K, V]]):
                r = {}
                for i in x:
                    r.update(i)
                return r

            return _

    @staticmethod
    def UnpackedCalling(
        func: typing.Callable[..., R],
    ) -> typing.Callable[[typing.Tuple[T]], R]:
        return lambda x: func(*x)

    upcl = UnpackedCalling

    def __init__(self, stream: typing.Iterable[T], always_unpack: bool = False):
        self._stream = iter(stream)
        self._always_unpack = always_unpack

    def clone(self, stream: typing.Iterable[T] = None) -> "Stream[T]":
        return Stream(stream=stream or self._stream, always_unpack=self._always_unpack)

    def always_unpack(self, always_unpack: bool = True) -> typing.Self:
        self._always_unpack = always_unpack
        return self

    def _unpacking_considered_pred(
        self,
        pred: typing.Callable[[T], bool] | typing.Callable[..., R],
        unpacking: bool = None,
    ) -> typing.Callable[[T], bool]:
        if unpacking is None:
            unpacking = self._always_unpack
        if self._always_unpack:
            return Stream.UnpackedCalling(pred)
        else:
            return pred

    def __iter__(self):
        return self._stream

    @staticmethod
    def of(*args: T) -> "Stream[T]":
        return Stream(args)

    def map(self, func: typing.Callable[[T], R]) -> "Stream[R]":
        func = self._unpacking_considered_pred(func)
        return self.clone(map(func, self._stream))

    def flat_map(self, func: typing.Callable[[T], "Stream[R]"]) -> "Stream[R]":
        func = self._unpacking_considered_pred(func)
        return self.clone(itertools.chain.from_iterable(map(func, self._stream)))

    def filter(self, func: typing.Callable[[T], bool]) -> "Stream[T]":
        func = self._unpacking_considered_pred(func)
        return self.clone(filter(func, self._stream))

    def for_each(self, func: typing.Callable[[T], None]) -> None:
        func = self._unpacking_considered_pred(func)
        for i in self._stream:
            func(i)

    def peek(self, func: typing.Callable[[T], None]) -> "Stream[T]":
        func = self._unpacking_considered_pred(func)

        def proc(_stream: Stream[T]) -> typing.Generator[T, None, None]:
            for i in _stream:
                func(i)
                yield i

        return self.clone(proc(self._stream))

    def distinct(self, pred: typing.Callable[[T], R] = lambda x: x) -> "Stream[T]":
        def generator(_stream: Stream[T]) -> typing.Generator[T, None, None]:
            existeds = set()
            for i in _stream:
                if pred(i) not in existeds:
                    existeds.add(pred(i))
                    yield i

        return self.clone(generator(self._stream))

    def sorted(self, func=None, reverse=False) -> "Stream[T]":
        if func:
            func = self._unpacking_considered_pred(func)
        return self.clone(sorted(self._stream, key=func, reverse=reverse))

    def count(self) -> int:
        cnt = itertools.count()
        collections.deque(zip(self._stream, cnt), maxlen=0)
        return next(cnt)

    def sum(self) -> "T":
        return sum(self._stream)

    def group_by(self, func: typing.Callable[[T], K]) -> dict[K, list[T]]:
        func = self._unpacking_considered_pred(func)
        groups = {}
        for i in self._stream:
            groups.setdefault(func(i), []).append(i)
        return groups

    def reduce(
        self, func: typing.Callable[[T, T], T], initial: T = None
    ) -> typing.Optional[T]:
        func = self._unpacking_considered_pred(func)
        if initial is not None:
            return functools.reduce(func, self._stream, initial)
        else:
            try:
                return functools.reduce(func, self._stream)
            except TypeError:
                return None

    def limit(self, max_size: int) -> "Stream[T]":
        return self.clone(itertools.islice(self._stream, max_size))

    def skip(self, n: int) -> "Stream[T]":
        return self.clone(itertools.islice(self._stream, n, None))

    def min(
        self, func: typing.Callable[[T], typing.Any] = lambda x: x, default: T = None
    ) -> typing.Optional[T]:
        """
        :param default: use default value when stream is empty
        :param key: at lease supported __lt__ method
        """
        func = self._unpacking_considered_pred(func)
        return min(self._stream, key=func, default=default)

    def max(
        self, func: typing.Callable[[T], typing.Any] = lambda x: x, default: T = None
    ) -> typing.Optional[T]:
        """
        :param default: use default value when stream is empty
        :param key: at lease supported __lt__ method
        """
        func = self._unpacking_considered_pred(func)
        return max(self._stream, key=func, default=default)

    def minmax(
        self, func: typing.Callable[[T], typing.Any] = lambda x: x, default: T = None
    ) -> typing.Optional[T]:
        """
        :param default: use default value when stream is empty
        :param key: at lease supported __lt__ method
        """
        func = self._unpacking_considered_pred(func)
        mini = maxi = None
        for i in self._stream:
            keyi = func(i)
            if mini is None or keyi < mini:
                mini = keyi
            if maxi is None or keyi > maxi:
                maxi = keyi
        return mini, maxi

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
        func = self._unpacking_considered_pred(func)
        return any(map(func, self._stream))

    def all_match(self, func: typing.Callable[[T], bool]) -> bool:
        func = self._unpacking_considered_pred(func)
        return all(map(func, self._stream))

    def none_match(self, func: typing.Callable[[T], bool]) -> bool:
        func = self._unpacking_considered_pred(func)
        return not self.any_match(func)

    def to_list(self) -> list[T]:
        return list(self._stream)

    def to_set(self) -> set[T]:
        return set(self._stream)

    def to_dict(
        self, k: typing.Callable[[T], K], v: typing.Callable[[T], V]
    ) -> dict[K, V]:
        k = self._unpacking_considered_pred(k)
        v = self._unpacking_considered_pred(v)
        return {k(i): v(i) for i in self._stream}

    def to_map(
        self, k: typing.Callable[[T], K], v: typing.Callable[[T], V]
    ) -> dict[K, V]:
        return self.to_dict(k, v)

    def collect(self, func: typing.Callable[[typing.Iterable[T]], R]) -> R:
        return func(self._stream)

    def gather_async(
        self: "Stream[types.CoroutineType[typing.Any, typing.Any, R]]",
        concurrent_limit: int = None,
    ) -> "Stream[R]":
        """
        use like:
            async def task(x):
                await asyncio.sleep(1)
                return x
            Stream(range(10)).map(task).gather_async(limit).count()
        """

        async def gather(_stream):
            return await asyncio.gather(*_stream)

        def run(_stream):
            yield from asyncio.run(gather(_stream))

        if concurrent_limit is not None:
            semaphore = asyncio.Semaphore(concurrent_limit)

            async def semaphored_task(
                task: types.CoroutineType[typing.Any, typing.Any, R],
            ) -> R:
                async with semaphore:
                    return await task

        else:

            async def semaphored_task(
                task: types.CoroutineType[typing.Any, typing.Any, R],
            ) -> R:
                return await task

        return self.map(semaphored_task).wrap_iterator(run)

    def gather_thread_future(
        self: "Stream[concurrent.futures.Future[T]]",
    ) -> "Stream[T]":
        """
        use like:
            def costly_function():
                v = 0
                for i in range(50000000):
                    v += 1
                return v
            pool = concurrent.futures.ThreadPoolExecutor()
            r = (
                Stream(range(10))
                .map(lambda x: pool.submit(costly_function))
                .gather_thread_future()
                .collect(list)
            )
        """
        return self.wrap_iterator(concurrent.futures.as_completed).map(
            lambda x: x.result()
        )

    def wrap_iterator(
        self, iterator: typing.Callable[[typing.Iterable[T]], typing.Iterable[R]]
    ) -> "Stream[R]":
        return self.clone(iterator(self._stream))

    def reversed(self) -> "Stream[T]":
        # not lazy, maybe costly
        return self.clone(reversed(list(self._stream)))


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
        handlers = NormalizeIterableOrSingleArgToIterable(handlers)
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


def SuccessOrNone(f: typing.Callable[[], R]) -> typing.Optional[R]:
    try:
        return f()
    except:
        return None
