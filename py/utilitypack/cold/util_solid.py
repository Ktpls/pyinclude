from ..util_solid import (
    Deduplicate,
    FSMUtil,
    NormalizeCrlf,
    Section,
    StoppableSomewhat,
    StoppableThread,
    FunctionalWrapper,
    Stream,
)
import io
import ast
import copy
import multiprocessing
import math
import sys
import typing
import os
import random
import enum
import traceback
import dataclasses
import time
import json

"""
solid
"""


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


class Logger:
    def __init__(self, path):
        self.path = path
        # wont fail
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.f = open(path, "wb+")

    def log(self, content):
        self.f.write((str(content) + "\n").encode("utf8"))
        self.f.flush()

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


def LongDelay(t, interval=0.5):
    round = math.ceil(t / interval)
    for i in range(round):
        time.sleep(interval)


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


def printAndRet(val):
    print(val)
    return val


class DistillLibraryFromDependency:
    """
    勉强能用
    unables to recongize found so far
        import_child_package
            code
                a=sqlalchemy.orm.Session
            lib
                import sqlalchemy
                import sqlalchemy.orm
            where orm is a package under sqlalchemy, but not a member of sqlalchemy
            will only distill like
                import sqlalchemy
            making sqlalchemy.orm unreachable
            try to do like this
                import sqlalchemy, sqlalchemy.orm
    """

    _builtins = {
        "str",
        "int",
        "float",
        "bool",
        "list",
        "tuple",
        "set",
        "dict",
        "str",
        "print",
        "enumerate",
        "zip",
        "map",
        "filter",
        "bytearray",
        "open",
        "range",
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "sorted",
        "reversed",
        "any",
        "all",
        "isinstance",
        "type",
        "dir",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "eval",
        "staticmethod",
        "classmethod",
        "super",
        "self",
        "None",
        "True",
        "False",
        "Ellipsis",
        "Exception",
        "ord",
        "chr",
        "exit",
    }

    class DefinitionType(enum.Enum):
        default = enum.auto()
        import_ = enum.auto()
        define = enum.auto()

        def priority(self):
            return {
                DistillLibraryFromDependency.DefinitionType.import_: 0,
                DistillLibraryFromDependency.DefinitionType.default: 1,
                DistillLibraryFromDependency.DefinitionType.define: 2,
            }[self]

    class DeclarationAndDependencyFinder(ast.NodeVisitor):
        @dataclasses.dataclass
        class MinimalPortableBody:
            node: ast.stmt
            sec: Section

            @classmethod
            def ofStmt(cls, node: ast.stmt):
                return cls(
                    node=node,
                    sec=DistillLibraryFromDependency.DeclarationAndDependencyFinder.AstStmtNode2Section(
                        node
                    ),
                )

        @dataclasses.dataclass
        class RichDefined:
            name: str
            file: str = None
            type: "DistillLibraryFromDependency.DefinitionType" = dataclasses.field(
                default_factory=lambda: (
                    DistillLibraryFromDependency.DefinitionType.default
                )
            )
            sec: Section = None
            minimal_portable_body: "DistillLibraryFromDependency.DeclarationAndDependencyFinder.MinimalPortableBody" = (None)

        @dataclasses.dataclass
        class _StackFrame:
            cur_frame_newly_defined: dict[
                str,
                "DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined",
            ] = dataclasses.field(default_factory=dict)
            all: dict[
                str,
                "DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined",
            ] = dataclasses.field(default_factory=dict)
            last_frame: "DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame" = (None)
            undefined_usage: set[str] = dataclasses.field(default_factory=set)

            def define(
                self,
                d: "DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined",
            ):
                self.all[d.name] = d
                self.cur_frame_newly_defined[d.name] = d

            def _follow_up_stack(self):
                f = self
                while f:
                    yield f
                    f = self.last_frame

            def all_recalced(self):
                self.all = (
                    Stream(self._follow_up_stack())
                    .reversed()
                    .map(lambda x: x.cur_frame_newly_defined)
                    .collect(Stream.Collectors.dict_union())
                )
                return self

            @staticmethod
            def InitFromParentStack(
                parentStack: "DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame",
            ):
                return DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame(
                    all=copy.copy(parentStack.all),
                    last_frame=parentStack,
                )

        def __init__(self):
            self.stack: list[
                "DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame"
            ] = [
                DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame(
                    cur_frame_newly_defined=Stream(
                        DistillLibraryFromDependency._builtins
                    ).to_dict(
                        lambda x: x,
                        lambda x: DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined(
                            name=x, file="__builtin__"
                        ),
                    )
                ).all_recalced()
            ]  # 按作用域分栈
            self.all_existed_defines: list[
                DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined
            ] = Stream(self.rootStackFrame().cur_frame_newly_defined.values()).to_list()
            self.cur_minimal_portable_body: (
                DistillLibraryFromDependency.DeclarationOptimizedFinder.MinimalPortableBody
            ) = None

        @dataclasses.dataclass
        class IndentedStructureMngr:
            declDepFinder: "DistillLibraryFromDependency.DeclarationAndDependencyFinder"
            node: ast.AST

            def __enter__(self):

                if self.declDepFinder.cur_minimal_portable_body is None:
                    self.declDepFinder.cur_minimal_portable_body = DistillLibraryFromDependency.DeclarationAndDependencyFinder.MinimalPortableBody.ofStmt(
                        self.node
                    )
                else:
                    # reentering is fine, cuz its so called "minimal"
                    pass

            def __exit__(self, *a, **kw):
                assert self.declDepFinder.cur_minimal_portable_body is not None
                if self.declDepFinder.cur_minimal_portable_body.node is self.node:
                    self.declDepFinder.cur_minimal_portable_body = None

        @dataclasses.dataclass
        class StackFrameMngr:
            declDepFinder: "DistillLibraryFromDependency.DeclarationAndDependencyFinder"

            def __enter__(self):
                self.declDepFinder.NewStackFrame()

            def __exit__(self, *a, **kw):
                self.declDepFinder.PopStackFrame()

        def lastStackFrame(
            self,
        ) -> "DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame":
            return self.stack[-1]

        def rootStackFrame(
            self,
        ) -> "DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame":
            return self.stack[0]

        def definedInCurStackFrame(self) -> set[str]:
            return set(self.lastStackFrame().all.keys())

        def NewStackFrame(self) -> None:
            self.stack.append(
                DistillLibraryFromDependency.DeclarationAndDependencyFinder._StackFrame.InitFromParentStack(
                    self.lastStackFrame()
                )
            )

        def PopStackFrame(self) -> None:
            # 弹出当前作用域，并添加内层被使用但仍未定义的变量，遗留到外层0作用域，留待后置定义
            childUndefined: set[str] = self.lastStackFrame().undefined_usage
            self.stack.pop()
            self.lastStackFrame().undefined_usage.update(childUndefined)

        def addDef(
            self,
            s: str,
            sec: Section = None,
            type: "DistillLibraryFromDependency.DefinitionType" = None,
        ) -> None:
            type = type or DistillLibraryFromDependency.DefinitionType.default
            if s not in self.definedInCurStackFrame():
                self.lastStackFrame().define(
                    DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined(
                        name=s,
                        type=type,
                        sec=sec,
                        minimal_portable_body=self.cur_minimal_portable_body,
                    )
                )
                # 后置定义
                # 已禁用
                # 因为后定义的正确性需要考虑代码的执行时机，但源码分析没法做到。
                # 这样禁用可能把有定义的东西当做没定义，但严格些也比报错好
                # self.lastStackFrame().undefined_usage.discard(s)

        def addUse(self, s: str) -> None:
            # 如果变量被使用且不在任何作用域中定义，则记录为 used
            if s not in self.definedInCurStackFrame():
                self.addUndeclUse(s)

        def addUndeclUse(self, s: str) -> None:
            self.lastStackFrame().undefined_usage.add(s)

        @classmethod
        def getFunctionOrClassRealBeginLineNoIncludingDecorator(
            self,
            node: ast.AST | ast.FunctionDef | ast.ClassDef,
        ) -> int:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # ast不认为装饰器是函数或类定义的一部分，但我们一般需要这样认为
                # 返回函数或类的真实开始行号，包括装饰器
                return min([node.lineno, *[d.lineno for d in node.decorator_list]])
            else:
                return node.lineno

        @classmethod
        def AstStmtNode2Section(self, node: ast.stmt) -> Section:
            return Section(
                self.getFunctionOrClassRealBeginLineNoIncludingDecorator(node),
                node.end_lineno,
            )

        def simple_indent_block_visit(self, node: ast.stmt):
            with self.IndentedStructureMngr(self, node):
                self.generic_visit(node)

        def visit_If(self, node: ast.If):
            self.simple_indent_block_visit(node)

        def visit_Try(self, node: ast.TryStar):
            self.simple_indent_block_visit(node)

        def visit_TryStar(self, node: ast.TryStar):
            self.simple_indent_block_visit(node)

        def visit_With(self, node: ast.With):
            self.simple_indent_block_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # 函数名本身视为已定义
            self.addDef(
                node.name,
                self.AstStmtNode2Section(node),
            )
            with self.IndentedStructureMngr(self, node):
                self.NewStackFrame()
                # 将函数的参数添加到当前作用域
                for arg in node.args.args:  # 获取函数的形参列表
                    self.addDef(arg.arg)
                # 访问函数体
                self.generic_visit(node)
                self.PopStackFrame()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.addDef(
                node.name,
                self.AstStmtNode2Section(node),
            )
            with self.IndentedStructureMngr(self, node):
                self.NewStackFrame()
                self.generic_visit(node)
                self.PopStackFrame()

        def visit_Assign(self, node: ast.Assign) -> None:
            # 处理赋值语句，将变量名添加到当前作用域
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.addDef(target.id, Section(node.lineno, node.end_lineno))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            # 处理变量使用
            if isinstance(node.ctx, ast.Load):
                self.addUse(node.id)
            elif isinstance(node.ctx, ast.Store):
                # 如果变量被定义，则添加到当前作用域
                self.addDef(node.id)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            # 进入 for 循环时，将循环变量添加到当前作用域
            if isinstance(node.target, ast.Name):  # 单个变量的情况
                self.addDef(node.target.id)
            elif isinstance(node.target, (ast.Tuple, ast.List)):  # 多个变量的情况
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self.addDef(elt.id)
            self.simple_indent_block_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            # 进入推导式循环时，将循环变量添加到当前作用域
            if isinstance(node.target, ast.Name):  # 单个变量的情况
                self.addDef(node.target.id)
            elif isinstance(node.target, (ast.Tuple, ast.List)):  # 多个变量的情况
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self.addDef(elt.id)
            self.generic_visit(node)

        def visit_ListComp(self, node: ast.ListComp) -> None:
            # 列表推导式：创建新的作用域
            self.NewStackFrame()

            # 提前访问推导式的循环部分，下同
            for gen in node.generators:
                self.visit_comprehension(gen)

            # 访问推导式的表达式部分
            self.generic_visit(node)

            # 弹出推导式的作用域
            self.PopStackFrame()

        def visit_DictComp(self, node: ast.DictComp) -> None:
            # 字典推导式：创建新的作用域
            self.NewStackFrame()

            # 访问推导式的循环部分
            for gen in node.generators:
                self.visit_comprehension(gen)

            # 访问推导式的键值表达式部分
            self.generic_visit(node)

            # 弹出推导式的作用域
            self.PopStackFrame()

        def visit_SetComp(self, node: ast.SetComp) -> None:
            # 集合推导式：创建新的作用域
            self.NewStackFrame()

            # 访问推导式的循环部分
            for gen in node.generators:
                self.visit_comprehension(gen)

            # 访问推导式的表达式部分
            self.generic_visit(node)

            # 弹出推导式的作用域
            self.PopStackFrame()

        def visit_Import(self, node: ast.Import) -> None:
            # 处理 import 语句，将模块名添加到当前作用域
            for alias in node.names:
                self.addDef(
                    alias.asname or alias.name.split(".")[0],
                    Section(node.lineno, node.end_lineno),
                    DistillLibraryFromDependency.DefinitionType.import_,
                )
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            # 处理 from ... import ...，将模块名添加到当前作用域
            for alias in node.names:
                name: str = alias.asname or alias.name
                if name != "*":
                    self.addDef(
                        alias.asname or alias.name,
                        Section(node.lineno, node.end_lineno),
                        DistillLibraryFromDependency.DefinitionType.import_,
                    )
            self.generic_visit(node)

        def find_undefined(self) -> set[str]:
            return self.lastStackFrame().undefined_usage

        def find_global_defined(self) -> set[str]:
            # 检查代码中定义的全局对象
            return (
                set(self.rootStackFrame().cur_frame_newly_defined.keys())
                - DistillLibraryFromDependency._builtins
            )

        def find_rich_global_defined(self) -> dict[str, RichDefined]:
            return (
                Stream(self.find_global_defined())
                .map(
                    lambda k: self.rootStackFrame().cur_frame_newly_defined.get(k, None)
                )
                .filter(lambda x: x is not None)
                .to_map(lambda x: x.name, lambda x: x)
            )

        def proc_text(self, text: str):
            tree = ast.parse(text)
            self.visit(tree)
            return self

    class DeclarationOptimizedFinder(DeclarationAndDependencyFinder):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.addDef(node.name, Section(node.lineno, node.end_lineno))
            # 不访问函数体，因为不关心局部变量。下同
            # self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.addDef(node.name, Section(node.lineno, node.end_lineno))

    @staticmethod
    def check_decls_and_undefs(content: str):
        buf = io.StringIO()
        oldstdout, sys.stdout = sys.stdout, buf

        finder = DistillLibraryFromDependency.DeclarationAndDependencyFinder()
        finder.proc_text(content)
        print(
            "Undefined variables:", json.dumps(list(finder.find_undefined()), indent=4)
        )
        print(
            "Globally defined variables:",
            json.dumps(list(finder.find_global_defined()), indent=4),
        )
        print("Globally defined variable details:")
        linedContent = content.splitlines()
        for k, v in finder.find_rich_global_defined().items():
            print(f"{k}:")
            if v.sec is not None:
                # cuz ast returns line number from 1, but we need it from 0
                v.sec.start -= 1
                # -=1, and +=1 cuz end line included
                # sec.end -= 1
                for l in v.sec.cut(linedContent):
                    print(f"\t{l}")
        sys.stdout = oldstdout
        return buf.getvalue()

    @staticmethod
    def find_undefined_variables(s: str) -> set[str]:
        # use print(ast.dump(tree, indent=2)) to see the structure
        tree = ast.parse(s)
        finder = DistillLibraryFromDependency.DeclarationAndDependencyFinder()
        finder.visit(tree)
        return finder.find_undefined()

    @staticmethod
    def find_globally_defined_variables(
        s: str,
    ) -> dict[
        str, "DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined"
    ]:
        tree = ast.parse(s)
        finder = DistillLibraryFromDependency.DeclarationAndDependencyFinder()
        finder.visit(tree)
        return finder.find_rich_global_defined()

    @staticmethod
    def DistillLibrary(sourceCode: list[str], library: list[str]):
        """
        TODO
            from utilitypack.util_xxx import *能够正常工作
                而如果from utilitypack.util_xxx import xxx，那么不会把xxx视为需要定义
                因为假设库都存在
                但正是要提取这个库的内容
                在被分析依赖的代码中，一般会去除utilitypack的导入，所以没问题
                但如果是utilitypack内，使用了from yyy import xxx的写法，则会跳过其导入
            优化定义合并
                将import移动至最前方
        """

        @dataclasses.dataclass
        class Definition:
            name: str
            library_index: int
            sec: Section
            type: DistillLibraryFromDependency.DefinitionType

        @dataclasses.dataclass
        class CodeFile:
            content: str = ""
            contentSplitline: list[str] = dataclasses.field(default_factory=list)

            def __post_init__(self):
                self.contentSplitline = self.content.splitlines()

        sourceCode: list[CodeFile] = [CodeFile(s) for s in sourceCode]
        library: list[CodeFile] = [CodeFile(s) for s in library]

        def DefinitionList2DistilledLibrary(
            namesToDefine: list[str],
            usableLibDefined: dict[str, "Definition"],
        ):
            defObjList: list[Definition] = [
                copy.copy(usableLibDefined[varName]) for varName in namesToDefine
            ]

            def overlaps(self: Definition, other: Definition) -> bool:
                return self.library_index == other.library_index and not (
                    self.sec.end <= other.sec.start or self.sec.start >= other.sec.end
                )

            def union(self: Definition, other: Definition):
                self.name += f"|{defObj.name}"
                self.sec = Section(
                    start=min(self.sec.start, other.sec.start),
                    end=max(self.sec.end, other.sec.end),
                )

            merged_definitions: list[Definition] = []
            for defObj in defObjList:
                merged = False
                for existing_defObj in merged_definitions:
                    if overlaps(defObj, existing_defObj):
                        union(existing_defObj, defObj)
                        merged = True
                        break
                if not merged:
                    merged_definitions.append(defObj)
            # 借助库内代码的顺序合理性来辅助保障提取代码的声明顺序的合理性
            sorted(merged_definitions, key=lambda x: (x.library_index, x.sec.start))

            defCodeList = []
            for defObj in merged_definitions:
                sec = defObj.sec
                sec = copy.copy(sec)
                if sec is not None:
                    sec.start -= 1
                defObjSourceCode = sec.cut(
                    library[defObj.library_index].contentSplitline
                )
                # defObjSourceCode[0] = defObjSourceCode[0].strip()
                defCodeList.append("\n".join(defObjSourceCode))
            return "\n".join(defCodeList)

        # 可能找到多处定义
        libDefined: dict[str, Definition] = {}
        undef: list[str] = list()

        # 遍历library，提取其中定义的全局对象，转化为definition类，以对象名为键，保存到libDefined:dict中
        for lib_index, lib in enumerate(library):
            defined_vars = DistillLibraryFromDependency.find_globally_defined_variables(
                lib.content
            )
            for name, richDef in defined_vars.items():
                def_ = Definition(
                    name,
                    lib_index,
                    (
                        richDef.minimal_portable_body.sec
                        if richDef.minimal_portable_body
                        else richDef.sec
                    ),
                    richDef.type,
                )
                if name not in libDefined:
                    libDefined[name] = def_
                else:
                    old = libDefined[name]
                    if old.type.priority() < def_.type.priority():
                        libDefined[name] = def_

        # 遍历sourceCode，提取其中未定义的对象名，保存到undef:list中
        undef = (
            Stream(sourceCode)
            .map(lambda file: file.content)
            .map(DistillLibraryFromDependency.find_undefined_variables)
            .flat_map(lambda x: Stream(x))
            .collect(Stream.Collectors.list)
        )
        defsRequiringAdding = []
        distilled_lib = ""
        while True:
            addedDef = (
                Stream(undef)
                .sorted()  # 去除代码分析器内部使用set导致的结果顺序不稳定性
                .filter(lambda x: x in libDefined)
                .collect(Stream.Collectors.list)
            )

            if len(addedDef) == 0:
                break

            defsRequiringAdding = (
                Stream(addedDef + defsRequiringAdding).distinct().collect(list)
            )
            distilled_lib = DefinitionList2DistilledLibrary(
                defsRequiringAdding, libDefined
            )
            undef = list(
                DistillLibraryFromDependency.find_undefined_variables(distilled_lib)
            )

        return distilled_lib


################################################
################# not so solid #################
################################################

try:
    from .regex_required import *

except ImportError:
    pass
