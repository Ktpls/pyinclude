from ..util_solid import *
import io
import ast
import copy
import multiprocessing
import math
import sys
import uuid

"""
solid
"""
EPS = 1e-10


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

    class DeclarationAndDependencyFinder(ast.NodeVisitor):
        @dataclasses.dataclass
        class RichDefined:
            name: str
            sec: Section = None

        def __init__(self):
            self.richDefineds: list[
                dict[
                    str,
                    DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined,
                ]
            ] = [dict()]
            self.defineds: list[set[str]] = [
                copy.copy(DistillLibraryFromDependency._builtins)
            ]  # 使用栈来管理作用域
            self.undeclUse: list[set[str]] = [set()]  # 存储未定义的变量
            self.definedInCurStackFrameFreshNow()  # buffer for definedInCurStackFrame()

        def definedInCurStackFrame(self) -> set[str]:
            return self.dicsf

        def definedInCurStackFrameFreshNow(self) -> None:
            self.dicsf = set.union(*self.defineds)

        def NewStackFrame(self) -> None:
            self.richDefineds.append(dict())
            self.defineds.append(set())
            self.undeclUse.append(set())
            self.definedInCurStackFrameFreshNow()

        def PopStackFrame(self) -> None:
            # 弹出当前作用域，并添加内层被使用但仍未定义的变量，遗留到外层0作用域，留待后置定义
            childUndefined: set[str] = self.undeclUse.pop()
            self.undeclUse[-1].update(childUndefined)
            self.defineds.pop()
            self.richDefineds.pop()
            self.definedInCurStackFrameFreshNow()

        def addDef(self, s: str, sec: Section = None) -> None:
            if s not in self.definedInCurStackFrame():
                self.richDefineds[-1][s] = (
                    DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined(
                        s, sec
                    )
                )
            self.defineds[-1].add(s)
            self.dicsf.add(s)  # maintain buffer
            # 后置定义
            # 已禁用
            # 因为后定义的正确性需要考虑代码的执行时机，但源码分析没法做到。
            # 这样禁用可能把有定义的东西当做没定义，但严格些也比报错好
            # self.undeclUse[-1].discard(s)

        def addUse(self, s: str) -> None:
            # 如果变量被使用且不在任何作用域中定义，则记录为 used
            if s not in self.definedInCurStackFrame():
                self.addUndeclUse(s)

        def addUndeclUse(self, s: str) -> None:
            self.undeclUse[-1].add(s)

        def getFunctionOrClassRealBeginLineNoIncludingDecorator(
            self, node: ast.FunctionDef | ast.ClassDef
        ) -> int:
            # ast不认为装饰器是函数或类定义的一部分，但我们一般需要这样认为
            # 返回函数或类的真实开始行号，包括装饰器
            return min([node.lineno, *[d.lineno for d in node.decorator_list]])

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # 函数名本身视为已定义
            self.addDef(
                node.name,
                Section(
                    self.getFunctionOrClassRealBeginLineNoIncludingDecorator(node),
                    node.end_lineno,
                ),
            )

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
                Section(
                    self.getFunctionOrClassRealBeginLineNoIncludingDecorator(node),
                    node.end_lineno,
                ),
            )
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
            self.generic_visit(node)

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
                    )
            self.generic_visit(node)

        def find_undefined(self) -> set[str]:
            return self.undeclUse[-1]

        def find_global_defined(self) -> set[str]:
            # 检查代码中定义的全局对象
            return self.defineds[0] - DistillLibraryFromDependency._builtins

        def find_rich_global_defined(self) -> dict[str, RichDefined]:
            gdefined = self.find_global_defined()
            return {
                k: self.richDefineds[0][k]
                for k in gdefined
                if k in self.richDefineds[0]
            }

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

        def find_undefined_variables(s: str) -> set[str]:
            tree = ast.parse(s)
            finder = DistillLibraryFromDependency.DeclarationAndDependencyFinder()
            finder.visit(tree)
            return finder.find_undefined()

        def find_globally_defined_variables(
            s: str,
        ) -> dict[
            str, DistillLibraryFromDependency.DeclarationAndDependencyFinder.RichDefined
        ]:
            tree = ast.parse(s)
            finder = DistillLibraryFromDependency.DeclarationAndDependencyFinder()
            finder.visit(tree)
            return finder.find_rich_global_defined()

        @dataclasses.dataclass
        class Definition:
            name: str
            library_index: int
            sec: Section

        @dataclasses.dataclass
        class CodeFile:
            content: str = ""
            contentSplitline: list[str] = dataclasses.field(default_factory=list)

            def __post_init__(self):
                self.contentSplitline = self.content.splitlines()

        sourceCode: list[CodeFile] = [CodeFile(s) for s in sourceCode]
        library: list[CodeFile] = [CodeFile(s) for s in library]

        libDefined: dict[str, Definition] = {}
        undef: set[str] = set()

        # 遍历library，提取其中定义的全局对象，转化为definition类，以对象名为键，保存到libDefined:dict中
        for lib_index, lib in enumerate(library):
            defined_vars = find_globally_defined_variables(lib.content)
            for name, richDef in defined_vars.items():
                # 假设每个库的代码只有一行，实际应用中可能需要更复杂的行号解析
                if richDef.sec is not None:
                    richDef.sec.start -= 1
                libDefined[name] = Definition(name, lib_index, richDef.sec)

        # 遍历sourceCode，提取其中未定义的对象名，保存到undef:list中
        for file in sourceCode:
            undef.update(find_undefined_variables(file.content))

        def DefinitionList2DistilledLibrary(defNameList: list[str]):
            defObjList: list[Definition] = [
                copy.copy(libDefined[varName]) for varName in defNameList
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
                defObjSourceCode = defObj.sec.cut(
                    library[defObj.library_index].contentSplitline
                )
                defObjSourceCode[0] = defObjSourceCode[0].strip()
                defCodeList.append("\n".join(defObjSourceCode))
            return "\n".join(defCodeList)

        distilledDef = []
        while True:
            addedDef = []
            for var in undef:
                if var in libDefined:
                    addedDef.append(var)

            if len(addedDef) == 0:
                break

            distilledDef = addedDef + distilledDef
            undef = find_undefined_variables(DefinitionList2DistilledLibrary(addedDef))

        return DefinitionList2DistilledLibrary(distilledDef)


################################################
################# not so solid #################
################################################

try:
    import regex

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
                return expparser.evaluator(
                    expparser.evaluator.EvalType.operator, opr, para
                )

            @staticmethod
            def ofLiteral(literal):
                return expparser.evaluator(
                    expparser.evaluator.EvalType.literal, literal, None
                )

            @staticmethod
            def ofFunc(func, para):
                return expparser.evaluator(
                    expparser.evaluator.EvalType.func, func, para
                )

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
                elif isinstance(nl, int):
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
                    return float(nl)
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

        regex_num = r"(?<eff>[0-9]+(?:\.[0-9]+)?)(?:e(?<pow>[+-]?[0-9]+))?"
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
            FSMUtil.RegexpTokenMatcher(
                exp=regex_num,
                type=_TokenType.NUMLIKE,
            ),
            # cant process r'"\\"' properly, but simply ignore it. cuz \ is not a thing requiring processing
            FSMUtil.RegexpTokenMatcher(exp=r'^".+?(?<!\\)"', type=_TokenType.NUMLIKE),
            FSMUtil.RegexpTokenMatcher(
                exp=r"^[A-Za-z_][A-Za-z0-9_]*", type=_TokenType.IDR
            ),
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
                        expparser._OprPriorityLeap(
                            len(tokenList) - 1, lastOprPrior, opr
                        )
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
                                if tokenList[
                                    oprRisingBeginPosList[-1].pos
                                ].value.isUnary()
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
                                if tokenList[
                                    oprRisingBeginPosList[-1].pos
                                ].value.isUnary()
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
            "log2": math.log2,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "abs": abs,
            "sign": lambda x: 1 if x > 0 else -1 if x < 0 else 0,
            "floor": math.floor,
            "ceil": math.ceil,
            "neg": lambda x: -x,
            "iif": lambda cond, x, y: (
                x if expparser._NumLikeUnionUtil.ToBool(cond) else y
            ),
            "eq": lambda x, y, ep=0.001: abs(x - y) < ep,
            "StrEq": lambda x, y: x == y,
            "CStr": str,
            "CNum": float,
            "CBool": bool,
            "CList": CList,
            "clip": lambda x, mini, maxi: max(mini, min(x, maxi)),
            "relerr": lambda a, b: abs(a - b) / (a + EPS),
        }

        BasicConstantLib = {
            "e": math.e,
            "pi": math.pi,
            "true": True,
            "false": False,
            "none": None,
        }

    class WrltDiscussLikeParser:
        indentedLineRegex = r"^(?<ind>\t*)(?<content>[^\n]*)\n?"

        @dataclasses.dataclass
        class Elm:
            level: int
            content: str = None
            children: list["WrltDiscussLikeParser.Elm"] = dataclasses.field(
                default_factory=list
            )

            @staticmethod
            def of(s: str):
                mat = regex.match(WrltDiscussLikeParser.indentedLineRegex, s)
                assert mat is not None
                return WrltDiscussLikeParser.Elm(
                    len(mat.group("ind")), mat.group("content")
                )

        class _TokenType(enum.Enum):
            indentedLine = 1
            eof = 2
            unexpected = 3

        _tokenizer = [
            FSMUtil.RegexpTokenMatcher(r"^$", _TokenType.eof),
            FSMUtil.RegexpTokenMatcher(indentedLineRegex, _TokenType.indentedLine),
            FSMUtil.RegexpTokenMatcher(r"^.", _TokenType.unexpected),
        ]

        class _Node(enum.Enum):
            start = 1

        def getAst(self, s: str):
            s = NormalizeCrlf(s)
            node = self._Node.start
            pltk = FSMUtil.PeekableLazyTokenizer(s, self._tokenizer)
            ast = WrltDiscussLikeParser.Elm(-1)
            curNodePath: list[WrltDiscussLikeParser.Elm] = [ast]
            while True:
                token = pltk.next()
                match node:
                    case self._Node.start:
                        match token.type:
                            case self._TokenType.indentedLine:
                                curNode = curNodePath[-1]
                                newElm = WrltDiscussLikeParser.Elm.of(token.value)
                                ind = newElm.level
                                if ind > curNode.level:
                                    curNode.children.append(newElm)
                                    curNodePath.append(newElm)
                                elif ind == curNode.level:
                                    curNodePath[-2].children.append(newElm)
                                    curNodePath[-1] = newElm
                                elif ind < curNode.level:
                                    while curNodePath[-1].level >= ind:
                                        curNodePath.pop()
                                    curNodePath[-1].children.append(newElm)
                                    curNodePath.append(newElm)
                                node = self._Node.start
                            case self._TokenType.eof:
                                break
                            case self._TokenType.unexpected:
                                raise Exception("unexpected token")
            return ast

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

        code = f"""\
def {lambdaName}{paraAndType}:
{body}
_setBackFun({lambdaName})"""
        caller_frame = sys._getframe(1)
        exec(
            code,
            caller_frame.f_globals,
            {**caller_frame.f_locals, "_setBackFun": _setBackFun},
        )
        return func

except ImportError:
    pass
