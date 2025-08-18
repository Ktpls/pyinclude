from __future__ import annotations
import regex
import dataclasses
import typing
import enum
from ..modules.solid_dep_required.regex_required import FSMUtil
from ..modules.misc import NormalizeCrlf, Section
import math
import sys
import uuid


class expparser:
    class _TokenType(enum.Enum):
        LITERAL = 1
        OPR = 2
        BRA = 3
        KET = 4
        EOF = 5
        IDENTIFIER = 6
        SPACE = 7
        COMMA = 8
        COMMENT = 9

    class _FSMGraphNode(enum.Enum):
        start = 1
        got_obj = 2

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
        CALL = 18

        @staticmethod
        def throw_opr_exception(s):
            raise ValueError(f"bad opr {s}")

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
            elif self in [expparser._OprType.NEG, expparser._OprType.NOT]:
                return 5
            elif self == expparser._OprType.POW:
                return 6
            elif self in [expparser._OprType.CALL]:
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
                return arg[0] + arg[1]
            elif self == expparser._OprType.SUB:
                return arg[0] - arg[1]
            elif self == expparser._OprType.MUL:
                return arg[0] * arg[1]
            elif self == expparser._OprType.DIV:
                return arg[0] / arg[1]
            elif self == expparser._OprType.POW:
                return arg[0] ** arg[1]
            elif self == expparser._OprType.NEG:
                return -arg[0]
            elif self == expparser._OprType.NEQ:
                return arg[0] != arg[1]
            elif self == expparser._OprType.EQ:
                return arg[0] == arg[1]
            elif self == expparser._OprType.GT:
                return arg[0] > arg[1]
            elif self == expparser._OprType.GE:
                return arg[0] >= arg[1]
            elif self == expparser._OprType.LT:
                return arg[0] < arg[1]
            elif self == expparser._OprType.LE:
                return arg[0] <= arg[1]
            elif self == expparser._OprType.NOT:
                return not arg[0]
            elif self == expparser._OprType.AND:
                return arg[0] and arg[1]
            elif self == expparser._OprType.OR:
                return arg[0] or arg[1]
            elif self == expparser._OprType.XOR:
                return arg[0] ^ arg[1]
            elif self == expparser._OprType.CALL:
                return arg[0](*(arg[1]))
            else:
                expparser._OprType.throw_opr_exception(self)

    regex_num = r"(?<eff>[0-9]+(?:\.[0-9]+)?)(?:e(?<pow>[+-]?[0-9]+))?"
    regex_str = r'"(?<content>.*?)(?<=[^\\]|\\\\)"'
    matchers = [
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
            exp="^" + regex_num,
            type=_TokenType.LITERAL,
        ),
        # cant process r'"\\"' properly, but simply ignore it. cuz \ is not a thing requiring processing
        FSMUtil.RegexpTokenMatcher(exp="^" + regex_str, type=_TokenType.LITERAL),
        FSMUtil.RegexpTokenMatcher(
            exp=r"^[A-Za-z_][A-Za-z0-9_]*", type=_TokenType.IDENTIFIER
        ),
        FSMUtil.RegexpTokenMatcher(exp=r"^\(", type=_TokenType.BRA),
        FSMUtil.RegexpTokenMatcher(exp=r"^\)", type=_TokenType.KET),
        FSMUtil.RegexpTokenMatcher(exp=r"^,", type=_TokenType.COMMA),
        FSMUtil.RegexpTokenMatcher(exp=r"^$", type=_TokenType.EOF),
        FSMUtil.RegexpTokenMatcher(exp=r"^[\s\r\n\t]+", type=_TokenType.SPACE),
    ]

    class Ast:
        @dataclasses.dataclass
        class Element:
            val: typing.Any
            sec: typing.Optional[Section]

            def eval(self, env: typing.Optional[dict[str, typing.Any]] = None): ...

        @dataclasses.dataclass
        class Object(Element):
            def eval(self, env: typing.Optional[dict[str, typing.Any]] = None):
                return self.val

        @dataclasses.dataclass
        class Literal(Object):
            @staticmethod
            def of_str_literal(str_lit: str, sec: Section):
                if m := regex.match(f"^{expparser.regex_num}$", str_lit):
                    eff, pow = m.groups()
                    return expparser.Ast.Literal(
                        val=float(eff) * 10 ** (float(pow) if pow else 0), sec=sec
                    )
                elif m := regex.match(f"^{expparser.regex_str}$", str_lit):
                    return expparser.Ast.Literal(val=m.group("content"), sec=sec)
                else:
                    raise ValueError("Invalid literal")

        @dataclasses.dataclass
        class Identifier(Object):
            def eval(self, env: typing.Optional[dict[str, typing.Any]] = None):
                assert self.val in env
                return env[self.val]

        @dataclasses.dataclass
        class ArgumentTuple(Object):
            val: list[expparser.Ast.Element]

            def eval(self, env: typing.Optional[dict[str, typing.Any]] = None):
                return [o.eval(env) for o in self.val]

        @dataclasses.dataclass
        class Operator(Element):
            val: expparser._OprType
            operands: typing.Optional[list[expparser.Ast.Object]]

            def eval(self, env: typing.Optional[dict[str, typing.Any]] = None):
                return self.val.do([o.eval(env) for o in self.operands])

    @dataclasses.dataclass
    class _ReadEndState:
        result: expparser._FlatOperatorOrganizer.Ir
        ended_by: FSMUtil.Token

    class _FlatOperatorOrganizer:

        class State(enum.Enum):
            BinaryOprWait1stObj = enum.auto()
            BinaryOprWait2ndObj = enum.auto()
            BinaryOprWaitOpr = enum.auto()
            BinaryOprWaitNextOpr = enum.auto()
            CallReadArg = enum.auto()
            ReadUnaryOpr = enum.auto()
            GotTargetOfUnaryWaitingBreak = enum.auto()

        @dataclasses.dataclass
        class Ir:
            elm: expparser.Ast.Element
            organized: bool = False
            # organized, meaning regard it as an object node.
            # used on nested elm node like opr(a,b), which is actually an obj node instead of pure opr node

            @staticmethod
            def lof(
                lelm: list[expparser.Ast.Element],
            ) -> list[expparser._FlatOperatorOrganizer.Ir]:
                return [expparser._FlatOperatorOrganizer.Ir(elm=e) for e in lelm]

            @staticmethod
            def strip(
                lir: list[expparser._FlatOperatorOrganizer.Ir],
            ):
                return [e.elm for e in lir]

            def IsCurNodeOpr(self):
                return (
                    isinstance(self.elm, expparser.Ast.Operator) and not self.organized
                )

            def IsCurNodeObj(self):
                return isinstance(self.elm, expparser.Ast.Object) or self.organized

        @staticmethod
        def reorganize_operator_sort(
            obj_opr_list: list[expparser._FlatOperatorOrganizer.Ir],
            pos: int = 0,
            until_opr_level: int = -1,
        ) -> expparser._FlatOperatorOrganizer.Ir:
            state = expparser._FlatOperatorOrganizer.State.BinaryOprWait1stObj
            children_list: list[expparser.Ast.Element] = []
            beg = pos

            def cleanup_children_as_binary():
                if state == expparser._FlatOperatorOrganizer.State.BinaryOprWaitNextOpr:
                    opr1st = children_list[-2]
                    opr1st: expparser.Ast.Operator
                    opr1st.operands = [children_list[-3], children_list[-1]]
                    children_list[-3:] = [opr1st]

            def cleanup_children_as_call():
                nonlocal children_list
                for c in range(1, len(children_list)):
                    caller = children_list[c - 1]
                    arg = children_list[c]
                    assert isinstance(arg, expparser.Ast.ArgumentTuple)
                    opr_call = expparser.Ast.Operator(
                        val=expparser._OprType.CALL, sec=None, operands=[caller, arg]
                    )
                    children_list[c] = opr_call
                children_list = [children_list[-1]]

            def cleanup_children_as_unary():
                nonlocal children_list
                for i in range(len(children_list) - 1, 0, -1):
                    opr = children_list[i - 1]
                    assert isinstance(opr, expparser.Ast.Operator)
                    opr.operands = [children_list[i]]
                children_list = [children_list[0]]

            while True:
                cur_ir_node = cur_elm = None
                if pos < len(obj_opr_list):
                    cur_ir_node = obj_opr_list[pos]
                    cur_elm = cur_ir_node.elm
                match state:
                    case (
                        expparser._FlatOperatorOrganizer.State.BinaryOprWait1stObj
                        | expparser._FlatOperatorOrganizer.State.BinaryOprWait2ndObj
                    ):
                        if cur_ir_node is None:
                            raise ValueError()
                        elif cur_ir_node.IsCurNodeObj():
                            children_list.append(cur_elm)
                            if (
                                state
                                == expparser._FlatOperatorOrganizer.State.BinaryOprWait1stObj
                            ):
                                state = (
                                    expparser._FlatOperatorOrganizer.State.BinaryOprWaitOpr
                                )
                            elif (
                                state
                                == expparser._FlatOperatorOrganizer.State.BinaryOprWait2ndObj
                            ):
                                state = (
                                    expparser._FlatOperatorOrganizer.State.BinaryOprWaitNextOpr
                                )
                            pos += 1
                        elif cur_ir_node.IsCurNodeOpr():
                            # unary
                            if (
                                cur_elm.val == expparser._OprType.SUB
                            ):  # manual transfering
                                cur_elm.val = expparser._OprType.NEG
                            cur_opr_pri = cur_elm.val.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos, cur_opr_pri
                                ).elm
                                children_list.append(child)
                                pos += 1
                                if (
                                    state
                                    == expparser._FlatOperatorOrganizer.State.BinaryOprWait1stObj
                                ):
                                    state = (
                                        expparser._FlatOperatorOrganizer.State.BinaryOprWaitOpr
                                    )
                                elif (
                                    state
                                    == expparser._FlatOperatorOrganizer.State.BinaryOprWait2ndObj
                                ):
                                    state = (
                                        expparser._FlatOperatorOrganizer.State.BinaryOprWaitNextOpr
                                    )
                            elif cur_opr_pri == until_opr_level:
                                if (
                                    state
                                    == expparser._FlatOperatorOrganizer.State.BinaryOprWait1stObj
                                ):
                                    state = (
                                        expparser._FlatOperatorOrganizer.State.ReadUnaryOpr
                                    )
                                else:
                                    raise ValueError()
                            elif cur_opr_pri < until_opr_level:
                                raise ValueError("not possible")
                        else:
                            raise ValueError()
                    case (
                        expparser._FlatOperatorOrganizer.State.BinaryOprWaitOpr
                        | expparser._FlatOperatorOrganizer.State.BinaryOprWaitNextOpr
                    ):
                        if cur_elm is None:
                            cleanup_children_as_binary()
                            break
                        elif isinstance(cur_elm, expparser.Ast.ArgumentTuple):
                            # call
                            cur_opr_pri = expparser._OprType.CALL.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos - 1, cur_opr_pri
                                ).elm
                                children_list[-1] = child
                                state = state
                            elif cur_opr_pri == until_opr_level:
                                if (
                                    state
                                    == expparser._FlatOperatorOrganizer.State.BinaryOprWaitOpr
                                ):
                                    state = (
                                        expparser._FlatOperatorOrganizer.State.CallReadArg
                                    )
                                else:
                                    raise ValueError()
                            elif cur_opr_pri < until_opr_level:
                                raise ValueError("not possible")
                        elif cur_ir_node.IsCurNodeOpr():
                            cur_elm: expparser.Ast.Operator
                            cur_opr_pri = cur_elm.val.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos - 1, cur_opr_pri
                                ).elm
                                children_list[-1] = child
                                state = state
                            elif cur_opr_pri == until_opr_level:
                                cleanup_children_as_binary()
                                children_list.append(cur_elm)
                                pos += 1
                                state = (
                                    expparser._FlatOperatorOrganizer.State.BinaryOprWait2ndObj
                                )
                            elif cur_opr_pri < until_opr_level:
                                if (
                                    state
                                    == expparser._FlatOperatorOrganizer.State.BinaryOprWaitNextOpr
                                ):
                                    cleanup_children_as_binary()
                                    break
                                else:
                                    raise ValueError()
                        else:
                            raise ValueError()
                    case expparser._FlatOperatorOrganizer.State.CallReadArg:
                        if cur_elm is None:
                            cleanup_children_as_call()
                            break
                        elif isinstance(cur_elm, expparser.Ast.ArgumentTuple):
                            children_list.append(cur_elm)
                            pos += 1
                            state = state
                        elif cur_ir_node.IsCurNodeOpr():
                            cur_elm: expparser.Ast.Operator
                            cur_opr_pri = cur_elm.val.getPriority()
                            if cur_opr_pri > until_opr_level:
                                raise ValueError("not possible")
                            elif cur_opr_pri == until_opr_level:
                                children_list.append(cur_elm)
                            elif cur_opr_pri < until_opr_level:
                                cleanup_children_as_call()
                                break
                    case expparser._FlatOperatorOrganizer.State.ReadUnaryOpr:
                        if cur_elm is None:
                            raise ValueError()
                        elif cur_ir_node.IsCurNodeObj():
                            children_list.append(cur_elm)
                            state = (
                                expparser._FlatOperatorOrganizer.State.GotTargetOfUnaryWaitingBreak
                            )
                            pos += 1
                        elif cur_ir_node.IsCurNodeOpr():
                            if (
                                cur_elm.val == expparser._OprType.SUB
                            ):  # manual transfering
                                cur_elm.val = expparser._OprType.NEG
                            cur_opr_pri = cur_elm.val.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos, cur_opr_pri
                                ).elm
                                children_list.append(child)
                                state = (
                                    expparser._FlatOperatorOrganizer.State.ReadUnaryOpr
                                )
                                pos += 1
                            elif cur_opr_pri == until_opr_level:
                                children_list.append(cur_elm)
                                state = (
                                    expparser._FlatOperatorOrganizer.State.ReadUnaryOpr
                                )
                                pos += 1
                            elif cur_opr_pri < until_opr_level:
                                raise ValueError()
                        else:
                            raise ValueError()
                    case (
                        expparser._FlatOperatorOrganizer.State.GotTargetOfUnaryWaitingBreak
                    ):
                        if cur_elm is None:
                            cleanup_children_as_unary()
                            break
                        elif isinstance(cur_elm, expparser.Ast.ArgumentTuple):
                            cur_opr_pri = expparser._OprType.CALL.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos - 1, cur_opr_pri
                                ).elm
                                children_list[-1] = child
                                state = state
                            else:
                                raise ValueError("not possible")
                        elif cur_ir_node.IsCurNodeOpr():
                            cur_opr_pri = cur_elm.val.getPriority()
                            if cur_opr_pri > until_opr_level:
                                child = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list, pos - 1, cur_opr_pri
                                ).elm
                                children_list[-1] = child
                            elif cur_opr_pri < until_opr_level:
                                cleanup_children_as_unary()
                                break
                            else:
                                raise ValueError()
                        else:
                            raise ValueError()
            ret = expparser._FlatOperatorOrganizer.Ir(children_list[0], organized=True)
            obj_opr_list[beg:pos] = [ret]
            return ret

    @staticmethod
    def read_recursively(
        s: str, start_pos: int, ending: set[expparser._TokenType]
    ) -> expparser._ReadEndState:
        obj_opr_list: list[expparser._FlatOperatorOrganizer.Ir] = []
        state = expparser._FSMGraphNode.start
        ptk = FSMUtil.PeekableLazyTokenizer(s, expparser.matchers, start_pos)
        while True:
            match state:
                case expparser._FSMGraphNode.start:
                    token = ptk.next()
                    match token.type:
                        case (
                            expparser._TokenType.IDENTIFIER
                            | expparser._TokenType.LITERAL
                        ):
                            sec = Section(token.start, token.end)
                            if token.type == expparser._TokenType.IDENTIFIER:
                                obj = expparser.Ast.Identifier(
                                    val=token.value,
                                    sec=sec,
                                )
                            else:
                                obj = expparser.Ast.Literal.of_str_literal(
                                    str_lit=token.value, sec=sec
                                )
                            obj_opr_list.append(
                                expparser._FlatOperatorOrganizer.Ir(elm=obj)
                            )
                            state = expparser._FSMGraphNode.got_obj
                        case expparser._TokenType.BRA:
                            endstate = expparser.read_recursively(
                                s, token.end, ending={expparser._TokenType.KET}
                            )
                            obj_opr_list.append(endstate.result)
                            ptk.seek(endstate.ended_by.end)
                            state = expparser._FSMGraphNode.got_obj
                        case expparser._TokenType.OPR:
                            sec = Section(token.start, token.end)
                            obj_opr_list.append(
                                expparser._FlatOperatorOrganizer.Ir(
                                    elm=expparser.Ast.Operator(
                                        val=expparser._OprType.fromStr(token.value),
                                        sec=sec,
                                        operands=None,
                                    ),
                                    organized=False,
                                )
                            )
                            state = expparser._FSMGraphNode.start
                        case expparser._TokenType.COMMENT | expparser._TokenType.SPACE:
                            pass
                        case _:
                            token.Unexpected()
                case expparser._FSMGraphNode.got_obj:
                    token = ptk.next()
                    match token.type:
                        case expparser._TokenType.BRA:
                            obj: list[expparser.Ast.Element] = []
                            pos = token.end
                            while True:
                                end_state: expparser._ReadEndState = (
                                    expparser.read_recursively(
                                        s,
                                        pos,
                                        ending={
                                            expparser._TokenType.KET,
                                            expparser._TokenType.COMMA,
                                        },
                                    )
                                )
                                obj.append(end_state.result.elm)
                                if end_state.ended_by.type == expparser._TokenType.KET:
                                    break
                                pos = end_state.ended_by.end
                            obj_opr_list.append(
                                expparser._FlatOperatorOrganizer.Ir(
                                    elm=expparser.Ast.ArgumentTuple(
                                        val=obj,
                                        sec=None,
                                        # (
                                        #     Section(start=obj[0].sec.start, end=obj[-1].sec.end)
                                        #     if obj
                                        #     else Section(pos, pos)
                                        # ),
                                    )
                                )
                            )
                            ptk.seek(end_state.ended_by.end)
                            state = expparser._FSMGraphNode.got_obj
                        case expparser._TokenType.OPR:
                            obj_opr_list.append(
                                expparser._FlatOperatorOrganizer.Ir(
                                    elm=expparser.Ast.Operator(
                                        val=expparser._OprType.fromStr(token.value),
                                        sec=sec,
                                        operands=None,
                                    )
                                )
                            )
                            state = expparser._FSMGraphNode.start
                        case (
                            expparser._TokenType.EOF
                            | expparser._TokenType.KET
                            | expparser._TokenType.COMMA
                        ):
                            if token.type in ending:
                                elm = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
                                    obj_opr_list,
                                    pos=0,
                                )
                                return expparser._ReadEndState(
                                    result=elm, ended_by=token
                                )
                            else:
                                token.Unexpected()
                        case expparser._TokenType.COMMENT | expparser._TokenType.SPACE:
                            pass
                        case _:
                            token.Unexpected()

    @staticmethod
    def compile(s: str) -> expparser.Ast.Element:
        return expparser.read_recursively(
            s, 0, ending={expparser._TokenType.EOF}
        ).result.elm

    BasicConstantLib = {
        "e": math.e,
        "pi": math.pi,
        "true": True,
        "false": False,
        "none": None,
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
        "iif": lambda cond, x, y: (x if cond else y),
        "eq": lambda x, y, eps=1e-3: abs(x - y) < eps,
        "strcmp": lambda x, y: x == y,
        "cstr": str,
        "cnum": float,
        "cbool": bool,
        "list": lambda *args: list(args),
        "clip": lambda x, mini, maxi: max(mini, min(x, maxi)),
        "relerr": lambda a, b, eps=1e-10: abs(a - b) / (a + eps),
        "vecadd": lambda a, b: [x + y for x, y in zip(a, b)],
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
