
import regex
import dataclasses
import typing
import enum
from ..modules.solid_dep_required.regex_required import FSMUtil
from ..modules.misc import NormalizeCrlf
import math
import sys
import uuid

class expparser:
    EPS = 1e-10
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
                newParam = [a or d for a, d in zip(param, defaultParam)]
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
        "relerr": lambda a, b: abs(a - b) / (a + expparser. EPS),
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
