from __future__ import annotations
from testcase.autotest_common import *
from utilitypack.cold.util_solid import *
from utilitypack.util_solid import *
from utilitypack.util_np import *
from utilitypack.util_cracked import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *


class UnionableClassTest(unittest.TestCase):
    def test_basic_usage(self):
        @dataclasses.dataclass
        class Setting(UnionableClass):
            v1: int = None
            v2: int = None

            def __all_field_names__(self):
                return [k.name for k in dataclasses.fields(self)]

        self.assertDictEqual(
            dataclasses.asdict(Setting(v1=1) | Setting(v2=2)), {"v1": 1, "v2": 2}
        )


class ExpNodeReorganizationTest(unittest.TestCase):
    def assert_reorg_opr_result(
        self, l: list[expparser.Ast.Element], expected: expparser.Ast.Element
    ):
        r = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
            expparser._FlatOperatorOrganizer.Ir.lof(l)
        ).elm
        self.assertDictEqual(dataclasses.asdict(r), dataclasses.asdict(expected))

    def print_reorg_opr_result(self, l: list[expparser.Ast.Element]):
        r = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
            expparser._FlatOperatorOrganizer.Ir.lof(l)
        ).elm
        pprint.pp(r)

    def test_1p2(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.Object(val=2, sec=None),
                ],
            ),
        )

    def test_1p2p3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.ADD,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    ),
                    expparser.Ast.Object(val=3, sec=None),
                ],
            ),
        )

    def test_1p2m3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.MUL, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.Operator(
                        val=expparser._OprType.MUL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=2, sec=None),
                            expparser.Ast.Object(val=3, sec=None),
                        ],
                    ),
                ],
            ),
        )

    def test_1m2p3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.MUL, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.MUL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    ),
                    expparser.Ast.Object(val=3, sec=None),
                ],
            ),
        )

    def test_n1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[expparser.Ast.Object(val=1, sec=None)],
            ),
        )

    def test_nn1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    )
                ],
            ),
        )

    def test_n1pow2(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.POW, None, None),
                expparser.Ast.Object(2, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.POW,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    )
                ],
            ),
        )

    def test_n1pn1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    ),
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    ),
                ],
            ),
        )

    def test_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.CALL,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                ],
            ),
        )

    def test_neg_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    )
                ],
            ),
        )

    def test_callp1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    ),
                    expparser.Ast.Object(1, None),
                ],
            ),
        )

    def test_call_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.CALL,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    ),
                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                ],
            ),
        )

    def test_complex(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[
                            expparser.Ast.Operator(
                                val=expparser._OprType.CALL,
                                sec=None,
                                operands=[
                                    expparser.Ast.Object(val=1, sec=None),
                                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                                ],
                            )
                        ],
                    ),
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[
                            expparser.Ast.Operator(
                                val=expparser._OprType.CALL,
                                sec=None,
                                operands=[
                                    expparser.Ast.Object(val=1, sec=None),
                                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        )


class ExpParseTest(unittest.TestCase):
    def expparseWithEnv(self, exp):
        return expparser.compile(exp).eval(expparser.BasicConstantLib)

    def test_neg(self):
        self.assertAlmostEqual(self.expparseWithEnv(r"-1"), -1)

    def test_negneg(self):
        self.assertAlmostEqual(self.expparseWithEnv(r"--1"), 1)

    def test_cstr(self):
        self.assertEqual(self.expparseWithEnv("str(1)"), "1.0")

    def test_array(self):
        self.assertEqual(self.expparseWithEnv(r"list(1,2,3)"), [1.0, 2.0, 3.0])

    def test_div(self):
        self.assertAlmostEqual(self.expparseWithEnv(r"2/2/2"), 0.5)

    def test_oper_priority(self):
        self.assertAlmostEqual(self.expparseWithEnv(r"2^2*2+--1"), 9.0)

    def test_eq_func(self):
        self.assertEqual(self.expparseWithEnv(r"eq(1+0.1,1)"), False)

    def test_eq_func_eps(self):
        self.assertEqual(self.expparseWithEnv(r"eq(1+0.1,1,0.2)"), True)

    ## complex string escape currently not supported
    ## regexp is not capabale of this, consider add a new string reader
    # def test_str(self):
    #     self.assertEqual(self.expparseWithEnv(r'"test \" str"'), 'test " str')

    def test_eq_opr(self):
        self.assertEqual(self.expparseWithEnv(r"1!=2"), True)

    def test_comp_opr(self):
        self.assertEqual(self.expparseWithEnv(r"2>=3"), False)

    def test_clist(self):
        self.assertListEqual(self.expparseWithEnv(r"list(1)"), [1.0])

    def test_cbool(self):
        self.assertEqual(self.expparseWithEnv(r"bool(1)"), True)

    def test_cbool2(self):
        self.assertEqual(self.expparseWithEnv(r"bool(0)"), False)

    def test_strcmp(self):
        self.assertEqual(self.expparseWithEnv(r'strcmp(str(1),"1.0")'), True)

    def test_cnum_from_str(self):
        self.assertAlmostEqual(self.expparseWithEnv(r'num("1.23")'), 1.23)

    def test_cnum_from_bool(self):
        self.assertEqual(self.expparseWithEnv(r"num(true)"), 1.0)

    def test_unmatched_ket(self):
        with self.assertRaises(Exception):
            self.expparseWithEnv(r"bool(0))))))))")

    def test_space(self):
        self.assertListEqual(
            self.expparseWithEnv("list(1 ,\t2,\r\n3)"), [1.0, 2.0, 3.0]
        )

    def test_vecadd(self):
        self.assertListEqual(
            self.expparseWithEnv(r"vecadd(list(1,2),list(3,4))"), [4.0, 6.0]
        )

    def test_complex_array(self):
        self.assertListEqual(
            self.expparseWithEnv(r"list(list(1,1),list(2,2),list(1),1)"),
            [[1.0, 1.0], [2.0, 2.0], [1.0], 1.0],
        )

    def test_comment(self):
        self.assertEqual(
            self.expparseWithEnv(
                r"""
                //comment one
                1
                /*
                comment
                two
                */
                +/*inline comment*/1
                """
            ),
            2.0,
        )

    def test_scientific_number(self):
        self.assertAlmostEqual(
            self.expparseWithEnv("1.1e-1"),
            0.11,
        )


class UrlFullResolutionLazyTest(unittest.TestCase):
    class example:
        url = r"https://picx.zhimg.com:8080/the_folder/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c&needBackground=1"
        baseHost = "zhimg.com"
        domain = "com"
        extName = "jpg"
        fileBaseName = "v2-abed1a8c04700ba7d72b45195223e0ff_l"
        fileName = "v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg"
        folder = "/the_folder/"
        host = "picx.zhimg.com:8080"
        param = "source=32738c0c&needBackground=1"
        path = "/the_folder/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg"
        port = "8080"
        protocol = "https"
        secondaryHost = "picx"

    def test_fields(self):
        res = UrlFullResolution.of(self.example.url)
        res.calcAll()
        fields = [
            "baseHost",
            "domain",
            "extName",
            "fileBaseName",
            "fileName",
            "folder",
            "host",
            "param",
            "path",
            "port",
            "protocol",
            "secondaryHost",
        ]
        self.assertDictEqual(
            {k: getattr(res, k) for k in fields},
            {k: getattr(self.example, k) for k in fields},
        )

    def test_lazy_resolving(self):
        res = UrlFullResolution.of(self.example.url)
        fields = [
            "protocol",
            "port",
            "folder",
        ]
        self.assertDictEqual(
            {k: getattr(res, k) for k in fields},
            {k: getattr(self.example, k) for k in fields},
        )

    def test_corner_dot_in_path(self):
        res = UrlFullResolution.of(
            r"C:\file\Gs\Storage\mc\.minecraft\versions\1.21.1-NeoForge_21.1.168\saves\Dragon Island"
        )
        self.assertEqual(res.fileName, r"Dragon Island")


class MaxRetryTest(unittest.TestCase):
    def test_succ(self):
        result = 0
        retry = MaxRetry(succCond=lambda: result >= 2, maxRetry=3)
        for i in retry:
            result = i
        self.assertEqual(result, 2)
        self.assertEqual(retry.isSuccessed, True)

    def test_fail(self):
        result = 0
        retry = MaxRetry(succCond=lambda: result >= 5, maxRetry=3, errOnMaxRetry=False)
        for i in retry:
            result = i
        self.assertEqual(result, 3)
        self.assertEqual(retry.isSuccessed, False)


class HotkeyManagerTest(unittest.TestCase):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        from utilitypack.util_app import HotkeyManager

        self.CAC = Switch()
        self.CC = Switch()
        self.hkm = HotkeyManager(
            [
                HotkeyManager.hotkeytask(
                    [
                        win32conComp.VK_CONTROL,
                        win32conComp.VK_MENU,
                        win32conComp.KeyOf("C"),
                    ],
                    lambda: self.CAC.on(),
                    lambda: self.CAC.off(),
                ),
                HotkeyManager.hotkeytask(
                    [win32conComp.VK_CONTROL, win32conComp.KeyOf("C")],
                    lambda: self.CC.on(),
                    lambda: self.CC.off(),
                ),
            ]
        )
        self.keyClear()
        self.hkm._getKeyConcernedState = lambda: self.KeyState

    def keyClear(self):
        self.KeyState = {
            win32conComp.VK_CONTROL: False,
            win32conComp.VK_MENU: False,
            win32conComp.KeyOf("C"): False,
        }

    def test_key_priority(self):
        self.keyClear()
        self.KeyState.update(
            {win32conComp.VK_CONTROL: True, win32conComp.VK_MENU: True}
        )
        self.hkm.dispatchMessage()

        self.assertEqual(self.CAC(), False)
        self.assertEqual(self.CC(), False)
        self.KeyState[win32conComp.KeyOf("C")] = True
        self.hkm.dispatchMessage()
        self.assertEqual(self.CAC(), True)
        self.assertEqual(self.CC(), False)

        self.KeyState.update({win32conComp.KeyOf("C"): False})
        self.hkm.dispatchMessage()
        self.assertEqual(self.CAC(), False)
        self.assertEqual(self.CC(), False)

        self.keyClear()
        self.KeyState.update(
            {win32conComp.VK_CONTROL: True, win32conComp.KeyOf("C"): True}
        )
        self.hkm.dispatchMessage()
        self.assertEqual(self.CAC(), False)
        self.assertEqual(self.CC(), True)


class EasyWrapperTest(unittest.TestCase, RedirectedPrint):
    def __init__(self, *arg, **kwarg):
        unittest.TestCase.__init__(self, *arg, **kwarg)
        RedirectedPrint.__init__(self)

    def test_Parensis(self):
        self.clearPrinted()

        @EasyWrapper
        def wParensis(f, p, op=1):
            def newF(*args, **kwargs):
                self.print(f"wParensis {p=} {op=}")
                return f(*args, **kwargs)

            return newF

        @RunThis
        @wParensis(p=1)
        def f():
            self.print("f")

        self.assertListEqual(self.getPrinted(), ["wParensis p=1 op=1", "f"])

    def test_Parensisless(self):
        @EasyWrapper
        def wParensisless(f):
            def newF(*args, **kwargs):
                self.print(f"wParensisless")
                return f(*args, **kwargs)

            return newF

        @RunThis
        @wParensisless
        def f():
            self.print("f")

        self.assertListEqual(self.getPrinted(), ["wParensisless", "f"])

    def test_WithTwoFunctionArg(self):
        @EasyWrapper
        def wWithTwoFunctionArg(f, g):
            def newF(*args, **kwargs):
                self.print(f"wWithTwoFunctionArg {(g!=None)=}")
                g()
                return f(*args, **kwargs)

            return newF

        def g():
            self.print("g")

        @RunThis
        @wWithTwoFunctionArg(g)
        def f():
            self.print("f")

        self.assertListEqual(
            self.getPrinted(), ["wWithTwoFunctionArg (g!=None)=True", "g", "f"]
        )

    def test_MethodWithinClassAsDecorator(self):
        class ClassWithDecoratorMethod:
            @EasyWrapper
            def decoratorMethod(f, self2):
                def newF(*args, **kwargs):
                    self.print(f"ClassWithDecoratorMethod.decoratorMethod")
                    return f(*args, **kwargs)

                return newF

        cwdmInstance = ClassWithDecoratorMethod()

        @RunThis
        @cwdmInstance.decoratorMethod()
        def f():
            self.print("f")

        self.assertListEqual(
            self.getPrinted(), ["ClassWithDecoratorMethod.decoratorMethod", "f"]
        )


class TimerTest(unittest.TestCase):
    # perf_statistic untested
    def __init__(self, *arg, **kwarg):
        unittest.TestCase.__init__(self, *arg, **kwarg)
        self.t = 0

    def initSingleSectionedTimer(self):
        class StubbedTimer(SingleSectionedTimer):
            def timeCounter(self2):
                return self.t

        return StubbedTimer

    def initPerfStatistic(self):

        class StubbedPs(perf_statistic):
            def __init__(self2, startnow=False):
                super().__init__(startnow=False)
                self2._singled = self.initSingleSectionedTimer()
                self2.clear()
                if startnow:
                    self2.start()

        return StubbedPs

    def test_ssdStartGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)

    def test_ssdStartGetClearGetGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)
        self.t = 2
        self.assertEqual(ssd.clear().get(), 0)
        ssd.start()
        self.t = 3
        self.assertEqual(ssd.get(), 1)

    def test_ssdStartGetStartGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)
        self.t = 2
        ssd.start()
        self.t = 3
        self.assertEqual(ssd.get(), 1)


class AutoFunctionalTest(unittest.TestCase, RedirectedPrint):
    def test_AutoFunctional(self):
        varStatic = 0

        @AutoFunctional
        class Clz:
            def __init__(self2, val):
                self2.val = val

            def inc(self2) -> typing.Self:
                self2.val += 1

            def assertVal(self2, val) -> typing.Self:
                self.assertEqual(self2.val, val)

            @staticmethod
            def staticMethod() -> typing.Self:
                nonlocal varStatic
                varStatic += 1

        a = Clz(1)
        a.inc().assertVal(2).inc().assertVal(3).staticMethod()
        self.assertEqual(a.val, 3)
        self.assertEqual(varStatic, 1)


# class BeanUtilTest(unittest.TestCase):
# # no furthuer update plan
#     def test_CopySimpleClass(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             i: int
#             s: str

#         a = A(1, "s")
#         b: B = BeanUtil.copyProperties(a, B)
#         self.assertEqual(b.i, 1)

#     def test_CopyNestedClass(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             a: A
#             b: str

#         inst = B(
#             A(1, "s"),
#             "s",
#         )
#         m = BeanUtil.toJsonCompatible(inst)
#         self.assertDictEqual(
#             m,
#             {
#                 "a": {
#                     "i": 1,
#                     "s": "s",
#                 },
#                 "b": "s",
#             },
#         )
#         inst: B = BeanUtil.copyProperties(m, B)
#         self.assertEqual(inst.a.i, 1)

#     def test_CopyList(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             la: list[A]
#             b: str

#         b = B(
#             [A(0, "s"), A(1, "s")],
#             "s",
#         )
#         m = BeanUtil.toJsonCompatible(b)
#         self.assertDictEqual(
#             m,
#             {
#                 "la": [
#                     {"i": 0, "s": "s"},
#                     {"i": 1, "s": "s"},
#                 ],
#                 "b": "s",
#             },
#         )
#         b2: B = BeanUtil.copyProperties(m, B)
#         self.assertEqual(b2.la[1].i, 1)


class DataclassInitReorderTest(unittest.TestCase):
    def test_inherit(self):
        @dataclasses.dataclass
        class Parent:
            x: str = ""  # 默认值
            y: typing.Any = dataclasses.field(
                default_factory=lambda: None
            )  # 使用 default_factory 作为默认值

        # 子类
        @dataclasses.dataclass
        class Child(Parent):
            z: str  # 没有默认值

        child = Child("z", "x", "y")
        self.assertDictEqual(
            BeanUtil.toJsonCompatible(child),
            {"x": "x", "y": "y", "z": "z"},
        )


class MlambdaTest(unittest.TestCase):
    def test_newLined(self):
        f = mlambda(
            """
            def (a, b):
                c=a+b
                return c
            """
        )
        ret = f(1, 1)
        self.assertEqual(ret, 2)

    def test_oneLined(self):
        f = mlambda("""def (a, b): return a+b""")
        ret = f(1, 1)
        self.assertEqual(ret, 2)


class SyncExecutableTest(unittest.TestCase):
    @dataclasses.dataclass
    class _testbed(Stage):
        # _testbed, for make it clear to unittest lib that this is not a test case(not startswith('test'))
        eosm: SyncExecutableManager
        t: float = 0

        def step(self, dt):
            self.t += dt
            self.eosm.step()

    def test_basicUsage(selfTest):

        class ScriptTest(SyncExecutable):
            def main(self):
                def sleep_specified_time(t):
                    t0 = self.sem.stage.t
                    self.sleep(t)
                    selfTest.assertTrue(self.sem.stage.t - t0 >= t)

                sleep_specified_time(3)
                sleep_specified_time(5)
                sleep_specified_time(10)

        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        script = ScriptTest(eosm).run()
        while True:
            stage.step(1)
            if script.state == SyncExecutable.STATE.stopped:
                break

    class ConsumerProducer:

        @dataclasses.dataclass
        class Production:
            value: int

            def consume(self):
                self.value -= 1

            def produce(self):
                self.value += 1

            def isEmpty(self):
                return self.value == 0

            def isConsumable(self):
                return not self.isEmpty()

        class MainScript(SyncExecutable):
            # consumes production by 1, limited times
            def main(
                self,
                production: "SyncExecutableTest.ConsumerProducer.Production",
                times: int,
            ):
                for i in range(times):
                    self.sleepUntil(lambda: production.isConsumable())
                    production.consume()

        class ProducerScript(SyncExecutable):
            # produce production by 1 if main script is alive
            def main(
                self,
                production: "SyncExecutableTest.ConsumerProducer.Production",
                mainScript: "SyncExecutableTest.ConsumerProducer.MainScript",
            ):
                while True:
                    self.sleepUntil(
                        lambda: production.isEmpty()
                        or mainScript.state == SyncExecutable.STATE.stopped
                    )
                    if mainScript.state == SyncExecutable.STATE.stopped:
                        break
                    production.produce()

    def test_multiScriptFlow(selfTest):
        production = SyncExecutableTest.ConsumerProducer.Production(0)
        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        ms = SyncExecutableTest.ConsumerProducer.MainScript(eosm).run(production, 5)
        ps = SyncExecutableTest.ConsumerProducer.ProducerScript(eosm).run(
            production, ms
        )
        while True:
            stage.step(1)
            if (
                ms.state == SyncExecutable.STATE.stopped
                and ps.state == SyncExecutable.STATE.stopped
            ):
                break

    def test_LaunchThreadInThread(selfTest):
        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(eosm=None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        records = list()

        class MainScriptLauchingThreadFromInside(SyncExecutable):
            def main(self, iteration: int):
                recorderStopSignal = False
                value = 0

                class RecorderScript(SyncExecutable):

                    def main(selfr):

                        def record():
                            if len(records) == 0 or records[-1] != value:
                                records.append(value)

                        while True:
                            if recorderStopSignal:
                                break
                            record()
                            selfr.stepOneFrame()

                recorder = RecorderScript(eosm).run()
                for i in range(iteration):
                    value = i
                    self.sleep(1)
                recorderStopSignal = True
                self.stepOneFrame()

        ms = MainScriptLauchingThreadFromInside(eosm).run(5)
        while True:
            stage.step(0.1)
            if ms.state == SyncExecutable.STATE.stopped:
                break
        pass


class LazyFieldEvaluationTest(unittest.TestCase):
    def test_basicUsage(self):
        inst = None
        evaluation_result = 1

        class EvaluationLazy(LazyLoading):
            def _init_value(selfel):
                self.assertIsNotNone(inst)
                return evaluation_result

            value = LazyLoading.LazyField(_init_value)

        inst = EvaluationLazy()
        value_got = inst.value
        self.assertEqual(value_got, evaluation_result)


class PositionalArgsResolvedAsNamedKwargsTest(unittest.TestCase):
    def test_basicUsage(self):
        @EasyWrapper
        def auto_inc(func, arg_names):
            reso = PositionalArgsResolvedAsNamedKwargs(func)

            @functools.wraps(func)
            def new_func(*args, **kwargs):
                args, kwargs = reso.apply(
                    lambda k, v: v + 1 if k in arg_names else v, args, kwargs
                )
                return func(*args, **kwargs)

            return new_func

        @auto_inc(["a", "b"])
        def test(a, b, c, d):
            return {"a": a, "b": b, "c": c, "d": d}

        expected = {"a": 1, "b": 1, "c": 0, "d": 0}
        self.assertEqual(test(0, 0, 0, 0), expected)
        self.assertEqual(test(0, b=0, c=0, d=0), expected)
        self.assertEqual(test(a=0, b=0, c=0, d=0), expected)


class DependencyDistillTest(unittest.TestCase):
    from utilitypack.cold.util_solid import DistillLibraryFromDependency

    def test_distill(self):
        code = ["Stream()"]
        lib = [
            f"class Stream:...",
            f"import Stream",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, f"class Stream:...")

    def test_definition_covers_importing(self):
        code = ["Stream()"]
        lib = [
            f"from somewhere import Stream",
            f"import Stream",
            f"class Stream:...",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, f"class Stream:...")

    def test_indented_definition(self):
        code = ["Stream()"]
        lib = [
            f"""\
if True:
    class Stream:
        ...
else:...\
""",
        ]
        result = DistillLibraryFromDependency.DistillLibrary(code, lib)
        self.assertEqual(result, lib[0])

    def test_usage_before_definition(self):
        code = f"""\
def foo(a:typing.Any):...
import typing
"""
        result = DistillLibraryFromDependency.find_undefined_variables(code)
        self.assertSetEqual(result, {"typing"})


class StreamTest(unittest.TestCase):

    def test_basic_use(self):
        result = Stream(range(10)).count()
        self.assertEqual(result, 10)

    def test_map(self):
        result = Stream([1, 2, 3]).map(lambda x: x * 2).to_list()
        self.assertEqual(result, [2, 4, 6])

    def test_flat_map(self):
        result = Stream([[1, 2], [3, 4]]).flat_map(lambda x: Stream(x)).to_list()
        self.assertEqual(result, [1, 2, 3, 4])

    def test_filter(self):
        result = Stream([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).to_list()
        self.assertEqual(result, [2, 4])

    def test_for_each(self):
        result = 0

        def add_to_result(x):
            nonlocal result
            result += x

        Stream(range(4)).for_each(add_to_result)
        self.assertEqual(result, 6)

    def test_peek(self):
        seen = []
        result = Stream([1, 2, 3]).peek(lambda x: seen.append(x)).to_list()
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(seen, [1, 2, 3])

    def test_distinct(self):
        result = Stream([1, 2, 2, 3]).distinct().to_list()
        self.assertEqual(result, [1, 2, 3])

    def test_sorted(self):
        result = Stream([3, 1, 2]).sorted().to_list()
        self.assertEqual(result, [1, 2, 3])

    def test_count(self):
        result = Stream([1, 2, 3]).count()
        self.assertEqual(result, 3)

    def test_sum(self):
        result = Stream([1, 2, 3]).sum()
        self.assertEqual(result, 6)

    def test_group_by(self):
        result = Stream([1, 2, 3, 4]).group_by(lambda x: x % 2)
        self.assertEqual(result[0], [2, 4])
        self.assertEqual(result[1], [1, 3])

    def test_reduce(self):
        result = Stream([1, 2, 3, 4]).reduce(lambda a, b: a + b)
        self.assertEqual(result, 10)

    def test_limit(self):
        result = Stream(range(100)).limit(5).to_list()
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_skip(self):
        result = Stream([0, 1, 2, 3, 4]).skip(2).to_list()
        self.assertEqual(result, [2, 3, 4])

    def test_min_max(self):
        s = lambda: Stream([3, 1, 4, 1, 5])
        self.assertEqual(s().min(), 1)
        self.assertEqual(s().max(), 5)
        self.assertTupleEqual(s().minmax(), (1, 5))

    def test_find_first(self):
        self.assertEqual(Stream([1, 2, 3]).find_first(), 1)
        self.assertIsNone(Stream([]).find_first())

    def test_any_all_none_match(self):
        s = lambda: Stream([2, 4, 6])
        self.assertTrue(s().any_match(lambda x: x % 2 == 0))
        self.assertTrue(s().all_match(lambda x: x % 2 == 0))
        self.assertTrue(s().none_match(lambda x: x > 10))

        s2 = lambda: Stream([1, 3, 5])
        self.assertFalse(s2().any_match(lambda x: x % 2 == 0))
        self.assertFalse(s2().all_match(lambda x: x % 2 == 0))
        self.assertFalse(s2().none_match(lambda x: x <= 1))

    def test_to_dict(self):
        result = Stream([(1, "a"), (2, "b")]).to_dict(lambda x: x[0], lambda x: x[1])
        self.assertEqual(result, {1: "a", 2: "b"})

    def test_to_set(self):
        result = Stream([1, 1, 2]).to_set()
        self.assertSetEqual(result, {1, 2})

    def test_gather_async(self):
        import asyncio

        async def async_task(x: int):
            await asyncio.sleep(0)
            return x + 1

        result = Stream([1, 2, 3]).map(async_task).gather_async().to_list()
        self.assertEqual(result, [2, 3, 4])

    def test_gather_thread(self):
        import concurrent.futures

        def task(x: int):
            return x + 1

        pool = concurrent.futures.ThreadPoolExecutor()
        result = (
            Stream([1, 2, 3])
            .map(lambda x: pool.submit(task, x=x))
            .gather_thread_future()
            .collect(list)
        )
        self.assertEqual(result, [2, 3, 4])

    def test_reversed(self):
        result = Stream([1, 2, 3]).reversed().to_list()
        self.assertEqual(result, [3, 2, 1])

    def test_auto_unpacked(self):
        result = Stream(range(4)).wrap_iterator(enumerate).map(lambda a, b: b).sum()
        self.assertEqual(result, 6)

    def test_auto_awaited(self):
        async def add1(x: int):
            return x + 1

        result = Stream(range(3)).map(add1).map(add1).gather_async().sum()
        self.assertEqual(result, 9)

    def test_auto_unpacked_and_awaited(self):
        async def add1(x: int, y: int):
            return x + 1, y + 1

        async def sum(x: int, y: int):
            return x + y

        result = (
            Stream(range(3))
            .wrap_iterator(enumerate)
            .map(add1)
            .map(sum)
            .gather_async()
            .sum()
        )
        self.assertEqual(result, 12)

    def test_collect_to_string_io(self):
        result = Stream(range(3)).map(str).collect(Stream.Collectors.stringIo())
        result.seek(0)
        result = result.read()
        self.assertEqual(result, "012")

    def test_collect_to_print(self):
        buf = io.StringIO()
        Stream(range(3)).collect(Stream.Collectors.print(end="", flush=True, file=buf))
        buf.seek(0)
        result = buf.read()
        self.assertEqual(result, "012")


class ThreadContextTest(unittest.TestCase):

    class ChildClass(ThreadLocalSingleton):
        def __init__(self):
            super().__init__()
            self.v = None

    class GrandsonClass(ChildClass): ...

    def if_existance(self, cls: ChildClass, expected_existance: bool):
        existance = cls.summon().v is not None
        self.assertEqual(existance, expected_existance)

    def test_accessable_in_thread(self):
        ThreadContextTest.ChildClass.summon().v = 1
        self.if_existance(ThreadContextTest.ChildClass, True)

    def test_inaccessable_in_another_thread(self):
        ThreadContextTest.ChildClass.summon().v = 1
        th = threading.Thread(
            target=ThreadContextTest.if_existance,
            args=(
                self,
                ThreadContextTest.ChildClass,
                False,
            ),
        )
        th.start()
        th.join()

    def test_inherited_again(self):
        ThreadContextTest.GrandsonClass.summon().v = 1
        self.if_existance(ThreadContextTest.GrandsonClass, True)
        th = threading.Thread(
            target=ThreadContextTest.if_existance,
            args=(
                self,
                ThreadContextTest.GrandsonClass,
                False,
            ),
        )
        th.start()
        th.join()

    def test_with_me_method_decorator(self):
        """
        测试WithMe方法作为装饰器使用时的功能
        """

        # 使用WithMe装饰器装饰一个函数
        @ThreadContextTest.ChildClass.WithMe()
        def test_function():
            instance = ThreadContextTest.ChildClass.summon()
            instance.v = 42
            return instance.v

        def new_thread():
            # 创建一个线程，并调用装饰后的函数
            result = test_function()
            # 验证函数被调用并且返回了正确值
            self.assertEqual(result, 42)
            # 资源在decorator的上下文结束后被立即释放，即使线程还未结束
            self.assertFalse(
                ThreadContextTest.ChildClass.__thread_local_singleton_test_val__()
            )

        t = threading.Thread(target=new_thread)
        t.start()
        t.join()


class SingletonTest(unittest.TestCase):
    @staticmethod
    def decl_singleton():
        @Singleton
        class Sg:
            def __init__(self, param):
                self.param = param

        return Sg

    def test_basic(self):
        Sg = SingletonTest.decl_singleton()
        ins = Sg(1)
        self.assertEqual(ins.param, 1)
        ins2 = Sg(2)
        self.assertEqual(ins2.param, 1)

    # def test_inheritance(self):
    #     Sg = SingletonTest.decl_singleton()

    #     class ChildSg(Sg):
    #         def __init__(self, param):
    #             # actually not called in current implementation
    #             super().__init__(param)
    #             self.is_child = True

    #     ins = ChildSg(1)
    #     self.assertEqual(ins.param, 1)
    #     self.assertEqual(ins.is_child, True)
    #     ins2 = ChildSg(2)
    #     self.assertEqual(ins2.param, 1)
    #     self.assertEqual(ins.is_child, True)


class TestReadWriteLock(unittest.TestCase):
    def setUp(self):
        self.rwlock = ReadWriteLock()
        self.shared_resource = 0
        self.read_count = 0
        self.write_count = 0

    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.rwlock, ReadWriteLock)

    def test_acquire_release_read_lock(self):
        """测试获取和释放读锁"""
        self.rwlock.acquire_read()
        # 验证内部状态
        self.assertEqual(self.rwlock._readers, 1)
        self.rwlock.release_read()
        self.assertEqual(self.rwlock._readers, 0)

    def test_acquire_release_write_lock(self):
        """测试获取和释放写锁"""
        self.rwlock.acquire_write()
        self.assertEqual(self.rwlock._writers, 1)
        self.assertEqual(self.rwlock._pending_writers, 0)
        self.rwlock.release_write()
        self.assertEqual(self.rwlock._writers, 0)

    def test_multiple_readers(self):
        """测试多个读锁可以同时获取"""

        def reader():
            self.rwlock.acquire_read()
            time.sleep(0.1)  # 模拟读操作
            self.rwlock.release_read()

        threads = []
        for _ in range(5):
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有读锁都已释放
        self.assertEqual(self.rwlock._readers, 0)

    def test_writer_blocks_reader(self):
        """测试写锁会阻塞读锁"""
        # 获取写锁
        self.rwlock.acquire_write()

        reader_acquired = threading.Event()

        def reader():
            self.rwlock.acquire_read()
            reader_acquired.set()
            self.rwlock.release_read()

        # 启动读线程
        t = threading.Thread(target=reader)
        t.start()

        # 等待一小段时间，确保读线程尝试获取锁
        time.sleep(0.1)

        # 读线程应该被阻塞
        self.assertFalse(reader_acquired.is_set())

        # 释放写锁
        self.rwlock.release_write()

        # 等待读线程完成
        t.join(timeout=1)

        # 读线程现在应该能获取到锁
        self.assertTrue(reader_acquired.is_set())

    def test_reader_blocks_writer(self):
        """测试读锁会阻塞写锁"""
        # 获取读锁
        self.rwlock.acquire_read()

        writer_acquired = threading.Event()

        def writer():
            self.rwlock.acquire_write()
            writer_acquired.set()
            self.rwlock.release_write()

        # 启动写线程
        t = threading.Thread(target=writer)
        t.start()

        # 等待一小段时间，确保写线程尝试获取锁
        time.sleep(0.1)

        # 写线程应该被阻塞
        self.assertFalse(writer_acquired.is_set())

        # 释放读锁
        self.rwlock.release_read()

        # 等待写线程完成
        t.join(timeout=1)

        # 写线程现在应该能获取到锁
        self.assertTrue(writer_acquired.is_set())

    def test_context_manager_read(self):
        """测试读锁的上下文管理器"""
        with self.rwlock.gen_rlock():
            self.assertEqual(self.rwlock._readers, 1)
        self.assertEqual(self.rwlock._readers, 0)

    def test_context_manager_write(self):
        """测试写锁的上下文管理器"""
        with self.rwlock.gen_wlock():
            self.assertEqual(self.rwlock._writers, 1)
        self.assertEqual(self.rwlock._writers, 0)

    def test_write_preference(self):
        """测试写锁优先（避免写锁饥饿）"""
        # 先让一些读锁等待
        reader_events = [threading.Event() for _ in range(3)]
        writer_event = threading.Event()

        def reader(i):
            with self.rwlock.gen_rlock():
                reader_events[i].set()

        def writer():
            with self.rwlock.gen_wlock():
                writer_event.set()

        # 先启动写线程（让它排队）
        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        # 等待一小段时间确保写线程已经开始等待
        time.sleep(0.1)

        # 然后启动读线程
        reader_threads = []
        for i in range(3):
            t = threading.Thread(target=reader, args=(i,))
            reader_threads.append(t)
            t.start()

        # 给一点时间让所有读线程尝试获取锁
        time.sleep(0.1)

        # 写线程应该在读线程之前获得锁
        self.assertTrue(writer_event.is_set())

        # 等待所有线程完成
        writer_thread.join()
        for t in reader_threads:
            t.join()


class TestAutoSizableNdarray(unittest.TestCase):

    def test_1d_array(self):
        """测试一维数组功能"""
        arr = AutoSizableNdarray(ndim=1)

        # 测试设置和获取单个值
        arr.set([5], 42)
        self.assertEqual(arr.get([5]), 42)

        # 测试自动扩展
        arr.set([10], 100)
        self.assertEqual(arr.get([10]), 100)
        self.assertEqual(arr.get([5]), 42)  # 确保原来的值还在

        # 测试负索引
        arr.set([-5], -10)
        self.assertEqual(arr.get([-5]), -10)

    def test_2d_array(self):
        """测试二维数组功能（原始功能）"""
        arr = AutoSizableNdarray(ndim=2)

        # 测试设置和获取单个值
        arr.set([2, 3], 5)
        self.assertEqual(arr.get([2, 3]), 5)

        # 测试自动扩展
        arr.set([5, 7], 15)
        self.assertEqual(arr.get([5, 7]), 15)
        self.assertEqual(arr.get([2, 3]), 5)  # 确保原来的值还在

        # 测试批量设置
        values = np.array([[1, 2, 3], [4, 5, 6]])
        arr.batch_set([0, 0], values)

        # 验证批量设置的值
        result = arr.batch_get([0, 0], [2, 3])
        np.testing.assert_array_equal(result, values)

    def test_3d_array(self):
        """测试三维数组功能"""
        arr = AutoSizableNdarray(ndim=3)

        # 测试设置和获取单个值
        arr.set([1, 2, 3], 99)
        self.assertEqual(arr.get([1, 2, 3]), 99)

        # 测试自动扩展
        arr.set([5, 10, 15], 55)
        self.assertEqual(arr.get([5, 10, 15]), 55)
        self.assertEqual(arr.get([1, 2, 3]), 99)  # 确保原来的值还在

        # 测试批量设置
        values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        arr.batch_set([0, 0, 0], values)

        # 验证批量设置的值
        result = arr.batch_get([0, 0, 0], [2, 2, 2])
        np.testing.assert_array_equal(result, values)

    def test_4d_array(self):
        """测试四维数组功能"""
        arr = AutoSizableNdarray(ndim=4)

        # 测试设置和获取单个值
        arr.set([1, 2, 3, 4], 1234)
        self.assertEqual(arr.get([1, 2, 3, 4]), 1234)

        # 测试自动扩展
        arr.set([2, 4, 6, 8], 2468)
        self.assertEqual(arr.get([2, 4, 6, 8]), 2468)
        self.assertEqual(arr.get([1, 2, 3, 4]), 1234)  # 确保原来的值还在

    def test_tight_data(self):
        """测试tight_data方法"""
        arr = AutoSizableNdarray(ndim=2)

        # 设置一些值
        arr.set([5, 5], 1)
        arr.set([7, 8], 2)

        # 获取紧凑数据
        tight = arr.tight_data()

        # 验证形状和内容
        self.assertEqual(tight.shape, (3, 4))  # 从(5,5)到(7,8)的范围，即3行4列
        self.assertEqual(tight[0, 0], 1)  # 对应坐标(5,5)
        self.assertEqual(tight[2, 3], 2)  # 对应坐标(7,8)

    def test_different_data_types(self):
        """测试不同的数据类型"""
        # 测试浮点数
        arr = AutoSizableNdarray(ndim=2)
        arr.set([0, 0], 3.14)
        self.assertEqual(arr.get([0, 0]), 3.14)

        # 测试字符串（对象数组）
        arr2 = AutoSizableNdarray(ndim=1)
        arr2.set([0], "test")
        self.assertEqual(arr2.get([0]), "test")

    def test_edge_cases(self):
        """测试边界情况"""
        arr = AutoSizableNdarray(ndim=2)

        # 测试零值
        arr.set([0, 0], 0)
        self.assertEqual(arr.get([0, 0]), 0)

        # 测试负坐标
        arr.set([-1, -1], -5)
        self.assertEqual(arr.get([-1, -1]), -5)

        # 测试大坐标值
        arr.set([1000, 2000], 100)
        self.assertEqual(arr.get([1000, 2000]), 100)


class SingletonTest(unittest.TestCase):

    def test_global_isolation(self):
        class A(SingletonGlobalIsolation): ...

        class B(A): ...

        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b = B.get_instance()
        self.assertTrue(b is not a1)

    def test_thread_isolation(self):

        class A(SingletonThreadIsolation): ...

        class B(A): ...

        # 在主线程中获取实例
        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b1 = B.get_instance()
        self.assertTrue(b1 is not a1)

        # 结果存储用于从另一个线程返回数据
        results = {}

        def worker():
            # 在工作线程中获取实例
            at = A.get_instance()
            bt = B.get_instance()

            # 验证线程隔离：不同线程应该有不同的实例
            results["a_same_thread"] = at is A.get_instance()
            results["b_same_thread"] = bt is B.get_instance()
            results["a_diff_thread"] = at is not a1
            results["b_diff_thread"] = bt is not b1

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        # 检查结果
        self.assertTrue(results["a_same_thread"])
        self.assertTrue(results["b_same_thread"])
        self.assertTrue(results["a_diff_thread"])
        self.assertTrue(results["b_diff_thread"])

    def test_context_isolation(self):
        class A(SingletonContextIsolation): ...

        class B(A): ...

        # 在当前上下文中获取实例
        a1 = A.get_instance()
        a2 = A.get_instance()
        self.assertTrue(a1 is a2)
        b1 = B.get_instance()
        self.assertTrue(b1 is not a1)

        # 结果存储用于从另一个上下文返回数据
        results = {}

        def worker():
            # 在新上下文中获取实例
            at = A.get_instance()
            bt = B.get_instance()

            # 验证上下文隔离：不同上下文应该有不同的实例
            results["a_same_context"] = at is A.get_instance()
            results["b_same_context"] = bt is B.get_instance()
            results["a_diff_context"] = at is not a1
            results["b_diff_context"] = bt is not b1

        # 创建新上下文并运行
        ctx = contextvars.Context()
        ctx.run(worker)

        # 检查结果
        self.assertTrue(results["a_same_context"])
        self.assertTrue(results["b_same_context"])
        self.assertTrue(results["a_diff_context"])
        self.assertTrue(results["b_diff_context"])


unittest.main()
