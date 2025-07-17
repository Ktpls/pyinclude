from testcase.autotest_common import *
from utilitypack.cold.util_solid import *
from utilitypack.util_solid import *
from utilitypack.util_cracked import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *


class ExpparserTest(unittest.TestCase):

    @staticmethod
    def vecadd(va: list, vb: list):
        assert len(va) == len(vb)
        return list(map(lambda x, y: x + y, va, vb))

    var = {**expparser.BasicConstantLib}
    func = {
        **expparser.BasicFunctionLib,
        "OptionalFunc": expparser.Utils.OptionalFunc(
            [expparser.Utils.NonOptional(), 0, 0], lambda x, y, z: x + y + z
        ),
        "vecadd": vecadd,
    }

    def expparseWithEnv(self, exp):
        return str(
            expparser.expparse(
                exp,
                var=ExpparserTest.var,
                func=ExpparserTest.func,
            )
        )

    def test_cstr(self):
        self.assertEqual(self.expparseWithEnv("CStr(1)"), "1.0")

    def test_array(self):
        self.assertEqual(self.expparseWithEnv(r"1,2,3"), "[1.0, 2.0, 3.0]")

    def test_div(self):
        self.assertEqual(self.expparseWithEnv(r"2/2/2"), "0.5")

    def test_oper_priority(self):
        self.assertEqual(self.expparseWithEnv(r"sin(pi/2)+2^2*2+--1"), "10.0")

    def test_eq_func(self):
        self.assertEqual(self.expparseWithEnv(r"eq(1+0.1,1)"), "False")

    def test_eq_func_eps(self):
        self.assertEqual(self.expparseWithEnv(r"eq(1+0.1,1,0.2)"), "True")

    def test_str(self):
        self.assertEqual(self.expparseWithEnv(r'"test \" str"'), 'test " str')

    def test_eq_opr(self):
        self.assertEqual(self.expparseWithEnv(r"1!=2"), "True")

    def test_comp_opr(self):
        self.assertEqual(self.expparseWithEnv(r"2>=3"), "False")

    def test_clist(self):
        self.assertEqual(self.expparseWithEnv(r"CList(1)"), "[1.0]")

    def test_cbool(self):
        self.assertEqual(self.expparseWithEnv(r"CBool(1)"), "True")

    def test_cbool2(self):
        self.assertEqual(self.expparseWithEnv(r"CBool(0)"), "False")

    def test_streq(self):
        self.assertEqual(self.expparseWithEnv(r'StrEq(CStr(1),"1.0")'), "True")

    def test_cnum(self):
        self.assertEqual(self.expparseWithEnv(r'CNum("1.23")+1'), "2.23")

    def test_cnum2(self):
        self.assertEqual(self.expparseWithEnv(r"CNum(true)+1"), "2.0")

    def test_unmatched_ket(self):
        self.assertEqual(self.expparseWithEnv(r"CBool(0))))))))"), "False")

    def test_space(self):
        self.assertEqual(self.expparseWithEnv("1 ,\t2,\r\n3"), "[1.0, 2.0, 3.0]")

    def test_vector(self):
        self.assertEqual(self.expparseWithEnv(r"vecadd((1,2),(3,4))"), "[4.0, 6.0]")

    def test_optional_func(self):
        self.assertEqual(self.expparseWithEnv(r"OptionalFunc(1,,)"), "1.0")

    def test_opt_func_bad_use(self):
        self.assertRaises(Exception, lambda: self.expparseWithEnv(r"OptionalFunc(,1,)"))

    def test_complex_array(self):
        self.assertEqual(
            self.expparseWithEnv(r"((1,1),(2,2),(1,),1)"),
            "[[1.0, 1.0], [2.0, 2.0], [1.0, None], 1.0]",
        )

    def test_complex_array(self):
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
            "2.0",
        )

    def test_scientific_number(self):
        self.assertEqual(
            self.expparseWithEnv("1.1e-2"),
            "0.011",
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
            def __init__(self2, val) -> None:
                self2.val = val

            def inc(self2) -> None:
                self2.val += 1

            def assertVal(self2, val) -> None:
                self.assertEqual(self2.val, val)

            @staticmethod
            def staticMethod() -> None:
                nonlocal varStatic
                varStatic += 1

        a = Clz(1)
        a.inc().assertVal(2).inc().assertVal(3).staticMethod()
        self.assertEqual(a.val, 3)
        self.assertEqual(varStatic, 1)


class BeanUtilTest(unittest.TestCase):

    def test_CopySimpleClass(self):
        @dataclasses.dataclass
        class A:
            i: int
            s: str

        @dataclasses.dataclass
        class B:
            i: int
            s: str

        a = A(1, "s")
        b: B = BeanUtil.copyProperties(a, B)
        self.assertEqual(b.i, 1)

    def test_CopyNestedClass(self):
        @dataclasses.dataclass
        class A:
            i: int
            s: str

        @dataclasses.dataclass
        class B:
            a: A
            b: str

        inst = B(
            A(1, "s"),
            "s",
        )
        m = BeanUtil.toJsonCompatible(inst)
        self.assertDictEqual(
            m,
            {
                "a": {
                    "i": 1,
                    "s": "s",
                },
                "b": "s",
            },
        )
        inst: B = BeanUtil.copyProperties(m, B)
        self.assertEqual(inst.a.i, 1)

    def test_CopyList(self):
        @dataclasses.dataclass
        class A:
            i: int
            s: str

        @dataclasses.dataclass
        class B:
            la: list[A]
            b: str

        b = B(
            [A(0, "s"), A(1, "s")],
            "s",
        )
        m = BeanUtil.toJsonCompatible(b)
        self.assertDictEqual(
            m,
            {
                "la": [
                    {"i": 0, "s": "s"},
                    {"i": 1, "s": "s"},
                ],
                "b": "s",
            },
        )
        b2: B = BeanUtil.copyProperties(m, B)
        self.assertEqual(b2.la[1].i, 1)


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
                self.stage: Stage

                def sleep_specified_time(t):
                    t0 = self.stage.t
                    self.sleep(t)
                    selfTest.assertTrue(self.stage.t - t0 >= t)

                sleep_specified_time(3)
                sleep_specified_time(5)
                sleep_specified_time(10)

        pool = concurrent.futures.ThreadPoolExecutor()
        eosm = SyncExecutableManager(pool)
        stage = SyncExecutableTest._testbed(eosm)
        script = ScriptTest(stage, eosm).run()
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
        eosm = SyncExecutableManager(pool)
        stage = SyncExecutableTest._testbed(eosm)
        ms = SyncExecutableTest.ConsumerProducer.MainScript(stage, eosm).run(
            production, 5
        )
        ps = SyncExecutableTest.ConsumerProducer.ProducerScript(stage, eosm).run(
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
        eosm = SyncExecutableManager(pool)
        stage = SyncExecutableTest._testbed(eosm)
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

                recorder = RecorderScript(stage, eosm).run()
                for i in range(iteration):
                    value = i
                    self.sleep(1)
                recorderStopSignal = True
                self.stepOneFrame()

        ms = MainScriptLauchingThreadFromInside(stage, eosm).run(5)
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
        self.assertEqual(Stream(range(10)).count(), 10)

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
        self.assertEqual(Stream([1, 2, 3]).count(), 3)

    def test_sum(self):
        self.assertEqual(Stream([1, 2, 3]).sum(), 6)

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

        def task(x):
            return x + 1

        pool = concurrent.futures.ThreadPoolExecutor()
        # to set cuz wont keep in order
        result = (
            Stream([1, 2, 3])
            .map(lambda x: pool.submit(task, x=x))
            .gather_thread_future()
            .to_set()
        )
        self.assertSetEqual(result, {2, 3, 4})

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


# StreamTest().test_auto_unpacked_awaited()
unittest.main()
