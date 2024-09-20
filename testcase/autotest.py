import unittest


from utilitypack.cold.util_solid import *
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


class UrlFullResolutionLazyTest(unittest.TestCase):
    def test_of(self):
        res = UrlFullResolution.of(
            r"https://picx.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c&needBackground=1"
        )
        res.calcAll()
        self.assertDictEqual(
            res._resultMap,
            {
                "baseHost": "zhimg.com",
                "domain": "com",
                "extName": ".jpg",
                "fileName": "v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg",
                "folder": "",
                "host": "picx.zhimg.com",
                "param": "?source=32738c0c&needBackground=1",
                "path": "/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg",
                "port": None,
                "protocol": "https://",
                "secondaryHost": "picx",
            },
        )


class MaxRetryTest(unittest.TestCase):
    def test_succ(self):
        result = 0
        retry = MaxRetry(lambda: result >= 2, maxRetry=3)
        for i in retry:
            result = i
        self.assertEqual(result, 2)
        self.assertEqual(retry.isSuccessed, True)

    def test_fail(self):
        result = 0
        retry = MaxRetry(lambda: result >= 5, maxRetry=3)
        for i in retry:
            result = i
        self.assertEqual(result, 3)
        self.assertEqual(retry.isSuccessed, False)


class HotkeyManagerTest(unittest.TestCase):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)

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


unittest.main()
