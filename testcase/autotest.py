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

    def do(self, exp: str, expected):
        try:
            result = str(
                expparser.expparse(
                    exp,
                    var=ExpparserTest.var,
                    func=ExpparserTest.func,
                )
            )
        except Exception as e:
            # raise e
            result = e
        if expected == Exception and isinstance(result, expected):
            pass
        else:
            self.assertEqual(result, expected)

    test_cstr = lambda self: self.do(r"CStr(1)", "1.0")
    test_array = lambda self: self.do(r"1,2,3", "[1.0, 2.0, 3.0]")
    test_div = lambda self: self.do(r"2/2/2", "0.5")
    test_oper_priortiy = lambda self: self.do(r"sin(pi/2)+2^2*2+--1", "10.0")
    test_eq_func = lambda self: self.do(r"eq(1+0.1,1)", "False")
    test_eq_func_eps = lambda self: self.do(r"eq(1+0.1,1,0.2)", "True")
    test_str = lambda self: self.do(r'"test \" str"', 'test " str')
    test_eq_opr = lambda self: self.do(r"1!=2", "True")
    test_comp_opr = lambda self: self.do(r"2>=3", "False")
    test_clist = lambda self: self.do(r"CList(1)", "[1.0]")
    test_cbool = lambda self: self.do(r"CBool(1)", "True")
    test_cbool2 = lambda self: self.do(r"CBool(0)", "False")
    test_streq = lambda self: self.do(r'StrEq(CStr(1),"1.0")', "True")
    test_cnum = lambda self: self.do(r'CNum("1.23")+1', "2.23")
    test_cnum2 = lambda self: self.do(r"CNum(true)+1", "2.0")
    test_unmatched_ket = lambda self: self.do(r"CBool(0))))))))", "False")
    test_space = lambda self: self.do("1 ,\t2,\r\n3", "[1.0, 2.0, 3.0]")
    test_vector = lambda self: self.do(r"vecadd((1,2),(3,4))", "[4.0, 6.0]")
    test_optional_func = lambda self: self.do(r"OptionalFunc(1,,)", "1.0")
    test_opt_func_bad_use = lambda self: self.do(r"OptionalFunc(,1,)", Exception)
    test_complex_array = (
        lambda self: self.do(
            r"((1,1),(2,2),(1,),1)", "[[1.0, 1.0], [2.0, 2.0], [1.0, None], 1.0]"
        ),
    )
    test_comment = (
        lambda self: self.do(
            r"""
                //comment one
                1
                /*
                comment
                two
                */
                +/*inline comment*/1
                """,
            "2.0",
        ),
    )


unittest.main()
