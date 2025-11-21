from test.autotest_common import *


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
