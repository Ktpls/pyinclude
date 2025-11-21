from test.autotest_common import *


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
