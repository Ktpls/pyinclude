import sys
import typing
import traceback


class OneLineInputCmdScript:
    @staticmethod
    def InputCmd():
        yield from sys.argv[1:]

    @staticmethod
    def InputCmdIfPresentedOrElse(
        another_input_source: typing.Generator[str, None, None],
    ):
        if len(sys.argv) > 1:
            yield from OneLineInputCmdScript.InputCmd()
        else:
            yield from another_input_source

    @staticmethod
    def InputLoop(prompt=">>> "):
        while True:
            yield input(prompt)

    @staticmethod
    def InputLoopMultiline(prompt_first, prompt_next=">>>"):
        while True:
            linput: list[str] = []
            linput.append(input(prompt_first))
            while True:
                line = input(prompt_next)
                if line == "":
                    break
                linput.append(line)
            yield "\n".join(linput)

    def serve(self, input_loop_impl: typing.Generator[str, None, None] = InputLoop()):
        def new_f(foo: typing.Callable[[str], typing.Any]):
            for arg in OneLineInputCmdScript.InputCmdIfPresentedOrElse(input_loop_impl):
                try:
                    foo(arg)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                print("#" * 100)

        return new_f
