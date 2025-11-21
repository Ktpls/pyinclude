import sys
import typing


class OneLineInputCmdScript:
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
            for arg in sys.argv[1:] or input_loop_impl:
                foo(arg)
                print("#" * 100)

        return new_f
