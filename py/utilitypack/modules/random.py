import random
import dataclasses


def InProbability(p: float) -> bool:
    return random.random() < p


def FlipCoin() -> bool:
    return InProbability(0.5)


@dataclasses.dataclass
class RandomString:
    class Charsets:
        DIGIT = "0123456789"
        LOWER = "abcdefghijklmnopqrstuvwxyz"
        UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        UPPER_DIGIT = DIGIT + UPPER

    length: int
    charset: str = Charsets.UPPER_DIGIT

    def __call__(self):
        return "".join(random.choices(self.charset, k=self.length))
