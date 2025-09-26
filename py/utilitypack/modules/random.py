import random


def InProbability(p: float) -> bool:
    return random.random() < p


def FlipCoin() -> bool:
    return InProbability(0.5)


class RandomString:
    class Charsets:
        DIGIT = "0123456789"
        LOWER = "abcdefghijklmnopqrstuvwxyz"
        UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        UPPER_DIGIT = DIGIT + UPPER

    def __new__(_, length, charset=Charsets.UPPER_DIGIT):
        return "".join(random.choices(charset, k=length))
