
import random


def InProbability(p: float) -> bool:
    return random.random() < p


def FlipCoin() -> bool:
    return InProbability(0.5)

