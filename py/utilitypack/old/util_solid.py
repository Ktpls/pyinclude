from ..modules.misc import *
from ..modules.time import *

# unused but kept for compatibility purposes


def BetterGroupBy(l: list, pred):
    # return {n: list(ll) for n, ll in itertools.groupby(sorted(l, key=pred), pred)}
    r = dict()
    for item in l:
        key = pred(item)
        if key in r:
            r[key].append(item)
        else:
            r[key] = [item]
    return r


def Coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def Deduplicate(l: list, key=None):
    key = Coalesce(key, IdentityMapping)
    m = BetterGroupBy(l, key)
    l = [v[0] for v in m.values()]
    return l


def Digitsof(s: str):
    return "".join(list(filter(str.isdigit, list(s))))


def Numinstr(s: str):
    # wont consider negative
    s = Digitsof(s)
    return int(s) if len(s) > 0 else 0


class Progress:
    """
    irreversable!
    """

    def __init__(self, total: float, cur=0, printPercentageStep: float = 0.1) -> None:
        self.total = total
        self.nowStage = 0
        self.printPercentageStep = printPercentageStep
        self.cur = cur
        self.ps = perf_statistic(startnow=True)

    def update(self, current: float) -> None:
        self.cur = current
        while True:
            if current / self.total > self.nowStage * self.printPercentageStep:
                self.nowStage += 1
                if current > 1:
                    self.ps.countcycle()
                    # not the first time
                    instantSpeed = (self.printPercentageStep * self.total) / (
                        self.ps.aveTime() + EPS
                    )
                else:
                    instantSpeed = 1
                print(
                    f"{100 * current / self.total:>3.2f}% of {self.total}, {instantSpeed:.2f}it/s",
                    end="\r",
                )
            else:
                break

    def setFinish(self):
        self.ps.stop()
        self.update(self.total)
        print("")
        print(f"finished within {self.ps.time():.2f}s")


class CollUtil:
    @staticmethod
    def split(iterable: typing.Iterable, size: int):
        return [
            iterable[i : min(i + size, len(iterable))]
            for i in range(0, len(iterable), size)
        ]
