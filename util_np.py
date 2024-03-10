from .util_solid import *

"""
numpy
"""

import numpy as np

np.seterr(all="raise")

EPS = 1e-10
LOGEPS = -10


def forceviewmaxmin(m):
    ma = m.max()
    mi = m.min()
    print("ma", ma)
    print("mi", mi)
    pass


def arrayshift(a, n, fill=np.nan):
    # n positive as right
    if n == 0:
        return a
    elif n > 0:
        # a[:-n] will not work as intended on n==0
        return np.concatenate((np.full(n, fill), a[:-n]))
    else:
        return np.concatenate((a[-n:], np.full(-n, fill)))


def integral(dx, x0, keepXM1=False):
    # keepXM1 to keep the last element in x, or deprecate it, cuz its nan if generated from derivative
    x = np.array(list(itertools.accumulate(dx, lambda t, e: t + e)))
    x = np.concatenate([[x0], x if keepXM1 else x[:-1]])
    return x


def derivative(x):
    return arrayshift(x, -1) - x


def summonCard(inteprob, generator=None):
    # summon from card pool
    # impl using np
    # norm
    prob = np.array(inteprob, dtype=np.float32)
    prob /= prob.sum()
    if generator is None:
        return np.random.choice(np.arange(len(prob)), p=prob)
    else:
        return generator.choice(np.arange(len(prob)), p=prob)


class ZFunc:
    """
    x1x2 at any order
    """

    def __init__(self, x1, y1, x2, y2) -> None:
        if x1 < x2:
            # [lower on x or higher on x, x or y]
            self.pt = np.array([[x1, y1], [x2, y2]])
        else:
            self.pt = np.array([[x2, y2], [x1, y1]])
        self.slope = (self.pt[1, 1] - self.pt[0, 1]) / (
            self.pt[1, 0] - self.pt[0, 0] + 0.0001
        )
        self.bias = self.pt[0, 1] - self.pt[0, 0] * self.slope

    def __CallOnNDArray(self, x: np.ndarray):
        y = self.slope * x + self.bias
        y[x < self.pt[0, 0]] = self.pt[0, 1]
        y[x > self.pt[1, 0]] = self.pt[1, 1]
        return y

    def __CallOnNum(self, x):
        if x < self.pt[0, 0]:
            y = self.pt[0, 1]
        elif x > self.pt[1, 0]:
            y = self.pt[1, 1]
        else:
            y = self.slope * x + self.bias
        return y

    def __call__(self, x):
        if type(x) is np.ndarray:
            return self.__CallOnNDArray(x)
        else:
            return self.__CallOnNum(x)


def randomString(charset, length):
    return "".join(
        [
            charset[i]
            for i in np.random.choice(range(len(charset)), length, replace=True)
        ]
    )


def ReLU(x):
    return np.maximum(0, x)


def SafeExp(x):
    x[x < LOGEPS] = LOGEPS
    y = np.exp(x)
    y[x < LOGEPS] = 0
    return y


def SafeLog(x):
    x[x < EPS] = EPS
    y = np.log(x)
    return y

def NormalizeIterableOrSingleArgToNdarray(x):
    if type(x) is np.ndarray:
        return x
    return np.array(NormalizeIterableOrSingleArgToIterable(x))

@dataclasses.dataclass
class BayesEstimator:
    xspace: np.ndarray
    distributionModel: typing.Callable  # to calc P(measuredVal=B|val=A)
    logPBASum: np.ndarray = dataclasses.field(init=False, default=None)

    logSumLowerLimit = -500

    def __post_init__(self):
        self.logPBASum = np.zeros_like(self.xspace)

    def update(self, measuredValue: float | list[float] | np.ndarray):
        measuredValue = NormalizeIterableOrSingleArgToNdarray(measuredValue)
        measuredValue = np.array(measuredValue)
        PBA = self.distributionModel(
            self.xspace.reshape((-1, 1)), measuredValue.reshape((1, -1))
        )
        self.logPBASum += np.sum(SafeLog(PBA), axis=1)
        self.logPBASum -= np.max(self.logPBASum)
        self.logPBASum[self.logPBASum < BayesEstimator.logSumLowerLimit] = (
            BayesEstimator.logSumLowerLimit
        )

    def getPossibility(self):
        possibility = SafeExp(self.logPBASum)
        possibility /= np.sum(possibility)
        return possibility
