from .util_solid import NormalizeIterableOrSingleArgToIterable
import itertools
import typing
import dataclasses

"""
numpy
"""

import numpy as np
import hashlib

np.seterr(under="ignore")

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
        assert np.fabs(x1 - x2) > EPS
        if x1 < x2:
            ptleft = np.array([x1, y1])
            ptright = np.array([x2, y2])
        else:
            ptright = np.array([x1, y1])
            ptleft = np.array([x2, y2])
        self.yminmax = np.array([min(y1, y2), max(y1, y2)])
        self.slope = (ptright[1] - ptleft[1]) / (ptright[0] - ptleft[0])
        self.bias = ptleft[1] - self.slope * ptleft[0]

    def __CallOnNDArray(self, x: np.ndarray):
        y = self.slope * x + self.bias
        y = np.clip(y, self.yminmax[0], self.yminmax[1])
        return y

    def __CallOnNum(self, x):
        y = self.slope * x + self.bias
        if y < self.yminmax[0]:
            y = self.yminmax[0]
        elif y > self.yminmax[1]:
            y = self.yminmax[1]
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

    logSumLowerLimit = -100

    def __post_init__(self):
        self.logPBASum = np.zeros_like(self.xspace)

    def update(self, measuredValue: float | list[float] | np.ndarray):
        measuredValue = NormalizeIterableOrSingleArgToNdarray(measuredValue)
        measuredValue = np.array(measuredValue)
        P_B_under_A = self.distributionModel(
            self.xspace.reshape((-1, 1)), measuredValue.reshape((1, -1))
        )
        self.logPBASum += np.sum(SafeLog(P_B_under_A), axis=1)
        self.logPBASum -= np.max(self.logPBASum)
        self.logPBASum[self.logPBASum < BayesEstimator.logSumLowerLimit] = (
            BayesEstimator.logSumLowerLimit
        )

    def getPossibility(self):
        possibility = SafeExp(self.logPBASum)
        possibility /= np.sum(possibility)
        return possibility


def NpGeneratorFromStrSeed(s: str):
    seed = s.encode("utf-8")
    seed = hashlib.md5(seed).digest()
    seed = int.from_bytes(seed[:8])
    return np.random.Generator(np.random.PCG64(seed))


def SpatialVectorWithShapeLikeX(x: np.ndarray, axis: int):
    return [1 if i != axis else -1 for i in range(len(x.shape))]


class AutoSizableNdarray:
    data: np.ndarray = None
    alloced_aabb: list = None
    """
    def set(self, x, z, value)
        如果data=aabb=None
            初始化data为1x1array，aabb=[x,x+1,z,z+1]
        否则如果xz不在aabb内
            将data扩展到包含xz范围
            更新aabb
        将data[x-aabb[0],z-aabb[2]]设置为value
    def get(self, x, z)
        如果xz在aabb内
            返回data[x-aabb[0],z-aabb[2]]
        否则
            返回0
    """

    def __init__(self):
        self.data: np.ndarray = None
        self.alloced_aabb: list = None  # [x_min, x_max, z_min, z_max]
        self.used_aabb: list = None

    def _expand_aabb_just_enough(self, x, z, aabb):
        if x < aabb[0]:
            aabb[0] = x
        elif x >= aabb[1]:
            aabb[1] = x
        if z < aabb[2]:
            aabb[2] = z
        elif z >= aabb[3]:
            aabb[3] = z
        return aabb

    def _expand_aabb_pow_of_2(self, x, z, aabb):
        # 计算新的 x 范围
        current_x_min, current_x_max = aabb[0], aabb[1]
        current_z_min, current_z_max = aabb[2], aabb[3]

        def exp2CeilLog2(x: int):
            return int(np.exp2(np.ceil(np.log2(x))))

        def expand_range(current_min, current_max, new_value):
            new_min = current_min
            new_max = current_max
            if new_value < current_min:
                req_size = current_max - new_value
                new_size = exp2CeilLog2(req_size)
                new_min = current_max - new_size
            elif new_value >= current_max:
                req_size = new_value - current_min + 1
                new_size = exp2CeilLog2(req_size)
                new_max = current_min + new_size
            return new_min, new_max

        new_x_min, new_x_max = expand_range(current_x_min, current_x_max, x)
        new_z_min, new_z_max = expand_range(current_z_min, current_z_max, z)
        return [new_x_min, new_x_max, new_z_min, new_z_max]

    def batch_set(self, xbeg, zbeg, values: np.ndarray):
        """
        在xb:xb+values.shape[0]行，zb:zb+values.shape[1]列，将values赋值给self.data
        需要将xb和zb依据self.alloced_aabb进行转换
        如果点xb,zb或xb+values.shape[0],zb:zb+values.shape[1]不在self.alloced_aabb中，则使用_expand_aabb_pow_of_2扩展aabb
        """
        xend = xbeg + values.shape[0]
        zend = zbeg + values.shape[1]
        # 如果未初始化，使用 set 方法逐个设置
        if self.not_inited():
            self.data = np.copy(values)
            self.alloced_aabb = [xbeg, xend, zbeg, xend]
            self.used_aabb = [xbeg, xend, zbeg, xend]
        elif not (self.is_in_aabb(xbeg, zbeg) and self.is_in_aabb(xend, zend)):
            # 需要扩展
            # 计算新的 aabb（包括整个 values 区域）
            new_aabb = self.alloced_aabb
            if not self.is_in_aabb(xbeg, zbeg, new_aabb):
                new_aabb = self._expand_aabb_pow_of_2(xbeg, zbeg, new_aabb)
            if not self.is_in_aabb(xend, zend, new_aabb):
                new_aabb = self._expand_aabb_pow_of_2(xend, zend, new_aabb)
            self.resize(new_aabb)

        # 将 values 数据复制到正确的位置
        target_xbeg, target_zbeg = self.xz2xzInData(xbeg, zbeg)
        target_xend, target_zend = self.xz2xzInData(xend, zend)

        self.data[target_zbeg:target_zend, target_xbeg:target_xend] = values

        # 更新 used_aabb 以包含新设置的区域=
        self.used_aabb = self._expand_aabb_just_enough(xbeg, zbeg, self.used_aabb)
        self.used_aabb = self._expand_aabb_just_enough(xend, zend, self.used_aabb)

    def set(self, x, z, value):
        """设置指定坐标点的值，自动扩展数组大小"""
        if self.not_inited():
            # 初始化 1x1 数组
            self.data = np.zeros((1, 1), dtype=int)
            self.alloced_aabb = [x, x + 1, z, z + 1]
            self.used_aabb = [x, x + 1, z, z + 1]
        elif not self.is_in_aabb(x, z):
            # 需要扩展数组
            new_aabb = self._expand_aabb_pow_of_2(x, z, self.alloced_aabb)
            self.resize(new_aabb)
        if not self.is_in_aabb(x, z, self.used_aabb):
            self.used_aabb = self._expand_aabb_just_enough(x, z, self.used_aabb)
        self.data[z - self.alloced_aabb[2], x - self.alloced_aabb[0]] = value

    def resize(self, new_aabb):
        current_x_min, current_x_max = self.alloced_aabb[0], self.alloced_aabb[1]
        current_z_min, current_z_max = self.alloced_aabb[2], self.alloced_aabb[3]

        new_x_min, new_x_max, new_z_min, new_z_max = new_aabb

        # 创建新数组
        new_shape = (new_z_max - new_z_min, new_x_max - new_x_min)
        new_data = np.empty(new_shape, dtype=self.data.dtype)

        # 计算旧数据在新数组中的位置
        old_x_start = current_x_min - new_x_min
        old_z_start = current_z_min - new_z_min

        # 复制旧数据到新数组
        new_data[
            old_z_start : old_z_start + self.data.shape[0],
            old_x_start : old_x_start + self.data.shape[1],
        ] = self.data

        # 更新属性
        self.data = new_data
        self.alloced_aabb = new_aabb

    def xz2xzInData(self, x, z):
        return x - self.alloced_aabb[0], z - self.alloced_aabb[2]

    def get(self, x, z):
        """获取指定坐标点的值"""
        assert not self.not_inited() and self.is_in_aabb(x, z)
        return self.data[self.xz2xzInData(x, z)]

    def batch_get(self, x, z):
        """获取指定坐标点的值"""
        assert self.not_inited() and self.is_in_aabb(x, z)
        return self.data[self.xz2xzInData(x, z)]

    def not_inited(self):
        assert (self.alloced_aabb is None) == (
            self.alloced_aabb is None or self.data is None or self.used_aabb is None
        )
        return self.alloced_aabb is None

    def is_in_aabb(self, x, z, aabb=None):
        if self.not_inited():
            return False
        if aabb is None:
            aabb = self.alloced_aabb
        return aabb[0] <= x < aabb[1] and aabb[2] <= z < aabb[3]

    def tight_data(self):
        # not really trimed, just return the tight data slice
        assert not self.not_inited()
        xb, zb = self.xz2xzInData(self.used_aabb[0], self.used_aabb[2])
        xe, ze = self.xz2xzInData(self.used_aabb[1], self.used_aabb[3])
        return self.data[zb:ze, xb:xe]
