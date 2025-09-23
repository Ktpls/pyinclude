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

    def __init__(self, ndim=2):
        self.ndim = ndim
        self.data: np.ndarray = None
        self.alloced_aabb: list = None  # [dim0_min, dim0_max, dim1_min, dim1_max, ..., dimN_min, dimN_max]
        self.used_aabb: list = None

    def _expand_aabb_just_enough(self, coords, aabb):
        """
        根据坐标扩展aabb刚好够用的空间
        coords: 坐标元组
        aabb: [dim0_min, dim0_max, dim1_min, dim1_max, ..., dimN_min, dimN_max]
        """
        new_aabb = aabb.copy()
        for i, coord in enumerate(coords):
            min_idx = i * 2
            max_idx = i * 2 + 1
            if coord < new_aabb[min_idx]:
                new_aabb[min_idx] = coord
            elif coord >= new_aabb[max_idx]:
                new_aabb[max_idx] = coord + 1
        return new_aabb

    def _expand_aabb_pow_of_2(self, coords, aabb):
        """
        根据坐标扩展aabb到2的幂次大小
        coords: 坐标元组
        aabb: [dim0_min, dim0_max, dim1_min, dim1_max, ..., dimN_min, dimN_max]
        """
        new_aabb = aabb.copy()
        
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

        for i, coord in enumerate(coords):
            min_idx = i * 2
            max_idx = i * 2 + 1
            current_min, current_max = new_aabb[min_idx], new_aabb[max_idx]
            new_min, new_max = expand_range(current_min, current_max, coord)
            new_aabb[min_idx], new_aabb[max_idx] = new_min, new_max
            
        return new_aabb

    def batch_set(self, coords_beg, values: np.ndarray):
        """
        在指定坐标开始的位置批量设置值
        coords_beg: 起始坐标元组
        values: 要设置的值的数组
        """
        coords_end = [coords_beg[i] + values.shape[i] for i in range(len(coords_beg))]
        
        # 如果未初始化，使用 values 初始化数组
        if self.not_inited():
            self.data = np.copy(values)
            self.alloced_aabb = []
            self.used_aabb = []
            for i in range(self.ndim):
                self.alloced_aabb.extend([coords_beg[i], coords_end[i]])
                self.used_aabb.extend([coords_beg[i], coords_end[i]])
        else:
            # 检查是否需要扩展
            needs_expansion = False
            temp_aabb = self.alloced_aabb.copy()
            
            # 检查起始点
            if not self.is_in_aabb(coords_beg):
                needs_expansion = True
                temp_aabb = self._expand_aabb_pow_of_2(coords_beg, temp_aabb)
                
            # 检查结束点
            if not self.is_in_aabb(coords_end):
                needs_expansion = True
                temp_aabb = self._expand_aabb_pow_of_2(coords_end, temp_aabb)
                
            if needs_expansion:
                self.resize(temp_aabb)

        # 将 values 数据复制到正确的位置
        target_beg = self.coords2indices(coords_beg)
        target_end = self.coords2indices(coords_end)
        
        slices = tuple(slice(target_beg[i], target_end[i]) for i in range(len(target_beg)))
        self.data[slices] = values

        # 更新 used_aabb 以包含新设置的区域
        self.used_aabb = self._expand_aabb_just_enough(coords_beg, self.used_aabb)
        self.used_aabb = self._expand_aabb_just_enough(coords_end, self.used_aabb)

    def set(self, coords, value):
        """设置指定坐标点的值，自动扩展数组大小"""
        if not isinstance(coords, (list, tuple)):
            coords = [coords]
            
        coords = list(coords)  # 确保是列表形式
        
        if self.not_inited():
            # 初始化 1x1x...x1 数组
            shape = tuple(1 for _ in range(self.ndim))
            self.data = np.zeros(shape, dtype=type(value) if type(value) != str else object)
            self.alloced_aabb = []
            self.used_aabb = []
            for coord in coords:
                self.alloced_aabb.extend([coord, coord + 1])
                self.used_aabb.extend([coord, coord + 1])
        elif not self.is_in_aabb(coords):
            # 需要扩展数组
            new_aabb = self._expand_aabb_pow_of_2(coords, self.alloced_aabb)
            self.resize(new_aabb)
            
        if not self.is_in_aabb(coords, self.used_aabb):
            self.used_aabb = self._expand_aabb_just_enough(coords, self.used_aabb)
            
        indices = self.coords2indices(coords)
        self.data[tuple(indices)] = value

    def resize(self, new_aabb):
        """
        调整数组大小到新的aabb
        new_aabb: [dim0_min, dim0_max, dim1_min, dim1_max, ..., dimN_min, dimN_max]
        """
        # 创建新数组
        new_shape = tuple(new_aabb[i*2+1] - new_aabb[i*2] for i in range(self.ndim))
        new_data = np.empty(new_shape, dtype=self.data.dtype)

        # 计算旧数据在新数组中的位置
        old_starts = []
        for i in range(self.ndim):
            old_start = self.alloced_aabb[i*2] - new_aabb[i*2]
            old_starts.append(old_start)

        # 复制旧数据到新数组
        old_slices = tuple(slice(0, self.data.shape[i]) for i in range(self.ndim))
        new_slices = tuple(slice(old_starts[i], old_starts[i] + self.data.shape[i]) for i in range(self.ndim))
        new_data[new_slices] = self.data[old_slices]

        # 更新属性
        self.data = new_data
        self.alloced_aabb = new_aabb

    def coords2indices(self, coords):
        """
        将坐标转换为数组内的索引
        coords: 坐标元组
        """
        return [coords[i] - self.alloced_aabb[i*2] for i in range(len(coords))]

    def get(self, coords):
        """获取指定坐标点的值"""
        if not isinstance(coords, (list, tuple)):
            coords = [coords]
        coords = list(coords)
        
        assert not self.not_inited() and self.is_in_aabb(coords)
        indices = self.coords2indices(coords)
        return self.data[tuple(indices)]

    def batch_get(self, coords_beg, shape):
        """批量获取指定坐标开始的值"""
        coords_end = [coords_beg[i] + shape[i] for i in range(len(coords_beg))]
        assert not self.not_inited() and self.is_in_aabb(coords_beg) and self.is_in_aabb(coords_end)
        indices_beg = self.coords2indices(coords_beg)
        indices_end = self.coords2indices(coords_end)
        
        slices = tuple(slice(indices_beg[i], indices_end[i]) for i in range(len(indices_beg)))
        return self.data[slices]

    def not_inited(self):
        is_none = self.alloced_aabb is None
        assert is_none == (self.data is None or self.used_aabb is None)
        return is_none

    def is_in_aabb(self, coords, aabb=None):
        """
        检查坐标是否在aabb范围内
        coords: 坐标元组
        aabb: [dim0_min, dim0_max, dim1_min, dim1_max, ..., dimN_min, dimN_max]
        """
        if self.not_inited():
            return False
        if aabb is None:
            aabb = self.alloced_aabb
        if not isinstance(coords, (list, tuple)):
            coords = [coords]
            
        for i, coord in enumerate(coords):
            min_idx = i * 2
            max_idx = i * 2 + 1
            if not (aabb[min_idx] <= coord < aabb[max_idx]):
                return False
        return True

    def tight_data(self):
        """返回紧凑的数据切片"""
        assert not self.not_inited()
        indices_beg = self.coords2indices(
            [self.used_aabb[i*2] for i in range(self.ndim)]
        )
        indices_end = self.coords2indices(
            [self.used_aabb[i*2+1] for i in range(self.ndim)]
        )
        
        slices = tuple(slice(indices_beg[i], indices_end[i]) for i in range(self.ndim))
        return self.data[slices]
