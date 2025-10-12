import typing
import dataclasses
import typing
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt


class RandVarDistr:
    def __init__(self, x: np.ndarray, f: np.ndarray):
        # 计算累积积分 F = ∫ f dt，使用 cumulative_trapezoid
        # 注意：cumulative_trapezoid 返回的数组比输入少一个点，除非指定 initial=0
        F = scipy.integrate.cumulative_trapezoid(f, x, initial=0)

        # 归一化得到概率密度函数 f
        # 即使f已经归一化过，也并非长尾分布函数而x未完美取遍定义域，F也仍有可能因为数值误差而不归一。将它们手动归一化
        f_tot = F[-1]
        F /= f_tot
        f = f / f_tot
        self.F = F
        self.f = f

        # 构建插值函数 distr_tr，使得 distr_tr(F) = t
        # 即 x'=F, y'=x，插值得到 x 作为 F 的函数
        self.distr_tr = scipy.interpolate.interp1d(
            F, x, kind="linear", bounds_error=False, fill_value=(x[0], x[-1])
        )

    def __call__(self, size: int | typing.Sequence[int]) -> np.ndarray:
        # 从 Uniform(0,1) 采样
        u_samples = np.random.uniform(0, 1, size)
        # 应用变换得到目标分布的样本
        return self.distr_tr(u_samples)

    @staticmethod
    def demo():

        # 假设 t 和 var 是你已有的 NumPy 数组
        # t: 自变量（例如时间或空间坐标）
        # var: 被积函数的值（例如概率密度）
        t = np.linspace(-3, 3, 1000)
        var = np.exp(-(t**2))
        distr = RandVarDistr(t, var)
        t_samples = distr(100000)

        # 绘图：直方图 + 真实密度
        plt.figure(figsize=(8, 5))
        # 直方图（归一化为密度）
        plt.hist(
            t_samples,
            bins=50,
            density=True,
            alpha=0.6,
            label="Sampled distribution (via distr_tr)",
        )

        # 真实概率密度（归一化后的 f）
        plt.plot(t, distr.f, "r-", linewidth=2, label="True PDF (normalized var)")

        plt.xlabel("t")
        plt.ylabel("Density")
        plt.title("Distribution of distr_tr(x) where x ~ Uniform(0,1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
