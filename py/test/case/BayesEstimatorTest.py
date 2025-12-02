from test.autotest_common import *


class BayesEstimatorTest(unittest.TestCase):

    def test_basic_functionality(self):
        """测试BayesEstimator基本功能"""

        # 创建一个简单的似然函数：正态分布
        def normal_distribution_model(x_vals, measured_vals):
            # x_vals: (n, 1) - 参数空间值
            # measured_vals: (1, m) - 测量值
            # 返回: (n, m) - P(测量值|参数值)
            return np.exp(-0.5 * ((x_vals - measured_vals) ** 2))

        # 定义参数空间
        xspace = np.linspace(0, 10, 100)

        # 创建BayesEstimator实例
        estimator = BayesEstimator(
            xspace=xspace, distributionModel=normal_distribution_model
        )

        # 检查初始状态
        initial_possibility = estimator.getPossibility()
        expected_uniform = np.ones_like(xspace) / len(xspace)
        np.testing.assert_array_almost_equal(initial_possibility, expected_uniform)

        # 添加一些测量值
        estimator.update(5.0)
        possibility_after_first = estimator.getPossibility()

        # 检查概率分布总和为1
        self.assertAlmostEqual(np.sum(possibility_after_first), 1.0)

        # 添加更多测量值
        estimator.update([4.5, 5.5])
        possibility_after_second = estimator.getPossibility()

        # 再次检查概率分布总和为1
        self.assertAlmostEqual(np.sum(possibility_after_second), 1.0)

        # 检查更新后的分布与初始分布不同
        self.assertFalse(
            np.array_equal(possibility_after_first, possibility_after_second)
        )

        # 检查峰值应该在测量值附近(大约在5.0左右)
        max_index = np.argmax(possibility_after_second)
        peak_value = xspace[max_index]
        self.assertTrue(4.0 <= peak_value <= 6.0)

    def test_multiple_updates(self):
        """测试多次更新的一致性"""

        def gaussian_model(x_vals, measured_vals):
            return np.exp(-0.5 * ((x_vals - measured_vals) ** 2) / (0.5**2))

        xspace = np.linspace(-5, 5, 100)
        estimator = BayesEstimator(xspace=xspace, distributionModel=gaussian_model)

        # 进行多次更新
        measurements = [0.0, 0.1, -0.1, 0.05, -0.05]
        for measurement in measurements:
            estimator.update(measurement)

        final_possibility = estimator.getPossibility()

        # 检查最终分布峰值应在0附近
        max_index = np.argmax(final_possibility)
        peak_value = xspace[max_index]
        self.assertTrue(-1.0 <= peak_value <= 1.0)

        # 检查分布仍然是有效的概率分布（非负且总和为1）
        self.assertTrue(np.all(final_possibility >= 0))
        self.assertAlmostEqual(np.sum(final_possibility), 1.0)

    def test_edge_cases(self):
        """测试边缘情况"""

        def simple_model(x_vals, measured_vals):
            # 简单模型，对所有输入返回相同值
            return np.ones((x_vals.shape[0], measured_vals.shape[1]))

        xspace = np.array([1, 2, 3, 4, 5])
        estimator = BayesEstimator(xspace=xspace, distributionModel=simple_model)

        # 使用空列表更新
        estimator.update([])
        possibility = estimator.getPossibility()
        # 应该保持均匀分布
        expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_array_almost_equal(possibility, expected)

        # 使用单个值更新
        estimator.update(3.0)
        possibility = estimator.getPossibility()
        self.assertAlmostEqual(np.sum(possibility), 1.0)
