from test.autotest_common import *


class AutoSizableNdarrayTest(unittest.TestCase):

    def test_1d_array(self):
        """测试一维数组功能"""
        arr = AutoSizableNdarray(ndim=1)

        # 测试设置和获取单个值
        arr.set([5], 42)
        self.assertEqual(arr.get([5]), 42)

        # 测试自动扩展
        arr.set([10], 100)
        self.assertEqual(arr.get([10]), 100)
        self.assertEqual(arr.get([5]), 42)  # 确保原来的值还在

        # 测试负索引
        arr.set([-5], -10)
        self.assertEqual(arr.get([-5]), -10)

    def test_2d_array(self):
        """测试二维数组功能（原始功能）"""
        arr = AutoSizableNdarray(ndim=2)

        # 测试设置和获取单个值
        arr.set([2, 3], 5)
        self.assertEqual(arr.get([2, 3]), 5)

        # 测试自动扩展
        arr.set([5, 7], 15)
        self.assertEqual(arr.get([5, 7]), 15)
        self.assertEqual(arr.get([2, 3]), 5)  # 确保原来的值还在

        # 测试批量设置
        values = np.array([[1, 2, 3], [4, 5, 6]])
        arr.batch_set([0, 0], values)

        # 验证批量设置的值
        result = arr.batch_get([0, 0], [2, 3])
        np.testing.assert_array_equal(result, values)

    def test_3d_array(self):
        """测试三维数组功能"""
        arr = AutoSizableNdarray(ndim=3)

        # 测试设置和获取单个值
        arr.set([1, 2, 3], 99)
        self.assertEqual(arr.get([1, 2, 3]), 99)

        # 测试自动扩展
        arr.set([5, 10, 15], 55)
        self.assertEqual(arr.get([5, 10, 15]), 55)
        self.assertEqual(arr.get([1, 2, 3]), 99)  # 确保原来的值还在

        # 测试批量设置
        values = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        arr.batch_set([0, 0, 0], values)

        # 验证批量设置的值
        result = arr.batch_get([0, 0, 0], [2, 2, 2])
        np.testing.assert_array_equal(result, values)

    def test_4d_array(self):
        """测试四维数组功能"""
        arr = AutoSizableNdarray(ndim=4)

        # 测试设置和获取单个值
        arr.set([1, 2, 3, 4], 1234)
        self.assertEqual(arr.get([1, 2, 3, 4]), 1234)

        # 测试自动扩展
        arr.set([2, 4, 6, 8], 2468)
        self.assertEqual(arr.get([2, 4, 6, 8]), 2468)
        self.assertEqual(arr.get([1, 2, 3, 4]), 1234)  # 确保原来的值还在

    def test_tight_data(self):
        """测试tight_data方法"""
        arr = AutoSizableNdarray(ndim=2)

        # 设置一些值
        arr.set([5, 5], 1)
        arr.set([7, 8], 2)

        # 获取紧凑数据
        tight = arr.tight_data()

        # 验证形状和内容
        self.assertEqual(tight.shape, (3, 4))  # 从(5,5)到(7,8)的范围，即3行4列
        self.assertEqual(tight[0, 0], 1)  # 对应坐标(5,5)
        self.assertEqual(tight[2, 3], 2)  # 对应坐标(7,8)

    def test_different_data_types(self):
        """测试不同的数据类型"""
        # 测试浮点数
        arr = AutoSizableNdarray(ndim=2)
        arr.set([0, 0], 3.14)
        self.assertEqual(arr.get([0, 0]), 3.14)

        # 测试字符串（对象数组）
        arr2 = AutoSizableNdarray(ndim=1)
        arr2.set([0], "test")
        self.assertEqual(arr2.get([0]), "test")

    def test_edge_cases(self):
        """测试边界情况"""
        arr = AutoSizableNdarray(ndim=2)

        # 测试零值
        arr.set([0, 0], 0)
        self.assertEqual(arr.get([0, 0]), 0)

        # 测试负坐标
        arr.set([-1, -1], -5)
        self.assertEqual(arr.get([-1, -1]), -5)

        # 测试大坐标值
        arr.set([1000, 2000], 100)
        self.assertEqual(arr.get([1000, 2000]), 100)
