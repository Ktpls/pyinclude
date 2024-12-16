import pandas as pd
import math


def load_xls_to_dict(file_path):
    """
    加载XLS文件，并将每一行映射为字典。

    :param file_path: XLS文件路径
    :return: 包含每一行数据的字典列表
    """
    # 读取XLS文件
    df = pd.read_excel(file_path)
    # 将DataFrame转换为字典列表
    dict_list = df.to_dict(orient="records")
    # 替换默认的float('NaN')为None
    for d in dict_list:
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
    return dict_list
