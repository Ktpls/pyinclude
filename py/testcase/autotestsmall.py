import unittest
from utilitypack.cold.util_solid import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *

"""
serves for a light enviroment for writing auto test
"""


# copied from autotest.py
class RedirectedPrint:
    def clearPrinted(self):
        self.msg = list()

    def __init__(self):
        self.clearPrinted()

    def print(self, m):
        self.msg.append(m)

    def getPrinted(self):
        return self.msg


unittest.main()
