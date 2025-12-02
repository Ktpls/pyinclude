import unittest
import pprint
from utilitypack.util_solid import *
from utilitypack.cold.util_solid import *
from utilitypack.util_np import *
from utilitypack.util_cracked import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *
from utilitypack.util_web import *

"""
serves for a light enviroment for writing auto test
"""


class RedirectedPrint:
    def clearPrinted(self):
        self.msg = list()

    def __init__(self):
        self.clearPrinted()

    def print(self, m):
        self.msg.append(m)

    def getPrinted(self):
        return self.msg
