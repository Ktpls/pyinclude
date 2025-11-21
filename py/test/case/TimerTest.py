from test.autotest_common import *


class TimerTest(unittest.TestCase):
    # perf_statistic untested
    def __init__(self, *arg, **kwarg):
        unittest.TestCase.__init__(self, *arg, **kwarg)
        self.t = 0

    def initSingleSectionedTimer(self):
        class StubbedTimer(SingleSectionedTimer):
            def timeCounter(self2):
                return self.t

        return StubbedTimer

    def initPerfStatistic(self):

        class StubbedPs(perf_statistic):
            def __init__(self2, startnow=False):
                super().__init__(startnow=False)
                self2._singled = self.initSingleSectionedTimer()
                self2.clear()
                if startnow:
                    self2.start()

        return StubbedPs

    def test_ssdStartGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)

    def test_ssdStartGetClearGetGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)
        self.t = 2
        self.assertEqual(ssd.clear().get(), 0)
        ssd.start()
        self.t = 3
        self.assertEqual(ssd.get(), 1)

    def test_ssdStartGetStartGet(self):
        stubed = self.initSingleSectionedTimer()
        ssd = stubed().start()
        self.t = 1
        self.assertEqual(ssd.get(), 1)
        self.t = 2
        ssd.start()
        self.t = 3
        self.assertEqual(ssd.get(), 1)
