from utilitypack.util_solid import *
from utilitypack.cold.util_solid import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *
from utilitypack.util_wt import *


def ExpParserBenchMark():
    var = {**expparser.BasicConstantLib}
    func = {**expparser.BasicFunctionLib}
    exps = ["1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20"]
    exps = [expparser.compile(e) for e in exps]
    turnNum = 100_000
    pg = Progress(turnNum)
    ps = perf_statistic().start()
    for t in range(turnNum):
        for e in exps:
            result = e.eval(var=var, func=func)
        pg.update(t)
    ps.stop()
    pg.setFinish()
    print(ps.time() / turnNum)

def Port8111BenchMark():
    turnNum = 1000
    pg = Progress(turnNum)
    ps = perf_statistic().start()
    for i in range(turnNum):
        obj = Port8111.get(Port8111.QueryType.indicator)
        ps.countcycle()
        pg.update(i)
    ps.stop()
    print(f"{ps.aveTime()=}")
