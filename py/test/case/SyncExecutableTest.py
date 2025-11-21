from test.autotest_common import *


class SyncExecutableTest(unittest.TestCase):
    @dataclasses.dataclass
    class _testbed(Stage):
        # _testbed, for make it clear to unittest lib that this is not a test case(not startswith('test'))
        eosm: SyncExecutableManager
        t: float = 0

        def step(self, dt):
            self.t += dt
            self.eosm.step()

    def test_basicUsage(selfTest):

        class ScriptTest(SyncExecutable):
            def main(self):
                def sleep_specified_time(t):
                    t0 = self.sem.stage.t
                    self.sleep(t)
                    selfTest.assertTrue(self.sem.stage.t - t0 >= t)

                sleep_specified_time(3)
                sleep_specified_time(5)
                sleep_specified_time(10)

        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        script = ScriptTest(eosm).run()
        while True:
            stage.step(1)
            if script.state == SyncExecutable.STATE.stopped:
                break

    class ConsumerProducer:

        @dataclasses.dataclass
        class Production:
            value: int

            def consume(self):
                self.value -= 1

            def produce(self):
                self.value += 1

            def isEmpty(self):
                return self.value == 0

            def isConsumable(self):
                return not self.isEmpty()

        class MainScript(SyncExecutable):
            # consumes production by 1, limited times
            def main(
                self,
                production: "SyncExecutableTest.ConsumerProducer.Production",
                times: int,
            ):
                for i in range(times):
                    self.sleepUntil(lambda: production.isConsumable())
                    production.consume()

        class ProducerScript(SyncExecutable):
            # produce production by 1 if main script is alive
            def main(
                self,
                production: "SyncExecutableTest.ConsumerProducer.Production",
                mainScript: "SyncExecutableTest.ConsumerProducer.MainScript",
            ):
                while True:
                    self.sleepUntil(
                        lambda: production.isEmpty()
                        or mainScript.state == SyncExecutable.STATE.stopped
                    )
                    if mainScript.state == SyncExecutable.STATE.stopped:
                        break
                    production.produce()

    def test_multiScriptFlow(selfTest):
        production = SyncExecutableTest.ConsumerProducer.Production(0)
        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        ms = SyncExecutableTest.ConsumerProducer.MainScript(eosm).run(production, 5)
        ps = SyncExecutableTest.ConsumerProducer.ProducerScript(eosm).run(
            production, ms
        )
        while True:
            stage.step(1)
            if (
                ms.state == SyncExecutable.STATE.stopped
                and ps.state == SyncExecutable.STATE.stopped
            ):
                break

    def test_LaunchThreadInThread(selfTest):
        pool = concurrent.futures.ThreadPoolExecutor()
        stage = SyncExecutableTest._testbed(eosm=None)
        eosm = SyncExecutableManager(pool=pool, stage=stage)
        stage.eosm = eosm
        records = list()

        class MainScriptLauchingThreadFromInside(SyncExecutable):
            def main(self, iteration: int):
                recorderStopSignal = False
                value = 0

                class RecorderScript(SyncExecutable):

                    def main(selfr):

                        def record():
                            if len(records) == 0 or records[-1] != value:
                                records.append(value)

                        while True:
                            if recorderStopSignal:
                                break
                            record()
                            selfr.stepOneFrame()

                recorder = RecorderScript(eosm).run()
                for i in range(iteration):
                    value = i
                    self.sleep(1)
                recorderStopSignal = True
                self.stepOneFrame()

        ms = MainScriptLauchingThreadFromInside(eosm).run(5)
        while True:
            stage.step(0.1)
            if ms.state == SyncExecutable.STATE.stopped:
                break
        pass
