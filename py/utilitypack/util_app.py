from .util_windows import *
import traceback


@dataclasses.dataclass
class TaskScheduleing:
    task: typing.Callable
    period: float = None
    lasttime: float = None

    class State(enum.Enum):
        RUNNING = 1
        STOPPED = 2

    state: State = State.RUNNING

    def stop(self):
        self.state = TaskScheduleing.State.STOPPED
        self.lasttime = None

    def start(self):
        self.state = TaskScheduleing.State.RUNNING

    def check(self, nowtime):
        if self.state != TaskScheduleing.State.RUNNING:
            return
        if (
            self.period is not None
            and self.lasttime is not None
            and nowtime - self.lasttime < self.period
        ):
            return
        self.lasttime = nowtime
        self.task()


class BulletinApp:
    def __init__(
        self,
        idlebulletincontents=None,
        bulletinoutputpos=None,
        fps=None,
        threadpool=None,
        hudFps=None,
        bulletinFps=None,
    ):
        idlebulletincontents = Coalesce(
            idlebulletincontents,
            [
                ["(*≧ω≦)", 66],
                ["(＞д＜)", 30],
                ["($w$)", 1],
                ["ヽ(≧Д≦)ノ", 1],
                ["(￣ω￣;)", 1],
                ["(OvO)", 1],
            ],
        )
        bulletinoutputpos = Coalesce(bulletinoutputpos, (100, 500))

        fps = Coalesce(fps, 10)

        threadpool = Coalesce(threadpool, futures.ThreadPoolExecutor(max_workers=10))
        self.hudFps = Coalesce(hudFps, 10)
        self.bulletinFps = Coalesce(bulletinFps, 10)

        self.bulletin = BulletinBoard(
            idlebulletincontents[
                summonCard(
                    [c[1] for c in idlebulletincontents],
                    NpGeneratorFromStrSeed(time.strftime("%Y-%m-%d", time.localtime())),
                )
            ][0]
        )
        self.fpsm = FpsManager(fps)

        self.threadpool: futures.ThreadPoolExecutor = threadpool

        self.hud: fullScrHUD = fullScrHUD()

        def swapHKM(newHkm):
            old = self.hkm
            self.hkm = newHkm
            return old

        self.inputSession = HotkeyManager.InputSession(swapHKM, self.bulletin)

        self.business: list[TaskScheduleing] = list()
        self.hotkeytask: list[HotkeyManager.hotkeytask] = list()
        self.hkm = None
        self.bulletinoutputpos = bulletinoutputpos

    @EasyWrapper
    def Async(
        foo,
        self: "BulletinApp",
        strategy_runonrunning=None,
        strategy_error=None,
    ):
        pool = self.threadpool
        implType = StoppableThread
        strategy_runonrunning = (
            strategy_runonrunning
            or StoppableSomewhat.StrategyRunOnRunning.stop_and_rerun
        )
        strategy_error = strategy_error or StoppableSomewhat.StrategyError.print_error

        return StoppableSomewhat.EasyUse(
            pool=pool,
            implType=implType,
            strategy_runonrunning=strategy_runonrunning,
            strategy_error=strategy_error,
        )(foo)

    @EasyWrapper
    def Business(foo, self: "BulletinApp", period=None):
        self.business.append(TaskScheduleing(task=foo, period=period))
        return foo

    def HotkeyFullFunction(self: "BulletinApp", name, key, *args, **kwargs):
        print(f"{name:<20}{HotkeyManager.hotkeytask.getKeyRepr(key)}")
        hkt = HotkeyManager.hotkeytask(key=key, *args, **kwargs)
        self.hotkeytask.append(hkt)
        return Coalesce(hkt.onKeyDown, hkt.onKeyPress, hkt.onKeyUp)

    @EasyWrapper
    def Hotkey(foo, self: "BulletinApp", name, key):
        self.HotkeyFullFunction(name, key, onKeyDown=foo)
        return foo

    def go(self):
        bulletinMaxSize = [400, 700]
        bulletinRegion = self.hud.addRegion(fullScrHUD.Region(self.bulletinoutputpos))

        @self.Business(period=1 / self.bulletinFps)
        def showBulletin():
            bulletinRegion.content = aPicWithTextWithPil(
                self.bulletin.read(), bulletinMaxSize, lineinterval=0
            )

        @self.Business(period=1 / self.hudFps)
        def UpdateHud():
            self.hud.update()

        self.hkm = HotkeyManager(self.hotkeytask)
        self.hud.setup()
        timer = perf_statistic().start()
        # activeWindow(self.hud.hwnd)

        # main loop
        while True:
            self.fpsm.WaitUntilNextFrame()
            try:
                self.hkm.dispatchMessage(True, True)
            except SystemExit as e:
                raise e
            except Exception as e:
                Rhythms.Error.play()
                print("#" * 10)
                traceback.print_exc()
                print("#" * 10)

            for bus in self.business:
                try:
                    bus.check(timer.time())
                except Exception as e:
                    Rhythms.Error.play()
                    print("#" * 10)
                    traceback.print_exc()
                    print("#" * 10)
