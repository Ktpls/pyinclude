from .util_windows import *
import traceback
import hashlib


class BulletinApp:
    def __init__(
        self,
        idlebulletincontents=None,
        bulletinoutputpos=None,
        fps=None,
        threadpool=None,
    ):
        def replaceNone(val, default_val):
            return val if val is not None else default_val

        idlebulletincontents = replaceNone(
            idlebulletincontents,
            [
                ["(*≧ω≦)", 66],
                ["(＞д＜)", 30],
                ["($w$)", 1],
                ["ヽ(≧Д≦)ノ", 1],
                ["(￣ω￣;)", 1],
                ["(0v0)", 1],
            ],
        )
        bulletinoutputpos = replaceNone(bulletinoutputpos, (100, 500))

        fps = replaceNone(fps, 10)

        threadpool = replaceNone(threadpool, ThreadPoolExecutor(max_workers=10))

        seed = time.strftime("%Y-%m-%d", time.localtime()).encode("utf-8")
        seed = hashlib.md5(seed).digest()
        seed = int.from_bytes(seed[:8], "big")
        self.bulletin = BulletinBoard(
            idlebulletincontents[
                summonCard(
                    [c[1] for c in idlebulletincontents],
                    np.random.Generator(np.random.PCG64(seed)),
                )
            ][0]
        )
        self.fpsm = FpsManager(fps)

        self.threadpool = threadpool

        self.AsyncLongScript = functools.partial(
            StoppableSomewhat.EasyUse,
            pool=self.threadpool,
            implType=StoppableThread,
            strategy_runonrunning=StoppableSomewhat.StrategyRunOnRunning.stop_and_rerun,
            strategy_error=StoppableSomewhat.StrategyError.print_error,
        )
        self.hud: fullScrHUD = fullScrHUD()

        def swapHKM(newHkm):
            old = self.hkm
            self.hkm = newHkm
            return old

        self.inputSession = HotkeyManager.InputSession(swapHKM, self.bulletin)

        self.business: list = list()
        self.hotkeytask: list = list()
        self.hkm = None
        self.bulletinoutputpos = bulletinoutputpos

    @EasyWrapper
    def Business(foo, self: "BulletinApp"):
        self.business.append(foo)
        return foo

    @EasyWrapper
    def Hotkey(foo, self: "BulletinApp", prompt, key, continiousPress=None):
        print(f"{prompt:<20}{HotkeyManager.hotkeytask.getKeyRepr(key)}")
        self.hotkeytask.append(
            HotkeyManager.hotkeytask(key=key, foo=foo, continiousPress=continiousPress)
        )
        return foo

    def go(self):
        @self.Business()
        def showBulletinAndUpdateHud():
            # show bulletin
            self.hud.writecontent(
                np.flip(self.bulletinoutputpos),
                aPicWithTextWithPil(
                    self.bulletin.read(), maxsize=[400, 700], lineinterval=0
                ),
            )

            self.hud.update()

        self.hkm = HotkeyManager(self.hotkeytask)
        self.hud.setup()
        # activeWindow(self.hud.hwnd)
        
        # main loop
        while True:
            self.fpsm.WaitUntilNextFrame()
            self.hud.clear()

            decideresult = self.hkm.decideAllHotKey()

            try:
                self.hkm.doAllDecidedKey(decideresult, True, True)
            except SystemExit as e:
                raise e
            except Exception as e:
                Rhythms.Error.play()
                print("#" * 10)
                traceback.print_exc()
                print("#" * 10)

            for bus in self.business:
                try:
                    bus()
                except Exception as e:
                    Rhythms.Error.play()
                    print("#" * 10)
                    traceback.print_exc()
                    print("#" * 10)
