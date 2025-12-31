from .util_solid import (
    Coalesce,
    Switch,
    Deduplicate,
    ArrayFlatten,
    FpsManager,
    EasyWrapper,
    StoppableThread,
    StoppableSomewhat,
    perf_statistic,
)
from .util_windows import TranslateHotKey, isKBDown, Rhythms, fullScrHUD
from .util_np import summonCard, NpGeneratorFromStrSeed
from .util_ocv import aPicWithTextWithPil
import traceback
import dataclasses
import typing
import enum
import concurrent.futures
import time
import win32con
import functools


class BulletinBoard:
    @dataclasses.dataclass
    class Poster:
        content: str
        starttime: float
        timeout: float
        overduetime: float

        def __init__(self, content, timeout=10):
            self.content = content
            self.starttime = time.perf_counter()
            self.timeout = timeout
            self.overduetime = time.perf_counter() + timeout

    def __init__(self, idlecontent):
        self.idlecontent = idlecontent
        self.content: list["BulletinBoard.Poster"] = []

    def putup(self, poster: typing.Union[Poster, str]):
        if type(poster) == str:
            poster = BulletinBoard.Poster(poster)
        self.content.append(poster)

    def read(self):
        self.content = [c for c in self.content if c.overduetime > time.perf_counter()]
        rctt = list(range(len(self.content)))
        rctt.reverse()
        if len(self.content):
            return ("\n" + "-" * 10 + "\n").join(
                [self.content[c].content for c in rctt]
            )
        else:
            return self.idlecontent


class HotkeyManager:
    """
    to piorer ctrl+c than c
        if hotkeys ctrl+c and c are both binded,
        on ctrl+c presented, only respond ctrl+c, and skip c
        that is, match ctrl+c priorly than c
        if match successed, skip lower priority keys
    key is given by win32con.VK_*,
        for letters, use ord([letter's UPPER case])
        for numbers, use ord([number])
    issue:
    1. ctrl+a,ctrl+a and ctrl+a,ctrl+s
        fastly press ctrl+a, and ctrl+s, could cause slightly overlap whose key state is [ctrl,a,s] physically
        that is, [ctrl,a]->[ctrl,a,s]->[ctrl,s]
        so at the second state, both [ctrl+a,ctrl+a] and [ctrl+a,ctrl+s] are triggered
        not severe problem
    """

    @dataclasses.dataclass
    class hotkeytask:
        key: list[int]
        onKeyDown: typing.Callable[[], None] = None
        onKeyUp: typing.Callable[[], None] = None
        onKeyPress: typing.Callable[[], None] = None

        # inner field
        _switch: Switch = dataclasses.field(init=False, default=None)

        def __post_init__(self) -> None:
            self.key = HotkeyManager.hotkeytask.formalize_key_param(self.key)
            # self.key is like [keyset1=[key1, key2], keyset2=[key3, key4]]
            self._switch = Switch(
                onSetOn=self.onKeyDown, onSetOff=self.onKeyUp, value=False
            )

        @staticmethod
        def formalize_key_param(key):
            if not isinstance(key, typing.Iterable):
                key = [key]
            else:
                key = key
            return key

        @staticmethod
        def getKeyRepr(key):
            key = HotkeyManager.hotkeytask.formalize_key_param(key)
            return " + ".join([TranslateHotKey()(k) for k in key])

    @dataclasses.dataclass
    class Key:
        code: int

        def GetKeyDown(self):
            return isKBDown(self.code)

    def __init__(self, hktl: list[hotkeytask]):
        keyconcerned = Deduplicate(ArrayFlatten([hkt.key for hkt in hktl]))
        self.kc = [HotkeyManager.Key(k) for k in keyconcerned]
        self.hktl = hktl
        self._calcPriorInfo()

        # clear all previous state
        self._getKeyConcernedState()

    def _calcPriorInfo(self):
        """
        costly!!!
        at m^2n^2, where m is #hotkeytask, n is #key of hotkeytask
        """

        def piorered(a: HotkeyManager.hotkeytask, b: HotkeyManager.hotkeytask):
            def include(a: HotkeyManager.hotkeytask, b: HotkeyManager.hotkeytask):
                for k in b.key:
                    if k not in a.key:
                        return False
                return True

            # a>b and b<a, not equal
            return include(a, b) and not include(b, a)

        self.piorinfo = [
            [
                aidx
                for aidx, a in enumerate(self.hktl)
                if aidx != bidx and piorered(a, b)
            ]
            for bidx, b in enumerate(self.hktl)
        ]

    def keyState2HotkeyState(self, keystate) -> list[bool]:

        class respondstate(enum.Enum):
            false = 0
            true = 1
            unknown = 2

        respondtable = [respondstate.unknown for hk in self.hktl]

        def decideRespondState(i: int):
            # checked
            if respondtable[i] != respondstate.unknown:
                return

            # all key pressed
            if all([keystate[k] for k in self.hktl[i].key]):
                # didnt check piored, check it
                [
                    decideRespondState(p)
                    for p in self.piorinfo[i]
                    if respondtable[p] == respondstate.unknown
                ]

                # no piored responded
                if all(
                    [respondtable[p] == respondstate.false for p in self.piorinfo[i]]
                ):
                    respondtable[i] = respondstate.true
                else:
                    respondtable[i] = respondstate.false
            else:
                # not respond this
                respondtable[i] = respondstate.false

        for hkidx, hk in enumerate(self.hktl):
            decideRespondState(hkidx)

        assert all([rt != respondstate.unknown for rt in respondtable])
        return [
            respondtable[hkidx] == respondstate.true
            for hkidx, hk in enumerate(self.hktl)
        ]

    def dispatchMessage(self):
        keystate = self._getKeyConcernedState()
        curHotkeyState = self.keyState2HotkeyState(keystate)
        for i, s in enumerate(curHotkeyState):
            self.hktl[i]._switch.setTo(s)
            if s and self.hktl[i].onKeyPress:
                self.hktl[i].onKeyPress()

    def _getKeyConcernedState(self):
        return {k.code: k.GetKeyDown() for k in self.kc}

    @dataclasses.dataclass
    class InputSession:
        @dataclasses.dataclass
        class SessionInstance:
            class SessionEndType(enum.Enum):
                UNSPECIFIED = 0
                OK = 1
                CANCEL = 2

            FooSessionDoneCallback: typing.Callable[
                [typing.Type["HotkeyManager.InputSession.SessionInstance"]], None
            ] = None
            content: str = ""
            sessionEndType: SessionEndType = SessionEndType.UNSPECIFIED

            def append(self, extraContent: str):
                self.content += extraContent

            def backSpace(self):
                self.content = self.content[:-1]

            def putup(self, bulletin: BulletinBoard):
                bulletin.putup(self.content)

        # foo that sets hkm and returns older hkm
        FooSwapHKM: typing.Callable[["HotkeyManager"], "HotkeyManager"]
        bulletin: BulletinBoard
        RunningSessionInstance: SessionInstance = dataclasses.field(
            default_factory=SessionInstance
        )
        hotkeymanagerStack: list["HotkeyManager"] = dataclasses.field(
            default_factory=list
        )

        class InputTypeEnabled(enum.Enum):
            NUMBER = 0
            LETTER = 1

        def _GetHotkeyReg(self, ite: list[InputTypeEnabled]):
            @dataclasses.dataclass
            class KeyMapping:
                char: str
                key: typing.Tuple[int]

            AllKeyMapping = [
                *(
                    [KeyMapping(k, (ord(k),)) for i, k in enumerate("0123456789")]
                    if HotkeyManager.InputSession.InputTypeEnabled.NUMBER in ite
                    else []
                ),
                *(
                    [
                        *[
                            KeyMapping(k, (ord(k.upper()),))
                            for i, k in enumerate("abcdefghijklmnopqrstuvwxyz")
                        ],
                        *[
                            KeyMapping(k, (win32con.VK_SHIFT, ord(k.upper())))
                            for i, k in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        ],
                    ]
                    if HotkeyManager.InputSession.InputTypeEnabled.LETTER in ite
                    else []
                ),
            ]
            KeyIndexed = {k.key: k.char for k in AllKeyMapping}

            def procKey(k):
                self.RunningSessionInstance.append(KeyIndexed.get(k, "?"))
                self.RunningSessionInstance.putup(self.bulletin)

            HotkeyReg = [
                HotkeyManager.hotkeytask(
                    key=[k.key], foo=functools.partial(procKey, k.key)
                )
                for k in AllKeyMapping
            ]

            def backSpace():
                self.RunningSessionInstance.backSpace()
                self.RunningSessionInstance.putup(self.bulletin)

            HotkeyReg.append(
                HotkeyManager.hotkeytask(
                    key=[win32con.VK_BACK],
                    foo=backSpace,
                )
            )

            def OutFromSession(
                endType: HotkeyManager.InputSession.SessionInstance.SessionEndType,
            ):
                self.RunningSessionInstance.sessionEndType = endType
                self.RunningSessionInstance.FooSessionDoneCallback(
                    self.RunningSessionInstance
                )
                old = self.hotkeymanagerStack.pop()
                inputer = self.FooSwapHKM(old)

            HotkeyReg.extend(
                [
                    HotkeyManager.hotkeytask(
                        key=[win32con.VK_ESCAPE],
                        foo=functools.partial(
                            OutFromSession,
                            HotkeyManager.InputSession.SessionInstance.SessionEndType.CANCEL,
                        ),
                    ),
                    HotkeyManager.hotkeytask(
                        key=[win32con.VK_RETURN],
                        foo=functools.partial(
                            OutFromSession,
                            HotkeyManager.InputSession.SessionInstance.SessionEndType.OK,
                        ),
                    ),
                ]
            )

            return HotkeyReg

        def IntoSession(
            self, callback, allowedInputType: list[InputTypeEnabled] = None
        ):
            self.RunningSessionInstance = HotkeyManager.InputSession.SessionInstance(
                callback
            )
            if not allowedInputType:
                allowedInputType = [
                    HotkeyManager.InputSession.InputTypeEnabled.NUMBER,
                    HotkeyManager.InputSession.InputTypeEnabled.LETTER,
                ]
            inputer = HotkeyManager(self._GetHotkeyReg(allowedInputType))
            old = self.FooSwapHKM(inputer)
            self.hotkeymanagerStack.append(old)


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

        threadpool = Coalesce(
            threadpool, concurrent.futures.ThreadPoolExecutor(max_workers=10)
        )
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

        self.threadpool: concurrent.futures.ThreadPoolExecutor = threadpool

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
                self.hkm.dispatchMessage()
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
