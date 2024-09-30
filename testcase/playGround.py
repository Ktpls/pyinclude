# %%
from utilitypack.util_import import *
from utilitypack.util_solid import *
from utilitypack.cold.util_solid import *
from utilitypack.util_windows import *
from utilitypack.util_winkey import *


def HotkeyManagerDemo():
    hkm = HotkeyManager(
        [
            HotkeyManager.hotkeytask(
                [
                    win32conComp.VK_CONTROL,
                    win32conComp.VK_MENU,
                    win32conComp.KeyOf("C"),
                ],
                lambda: print("Ctrl+Alt+C"),
                onKeyPress=lambda: print("Ctrl+Alt+C press"),
            ),
            HotkeyManager.hotkeytask(
                [win32conComp.VK_CONTROL, win32conComp.KeyOf("C")],
                lambda: print("Ctrl+C"),
            ),
        ]
    )
    fpsm = FpsManager()
    while True:
        fpsm.WaitUntilNextFrame()
        hkm.dispatchMessage()
