from utilitypack.util_windows import *
from utilitypack.util_winkey import *

hkm = HotkeyManager(
    [
        HotkeyManager.hotkeytask(
            [win32conComp.VK_CONTROL, win32conComp.VK_MENU, win32conComp.KeyOf("C")],
            lambda: print("Ctrl+Alt+C"),
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
