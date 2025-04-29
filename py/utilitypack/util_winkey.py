import time
from utilitypack.util_solid import *
from utilitypack.util_windows import *


class keycodeWinScanCode:
    key_Esc = 1
    key_1 = 2
    key_2 = 3
    key_3 = 4
    key_4 = 5
    key_5 = 6
    key_6 = 7
    key_7 = 8
    key_8 = 9
    key_9 = 10
    key_0 = 11
    key_Minus = 12
    key_Equals = 13
    key_Backspace = 14
    key_Tab = 15
    key_Q = 16
    key_W = 17
    key_E = 18
    key_R = 19
    key_T = 20
    key_Y = 21
    key_U = 22
    key_I = 23
    key_O = 24
    key_P = 25
    key_LeftBracket = 26
    key_RightBracket = 27
    key_Enter = 28
    key_LeftControl = 29
    key_A = 30
    key_S = 31
    key_D = 32
    key_F = 33
    key_G = 34
    key_H = 35
    key_J = 36
    key_K = 37
    key_L = 38
    key_Semicolon = 39
    key_Apostrophe = 40
    key_Tilde = 41  # ~
    key_LeftShift = 42
    key_BackSlash = 43
    key_Z = 44
    key_X = 45
    key_C = 46
    key_V = 47
    key_B = 48
    key_N = 49
    key_M = 50
    key_Comma = 51
    key_Period = 52
    key_ForwardSlash = 53
    key_RightShift = 54
    key_Numpad = 55  # *
    key_LeftAlt = 56
    key_Spacebar = 57
    key_CapsLock = 58
    key_F1 = 59
    key_F2 = 60
    key_F3 = 61
    key_F4 = 62
    key_F5 = 63
    key_F6 = 64
    key_F7 = 65
    key_F8 = 66
    key_F9 = 67
    key_F10 = 68
    key_NumLock = 69
    key_ScrollLock = 70
    key_Numpad7 = 71
    key_Numpad8 = 72
    key_Numpad9 = 73
    key_NumpadMinus = 74
    key_Numpad4 = 75
    key_Numpad5 = 76
    key_Numpad6 = 77
    key_NumpadPlus = 78
    key_Numpad1 = 79
    key_Numpad2 = 80
    key_Numpad3 = 81
    key_Numpad0 = 82
    key_dot = 83
    key_F11 = 87
    key_F12 = 88
    key_NumpadEnter = 156
    key_RightControl = 157
    key_Numpad /= 181
    key_RightAlt = 184
    key_Home = 199
    key_Up = 200
    key_PageUp = 201
    key_Left = 203
    key_Right = 205
    key_End = 207
    key_Down = 208
    key_PageDown = 209
    key_Insert = 210
    key_Delete = 211
    key_LeftMouseButton = 256
    key_RightMouseButton = 257
    key_MiddleMouse = 258
    key_MouseButton3 = 259
    key_MouseButton4 = 260
    key_MouseButton5 = 261
    key_MouseButton6 = 262
    key_MouseButton7 = 263
    key_MouseWheelUp = 264
    key_MouseWheelDown = 265


class win32conComp:
    VK_LBUTTON = 0x01  # 鼠标左键
    VK_RBUTTON = 0x02  # 鼠标右键
    VK_CANCEL = 0x03  # 控制中断处理
    VK_MBUTTON = 0x04  # 鼠标中键
    VK_XBUTTON1 = 0x05  # X1 鼠标按钮
    VK_XBUTTON2 = 0x06  # X2 鼠标按钮
    # - = 0x07 #保留
    VK_BACK = 0x08  # BACKSPACE 键
    VK_TAB = 0x09  # Tab 键
    # - = 0x0A-0B #预留
    VK_CLEAR = 0x0C  # CLEAR 键
    VK_RETURN = 0x0D  # Enter 键
    # - = 0x0E-0F #未分配
    VK_SHIFT = 0x10  # SHIFT 键
    VK_CONTROL = 0x11  # CTRL 键
    VK_MENU = 0x12  # Alt 键
    VK_PAUSE = 0x13  # PAUSE 键
    VK_CAPITAL = 0x14  # CAPS LOCK 键
    VK_KANA = 0x15  # IME Kana 模式
    VK_HANGUL = 0x15  # IME Hanguel 模式
    VK_IME_ON = 0x16  # IME 打开
    VK_JUNJA = 0x17  # IME Junja 模式
    VK_FINAL = 0x18  # IME 最终模式
    VK_HANJA = 0x19  # IME Hanja 模式
    VK_KANJI = 0x19  # IME Kanji 模式
    VK_IME_OFF = 0x1A  # IME 关闭
    VK_ESCAPE = 0x1B  # ESC 键
    VK_CONVERT = 0x1C  # IME 转换
    VK_NONCONVERT = 0x1D  # IME 不转换
    VK_ACCEPT = 0x1E  # IME 接受
    VK_MODECHANGE = 0x1F  # IME 模式更改请求
    VK_SPACE = 0x20  # 空格键
    VK_PRIOR = 0x21  # PAGE UP 键
    VK_NEXT = 0x22  # PAGE DOWN 键
    VK_END = 0x23  # END 键
    VK_HOME = 0x24  # HOME 键
    VK_LEFT = 0x25  # LEFT ARROW 键
    VK_UP = 0x26  # UP ARROW 键
    VK_RIGHT = 0x27  # RIGHT ARROW 键
    VK_DOWN = 0x28  # DOWN ARROW 键
    VK_SELECT = 0x29  # SELECT 键
    VK_PRINT = 0x2A  # PRINT 键
    VK_EXECUTE = 0x2B  # EXECUTE 键
    VK_SNAPSHOT = 0x2C  # PRINT SCREEN 键
    VK_INSERT = 0x2D  # INS 键
    VK_DELETE = 0x2E  # DEL 键
    VK_HELP = 0x2F  # HELP 键
    VK_0 = 0x30  # 0 键
    VK_1 = 0x31  # 1 个键
    VK_2 = 0x32  # 2 键
    VK_3 = 0x33  # 3 键
    VK_4 = 0x34  # 4 键
    VK_5 = 0x35  # 5 键
    VK_6 = 0x36  # 6 键
    VK_7 = 0x37  # 7 键
    VK_8 = 0x38  # 8 键
    VK_9 = 0x39  # 9 键
    # - = 0x3A-40 #Undefined
    VK_A = 0x41  # A 键
    VK_B = 0x42  # B 键
    VK_C = 0x43  # C 键
    VK_D = 0x44  # D 键
    VK_E = 0x45  # E 键
    VK_F = 0x46  # F 键
    VK_G = 0x47  # G 键
    VK_H = 0x48  # H 键
    VK_I = 0x49  # I 键
    VK_J = 0x4A  # J 键
    VK_K = 0x4B  # K 键
    VK_L = 0x4C  # L 键
    VK_M = 0x4D  # M 键
    VK_N = 0x4E  # N 键
    VK_O = 0x4F  # O 键
    VK_P = 0x50  # P 键
    VK_Q = 0x51  # Q 键
    VK_R = 0x52  # R 键
    VK_S = 0x53  # S 键
    VK_T = 0x54  # T 键
    VK_U = 0x55  # U 键
    VK_V = 0x56  # V 键
    VK_W = 0x57  # W 键
    VK_X = 0x58  # X 键
    VK_Y = 0x59  # Y 键
    VK_Z = 0x5A  # Z 键
    VK_LWIN = 0x5B  # 左 Windows 键
    VK_RWIN = 0x5C  # 右侧 Windows 键
    VK_APPS = 0x5D  # 应用程序密钥
    # - = 0x5E #预留
    VK_SLEEP = 0x5F  # 计算机休眠键
    VK_NUMPAD0 = 0x60  # 数字键盘 0 键
    VK_NUMPAD1 = 0x61  # 数字键盘 1 键
    VK_NUMPAD2 = 0x62  # 数字键盘 2 键
    VK_NUMPAD3 = 0x63  # 数字键盘 3 键
    VK_NUMPAD4 = 0x64  # 数字键盘 4 键
    VK_NUMPAD5 = 0x65  # 数字键盘 5 键
    VK_NUMPAD6 = 0x66  # 数字键盘 6 键
    VK_NUMPAD7 = 0x67  # 数字键盘 7 键
    VK_NUMPAD8 = 0x68  # 数字键盘 8 键
    VK_NUMPAD9 = 0x69  # 数字键盘 9 键
    VK_MULTIPLY = 0x6A  # 乘号键
    VK_ADD = 0x6B  # 加号键
    VK_SEPARATOR = 0x6C  # 分隔符键
    VK_SUBTRACT = 0x6D  # 减号键
    VK_DECIMAL = 0x6E  # 句点键
    VK_DIVIDE = 0x6F  # 除号键
    VK_F1 = 0x70  # F1 键
    VK_F2 = 0x71  # F2 键
    VK_F3 = 0x72  # F3 键
    VK_F4 = 0x73  # F4 键
    VK_F5 = 0x74  # F5 键
    VK_F6 = 0x75  # F6 键
    VK_F7 = 0x76  # F7 键
    VK_F8 = 0x77  # F8 键
    VK_F9 = 0x78  # F9 键
    VK_F10 = 0x79  # F10 键
    VK_F11 = 0x7A  # F11 键
    VK_F12 = 0x7B  # F12 键
    VK_F13 = 0x7C  # F13 键
    VK_F14 = 0x7D  # F14 键
    VK_F15 = 0x7E  # F15 键
    VK_F16 = 0x7F  # F16 键
    VK_F17 = 0x80  # F17 键
    VK_F18 = 0x81  # F18 键
    VK_F19 = 0x82  # F19 键
    VK_F20 = 0x83  # F20 键
    VK_F21 = 0x84  # F21 键
    VK_F22 = 0x85  # F22 键
    VK_F23 = 0x86  # F23 键
    VK_F24 = 0x87  # F24 键
    # - = 0x88-8F #保留
    VK_NUMLOCK = 0x90  # NUM LOCK 键
    VK_SCROLL = 0x91  # SCROLL LOCK 键
    # - = 0x92-96 #OEM 特有
    # - = 0x97-9F #未分配
    VK_LSHIFT = 0xA0  # 左 SHIFT 键
    VK_RSHIFT = 0xA1  # 右 SHIFT 键
    VK_LCONTROL = 0xA2  # 左 Ctrl 键
    VK_RCONTROL = 0xA3  # 右 Ctrl 键
    VK_LMENU = 0xA4  # 左 ALT 键
    VK_RMENU = 0xA5  # 右 ALT 键
    VK_BROWSER_BACK = 0xA6  # 浏览器后退键
    VK_BROWSER_FORWARD = 0xA7  # 浏览器前进键
    VK_BROWSER_REFRESH = 0xA8  # 浏览器刷新键
    VK_BROWSER_STOP = 0xA9  # 浏览器停止键
    VK_BROWSER_SEARCH = 0xAA  # 浏览器搜索键
    VK_BROWSER_FAVORITES = 0xAB  # 浏览器收藏键
    VK_BROWSER_HOME = 0xAC  # 浏览器“开始”和“主页”键
    VK_VOLUME_MUTE = 0xAD  # 静音键
    VK_VOLUME_DOWN = 0xAE  # 音量减小键
    VK_VOLUME_UP = 0xAF  # 音量增加键
    VK_MEDIA_NEXT_TRACK = 0xB0  # 下一曲目键
    VK_MEDIA_PREV_TRACK = 0xB1  # 上一曲目键
    VK_MEDIA_STOP = 0xB2  # 停止媒体键
    VK_MEDIA_PLAY_PAUSE = 0xB3  # 播放/暂停媒体键
    VK_LAUNCH_MAIL = 0xB4  # 启动邮件键
    VK_LAUNCH_MEDIA_SELECT = 0xB5  # 选择媒体键
    VK_LAUNCH_APP1 = 0xB6  # 启动应用程序 1 键
    VK_LAUNCH_APP2 = 0xB7  # 启动应用程序 2 键
    # - = 0xB8-B9 #预留
    VK_OEM_1 = 0xBA  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键;:
    VK_OEM_PLUS = 0xBB  # 对于任何国家/地区，键+
    VK_OEM_COMMA = 0xBC  # 对于任何国家/地区，键,
    VK_OEM_MINUS = 0xBD  # 对于任何国家/地区，键-
    VK_OEM_PERIOD = 0xBE  # 对于任何国家/地区，键.
    VK_OEM_2 = 0xBF  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键/?
    VK_OEM_3 = 0xC0  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键`~
    # - = 0xC1-DA #保留
    VK_OEM_4 = 0xDB  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键[{
    VK_OEM_5 = 0xDC  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键\\|
    VK_OEM_6 = 0xDD  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键]}
    VK_OEM_7 = 0xDE  # 用于杂项字符；它可能因键盘而异。 对于美国标准键盘，键'"
    VK_OEM_8 = 0xDF  # 用于杂项字符；它可能因键盘而异。
    # - = 0xE0 #预留
    # - = 0xE1 #OEM 特有
    VK_OEM_102 = 0xE2  # 美国标准键盘上的 <> 键，或非美国 102 键键盘上的 \\| 键
    # - = 0xE3-E4 #OEM 特有
    VK_PROCESSKEY = 0xE5  # IME PROCESS 键
    # - = 0xE6 #OEM 特有
    VK_PACKET = 0xE7  # 用于将 Unicode 字符当作键击传递。 VK_PACKET 键是用于非键盘输入法的 32 位虚拟键值的低位字。 有关更多信息，请参阅 KEYBDINPUT、SendInput、WM_KEYDOWN 和 WM_KEYUP 中的注释
    # - = 0xE8 #未分配
    # - = 0xE9-F5 #OEM 特有
    VK_ATTN = 0xF6  # Attn 键
    VK_CRSEL = 0xF7  # CrSel 键
    VK_EXSEL = 0xF8  # ExSel 键
    VK_EREOF = 0xF9  # Erase EOF 键
    VK_PLAY = 0xFA  # Play 键
    VK_ZOOM = 0xFB  # Zoom 键
    VK_NONAME = 0xFC  # 预留
    VK_PA1 = 0xFD  # PA1 键
    VK_OEM_CLEAR = 0xFE  # Clear 键

    # oem keys easy to use
    VK_SEMICOLON = VK_OEM_1
    VK_SLASH = VK_OEM_2
    VK_TILDE = VK_OEM_3
    VK_LEFT_MID_BRACKET = VK_OEM_4
    VK_BACK_SLASH = VK_OEM_5
    VK_RIGHT_MID_BRACKET = VK_OEM_6
    VK_QUOTE = VK_OEM_7
    @staticmethod
    def KeyOf(key:str):
        assert len(key)==1
        if key.isalpha():
            return ord(key.upper())
        elif key.isdigit():
            return ord(key)
        raise ValueError(f'Unsupported key: {key}')

virtualKeyCode2ScanCode = {
    win32conComp.VK_LBUTTON: keycodeWinScanCode.key_LeftMouseButton,
    win32conComp.VK_RBUTTON: keycodeWinScanCode.key_RightMouseButton,
    win32conComp.VK_CANCEL: 0,
    win32conComp.VK_MBUTTON: keycodeWinScanCode.key_MiddleMouse,
    win32conComp.VK_XBUTTON1: keycodeWinScanCode.key_MouseButton4,
    win32conComp.VK_XBUTTON2: keycodeWinScanCode.key_MouseButton5,
    win32conComp.VK_BACK: keycodeWinScanCode.key_Backspace,
    win32conComp.VK_TAB: keycodeWinScanCode.key_Tab,
    win32conComp.VK_CLEAR: 0,
    win32conComp.VK_RETURN: keycodeWinScanCode.key_Enter,
    win32conComp.VK_SHIFT: keycodeWinScanCode.key_LeftShift,
    win32conComp.VK_CONTROL: keycodeWinScanCode.key_LeftControl,
    win32conComp.VK_MENU: keycodeWinScanCode.key_LeftAlt,
    win32conComp.VK_PAUSE: 0,
    win32conComp.VK_CAPITAL: keycodeWinScanCode.key_CapsLock,
    win32conComp.VK_KANA: 0,
    win32conComp.VK_HANGUL: 0,
    win32conComp.VK_IME_ON: 0,
    win32conComp.VK_JUNJA: 0,
    win32conComp.VK_FINAL: 0,
    win32conComp.VK_HANJA: 0,
    win32conComp.VK_KANJI: 0,
    win32conComp.VK_IME_OFF: 0,
    win32conComp.VK_ESCAPE: keycodeWinScanCode.key_Esc,
    win32conComp.VK_CONVERT: 0,
    win32conComp.VK_NONCONVERT: 0,
    win32conComp.VK_ACCEPT: 0,
    win32conComp.VK_MODECHANGE: 0,
    win32conComp.VK_SPACE: keycodeWinScanCode.key_Spacebar,
    win32conComp.VK_PRIOR: keycodeWinScanCode.key_PageUp,
    win32conComp.VK_NEXT: keycodeWinScanCode.key_PageDown,
    win32conComp.VK_END: keycodeWinScanCode.key_End,
    win32conComp.VK_HOME: keycodeWinScanCode.key_Home,
    win32conComp.VK_LEFT: keycodeWinScanCode.key_Left,
    win32conComp.VK_UP: keycodeWinScanCode.key_Up,
    win32conComp.VK_RIGHT: keycodeWinScanCode.key_Right,
    win32conComp.VK_DOWN: keycodeWinScanCode.key_Down,
    win32conComp.VK_SELECT: 0,
    win32conComp.VK_PRINT: 0,
    win32conComp.VK_EXECUTE: 0,
    win32conComp.VK_SNAPSHOT: 0,
    win32conComp.VK_INSERT: keycodeWinScanCode.key_Insert,
    win32conComp.VK_DELETE: keycodeWinScanCode.key_Delete,
    win32conComp.VK_HELP: 0,
    win32conComp.VK_0: keycodeWinScanCode.key_0,
    win32conComp.VK_1: keycodeWinScanCode.key_1,
    win32conComp.VK_2: keycodeWinScanCode.key_2,
    win32conComp.VK_3: keycodeWinScanCode.key_3,
    win32conComp.VK_4: keycodeWinScanCode.key_4,
    win32conComp.VK_5: keycodeWinScanCode.key_5,
    win32conComp.VK_6: keycodeWinScanCode.key_6,
    win32conComp.VK_7: keycodeWinScanCode.key_7,
    win32conComp.VK_8: keycodeWinScanCode.key_8,
    win32conComp.VK_9: keycodeWinScanCode.key_9,
    win32conComp.VK_A: keycodeWinScanCode.key_A,
    win32conComp.VK_B: keycodeWinScanCode.key_B,
    win32conComp.VK_C: keycodeWinScanCode.key_C,
    win32conComp.VK_D: keycodeWinScanCode.key_D,
    win32conComp.VK_E: keycodeWinScanCode.key_E,
    win32conComp.VK_F: keycodeWinScanCode.key_F,
    win32conComp.VK_G: keycodeWinScanCode.key_G,
    win32conComp.VK_H: keycodeWinScanCode.key_H,
    win32conComp.VK_I: keycodeWinScanCode.key_I,
    win32conComp.VK_J: keycodeWinScanCode.key_J,
    win32conComp.VK_K: keycodeWinScanCode.key_K,
    win32conComp.VK_L: keycodeWinScanCode.key_L,
    win32conComp.VK_M: keycodeWinScanCode.key_M,
    win32conComp.VK_N: keycodeWinScanCode.key_N,
    win32conComp.VK_O: keycodeWinScanCode.key_O,
    win32conComp.VK_P: keycodeWinScanCode.key_P,
    win32conComp.VK_Q: keycodeWinScanCode.key_Q,
    win32conComp.VK_R: keycodeWinScanCode.key_R,
    win32conComp.VK_S: keycodeWinScanCode.key_S,
    win32conComp.VK_T: keycodeWinScanCode.key_T,
    win32conComp.VK_U: keycodeWinScanCode.key_U,
    win32conComp.VK_V: keycodeWinScanCode.key_V,
    win32conComp.VK_W: keycodeWinScanCode.key_W,
    win32conComp.VK_X: keycodeWinScanCode.key_X,
    win32conComp.VK_Y: keycodeWinScanCode.key_Y,
    win32conComp.VK_Z: keycodeWinScanCode.key_Z,
    win32conComp.VK_LWIN: 0,
    win32conComp.VK_RWIN: 0,
    win32conComp.VK_APPS: 0,
    win32conComp.VK_SLEEP: 0,
    win32conComp.VK_NUMPAD0: keycodeWinScanCode.key_Numpad0,
    win32conComp.VK_NUMPAD1: keycodeWinScanCode.key_Numpad1,
    win32conComp.VK_NUMPAD2: keycodeWinScanCode.key_Numpad2,
    win32conComp.VK_NUMPAD3: keycodeWinScanCode.key_Numpad3,
    win32conComp.VK_NUMPAD4: keycodeWinScanCode.key_Numpad4,
    win32conComp.VK_NUMPAD5: keycodeWinScanCode.key_Numpad5,
    win32conComp.VK_NUMPAD6: keycodeWinScanCode.key_Numpad6,
    win32conComp.VK_NUMPAD7: keycodeWinScanCode.key_Numpad7,
    win32conComp.VK_NUMPAD8: keycodeWinScanCode.key_Numpad8,
    win32conComp.VK_NUMPAD9: keycodeWinScanCode.key_Numpad9,
    win32conComp.VK_MULTIPLY: keycodeWinScanCode.key_Numpad,
    win32conComp.VK_ADD: keycodeWinScanCode.key_NumpadPlus,
    win32conComp.VK_SEPARATOR: 0,
    win32conComp.VK_SUBTRACT: keycodeWinScanCode.key_NumpadMinus,
    win32conComp.VK_DECIMAL: keycodeWinScanCode.key_dot,
    win32conComp.VK_DIVIDE: 0,
    win32conComp.VK_F1: keycodeWinScanCode.key_F1,
    win32conComp.VK_F2: keycodeWinScanCode.key_F2,
    win32conComp.VK_F3: keycodeWinScanCode.key_F3,
    win32conComp.VK_F4: keycodeWinScanCode.key_F4,
    win32conComp.VK_F5: keycodeWinScanCode.key_F5,
    win32conComp.VK_F6: keycodeWinScanCode.key_F6,
    win32conComp.VK_F7: keycodeWinScanCode.key_F7,
    win32conComp.VK_F8: keycodeWinScanCode.key_F8,
    win32conComp.VK_F9: keycodeWinScanCode.key_F9,
    win32conComp.VK_F10: keycodeWinScanCode.key_F10,
    win32conComp.VK_F11: keycodeWinScanCode.key_F11,
    win32conComp.VK_F12: keycodeWinScanCode.key_F12,
    win32conComp.VK_F13: 0,
    win32conComp.VK_F14: 0,
    win32conComp.VK_F15: 0,
    win32conComp.VK_F16: 0,
    win32conComp.VK_F17: 0,
    win32conComp.VK_F18: 0,
    win32conComp.VK_F19: 0,
    win32conComp.VK_F20: 0,
    win32conComp.VK_F21: 0,
    win32conComp.VK_F22: 0,
    win32conComp.VK_F23: 0,
    win32conComp.VK_F24: 0,
    win32conComp.VK_NUMLOCK: keycodeWinScanCode.key_NumLock,
    win32conComp.VK_SCROLL: keycodeWinScanCode.key_ScrollLock,
    win32conComp.VK_LSHIFT: keycodeWinScanCode.key_LeftShift,
    win32conComp.VK_RSHIFT: keycodeWinScanCode.key_RightShift,
    win32conComp.VK_LCONTROL: keycodeWinScanCode.key_LeftControl,
    win32conComp.VK_RCONTROL: keycodeWinScanCode.key_RightControl,
    win32conComp.VK_LMENU: keycodeWinScanCode.key_LeftAlt,
    win32conComp.VK_RMENU: keycodeWinScanCode.key_RightAlt,
    win32conComp.VK_BROWSER_BACK: 0,
    win32conComp.VK_BROWSER_FORWARD: 0,
    win32conComp.VK_BROWSER_REFRESH: 0,
    win32conComp.VK_BROWSER_STOP: 0,
    win32conComp.VK_BROWSER_SEARCH: 0,
    win32conComp.VK_BROWSER_FAVORITES: 0,
    win32conComp.VK_BROWSER_HOME: 0,
    win32conComp.VK_VOLUME_MUTE: 0,
    win32conComp.VK_VOLUME_DOWN: 0,
    win32conComp.VK_VOLUME_UP: 0,
    win32conComp.VK_MEDIA_NEXT_TRACK: 0,
    win32conComp.VK_MEDIA_PREV_TRACK: 0,
    win32conComp.VK_MEDIA_STOP: 0,
    win32conComp.VK_MEDIA_PLAY_PAUSE: 0,
    win32conComp.VK_LAUNCH_MAIL: 0,
    win32conComp.VK_LAUNCH_MEDIA_SELECT: 0,
    win32conComp.VK_LAUNCH_APP1: 0,
    win32conComp.VK_LAUNCH_APP2: 0,
    win32conComp.VK_OEM_1: keycodeWinScanCode.key_Semicolon,
    win32conComp.VK_OEM_PLUS: keycodeWinScanCode.key_Equals,
    win32conComp.VK_OEM_COMMA: keycodeWinScanCode.key_Comma,
    win32conComp.VK_OEM_MINUS: keycodeWinScanCode.key_Minus,
    win32conComp.VK_OEM_PERIOD: keycodeWinScanCode.key_Period,
    win32conComp.VK_OEM_2: keycodeWinScanCode.key_ForwardSlash,
    win32conComp.VK_OEM_3: keycodeWinScanCode.key_Tilde,
    win32conComp.VK_OEM_4: keycodeWinScanCode.key_LeftBracket,
    win32conComp.VK_OEM_5: keycodeWinScanCode.key_BackSlash,
    win32conComp.VK_OEM_6: keycodeWinScanCode.key_RightBracket,
    win32conComp.VK_OEM_7: keycodeWinScanCode.key_Apostrophe,
    win32conComp.VK_OEM_8: 0,
    win32conComp.VK_OEM_102: 0,
    win32conComp.VK_PROCESSKEY: 0,
    win32conComp.VK_PACKET: 0,
    win32conComp.VK_ATTN: 0,
    win32conComp.VK_CRSEL: 0,
    win32conComp.VK_EXSEL: 0,
    win32conComp.VK_EREOF: 0,
    win32conComp.VK_PLAY: 0,
    win32conComp.VK_ZOOM: 0,
    win32conComp.VK_NONAME: 0,
    win32conComp.VK_PA1: 0,
    win32conComp.VK_OEM_CLEAR: 0,
}
import dataclasses
import ctypes

SendInput = windll.user32.SendInput



class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


# Actuals Functions


@Singleton
class Vk2Sk:
    def tr(self, vk):
        return virtualKeyCode2ScanCode.get(vk, 0)


class Keyboard:

    @staticmethod
    def KeyDown(hexKeyCode):
        '''
        use like:
            Keyboard.KeyDown(ord("S"))
        to push key "S" down
        '''
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, Vk2Sk().tr(hexKeyCode), 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    @staticmethod
    def KeyUp(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(
            0, Vk2Sk().tr(hexKeyCode), 0x0008 | 0x0002, 0, ctypes.pointer(extra)
        )
        x = Input(ctypes.c_ulong(1), ii_)
        windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    @staticmethod
    def KeyHold(k, t):
        Keyboard.KeyDown(k)
        PreciseSleep(t)
        Keyboard.KeyUp(k)

    @staticmethod
    def KeyPress(k, interval=0.1):
        Keyboard.KeyHold(k, interval)

    @staticmethod
    def KeyDownDelay(k):
        Keyboard.KeyDown(k)
        PreciseSleep(0.1)

    @staticmethod
    def KeyUpDelay(k):
        Keyboard.KeyUp(k)
        PreciseSleep(0.1)

    @staticmethod
    def KeyPressDelay(k):
        Keyboard.KeyDown(k)
        PreciseSleep(0.1)
        Keyboard.KeyUp(k)
        PreciseSleep(0.1)

    @dataclasses.dataclass
    class HoldingKey:
        key: int

        def __enter__(self):
            Keyboard.KeyDown(self.key)

        def __exit__(self, exc_type, exc_value, traceback):
            Keyboard.KeyUp(self.key)

    class FunctionalKey:
        key: list[int]

        def __init__(self, key: int | list[int]) -> None:
            self.key = NormalizeIterableOrSingleArgToIterable(key)
            assert len(self.key) >= 1

        def hold(self, holdTime):
            for k in self.key:
                Keyboard.KeyDown(k)
            PreciseSleep(holdTime)
            for k in reversed(self.key):
                Keyboard.KeyUp(k)

        def press(self, pressTime=None):
            if pressTime is None:
                pressTime = 0.05
            self.hold(pressTime)


def moveto(p):
    windll.user32.SetCursorPos(int(p[0]), int(p[1]))


import win32con


class mouse:
    __downk2f = {
        0: win32con.MOUSEEVENTF_LEFTDOWN,
        1: win32con.MOUSEEVENTF_RIGHTDOWN,
        2: win32con.MOUSEEVENTF_MIDDLEDOWN,
    }
    __upk2f = {
        0: win32con.MOUSEEVENTF_LEFTUP,
        1: win32con.MOUSEEVENTF_RIGHTUP,
        2: win32con.MOUSEEVENTF_MIDDLEUP,
    }

    @staticmethod
    def __callevent(dwflags, x=0, y=0):
        windll.user32.mouse_event(dwflags, x, y, 0, 0)

    @staticmethod
    def down(key):
        mouse.__callevent(mouse.__downk2f[key])

    @staticmethod
    def up(key):
        mouse.__callevent(mouse.__upk2f[key])

    @staticmethod
    def click(key, interval=0.1):
        mouse.down(key)
        PreciseSleep(interval)
        mouse.up(key)

    @staticmethod
    def mov(x, y):
        mouse.__callevent(
            win32con.MOUSEEVENTF_MOVE + win32con.MOUSEEVENTF_ABSOLUTE, x, y
        )

    @staticmethod
    def movr(x, y):
        mouse.__callevent(win32con.MOUSEEVENTF_MOVE, x, y)


def mouseup():
    windll.user32.mouse_event(4, 0, 0, 0, 0)


def mousedown():
    windll.user32.mouse_event(2, 0, 0, 0, 0)


def click(p):
    moveto(p)
    time.sleep(0.05)
    mousedown()
    time.sleep(0.01)
    mouseup()
    time.sleep(0.1)
