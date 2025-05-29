try:
    import regex
    from .regex_required import *
except ImportError:
    pass

try:
    import aenum
    from .aenum_required import *
except ImportError:
    pass
