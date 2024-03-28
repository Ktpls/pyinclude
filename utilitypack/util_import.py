import importlib
import sys


def importNoCache(name, package=None):
    if name in sys.modules:
        return importlib.reload(sys.modules.get(name))
    else:
        return importlib.import_module(name, package)
