import importlib
import sys


def import_or_reload(module_name):
    module = sys.modules.get(module_name)
    try:
        if module is None:
            module = importlib.import_module(module_name)
        else:
            module = importlib.reload(module)
    except ImportError:
        module = None

    return module
