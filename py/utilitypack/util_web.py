try:
    from .modules.web.flask_required import *
except ImportError:
    pass
try:
    from .modules.web.sqlalchemy_required import *
except ImportError:
    pass
try:
    from .modules.web.sqlalchemy_jinja_required import *
except ImportError:
    pass
try:
    from .modules.web.pydantic_required import *
except ImportError:
    pass
try:
    from .modules.web.flask_pydantic_required import *
except ImportError:
    pass
try:
    from .modules.web.pydantic_sqlalchemy_required import *
except ImportError:
    pass
