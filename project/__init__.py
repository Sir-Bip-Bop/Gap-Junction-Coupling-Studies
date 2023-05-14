try: 
    __PROJECT_PACKAGE_IMPORTED__
except NameError:
    __PROJECT_PACKAGE_IMPORTED__ = False

if not __PROJECT_PACKAGE_IMPORTED__:
    from . import utils 
    from . import models

__PROJECT_PACKAGE_IMPORTED__ = True