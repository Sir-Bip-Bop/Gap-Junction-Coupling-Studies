try:
    __PROJECT_IMPORTED__
except NameError:
    __PROJECT_IMPORTED__ = False 

__version__ = '1.0'

if not __PROJECT_IMPORTED__:
    from . import phaseportraits
    from . import models 
    from . import utils 
__PROJECT_IMPORTED__ = True