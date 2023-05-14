try:
    __PAIRS_IMPORTED__
except NameError:
    __PAIRS_IMPORTED__ = False 

if not __PAIRS_IMPORTED__:
    from . import utils_pairs
    from . import models_pairs

__PAIRS_IMPORTED__ = True