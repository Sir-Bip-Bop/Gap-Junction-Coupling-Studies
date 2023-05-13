try:
    __PHASE_IMPORTED__
except NameError:
    __PHASE_IMPORTED__ = False 

if not __PHASE_IMPORTED__:
    from . import generatedata

__PHASE_IMPORTED__ = True