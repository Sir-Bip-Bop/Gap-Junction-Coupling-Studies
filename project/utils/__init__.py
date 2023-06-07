try: 
    __PROJECT_UTILS_IMPORTED__
except NameError:
    __PROJECT_UTILS_IMPORTED__ = False

if not __PROJECT_UTILS_IMPORTED__:
    from .compue_chi import compute_chi
    from .compute_r import compute_Reliability
    from .compute_van_Rossum_distance import compute_van_Rossum_distance
    from .compute_volt_var import compute_volt_var
    from .connectivity import create_matrix
    from .firing_rate import compute_firing_rate
    from .load import load_HH, load_ISH, load_LIF, load_ML
    from .phases import phases

__PROJECT_UTILS_IMPORTED__ = True