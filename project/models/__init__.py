try:
    __PROJECT_MODELS_IMPORTED__
except NameError:
    __PROJECT_MODELS_IMPORTED__ = False

if not __PROJECT_MODELS_IMPORTED__:
    from .HH import HH_Equation_Pairs, HH_Equation_Network, HH_Neuron_Network, HH_Neuron_Pairs, HH_Equation_Network_test, HH_Neuron_Network_tests, HH_Equation_Pairs_tests, HH_Neuron_Pairs_test
    from .Ishikevich import IZH_Equation_Network, IZH_Equation_Pairs, IZH_Neuron_Network, IZH_Neuron_Pairs, IZH_Equation_Network_tests, IZH_Neuron_Network_tests, IZH_Equation_Pairs_tests, IZH_Neuron_Pairs_tests
    from .LIF import  LIF_Equation_Pairs, LIF_Neuron_Pairs, LIF_Equation_Network, LIF_Neuron_Network, LIF_Equation_Network_tests, LIF_Neuron_Network_tests, LIF_Equation_Pairs_tests, LIF_Neuron_Pairs_tests
    from .morrislecar import ML_Equation_Network, ML_Equation_Pairs, ML_Neuron_Network, ML_Neuron_Pairs, ML_Equation_Network_tests, ML_Neuron_Network_tests, ML_Equation_Pairs_tests, ML_Neuron_Pairs_tests

__PROJECT_MODELS_IMPORTED__ = True