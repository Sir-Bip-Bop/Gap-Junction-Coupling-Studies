# MastersProject
Repository with the code of the masters project and a link to the overleaf document, to be able to move through both spaces quickly, and be able to see changes done.

### Project Structure


**MultipleNeurons:** A directory containing all the experiments done for networks of neurons (more than two).

**NeuronPairs:** A directory containing all the experiments done for pairs of two interconnected neurons.

**old:** A directory containing all the outdated code, notebooks that were used for testing, or code from matlab

**phaseportrait:** A directory containing all the work done in order to plot and study the phase portraits and the intersection of the nullclines.

**project:** The directory containing all the functions used in this project, it is necessary to install it using pip to run the functions correctly. This directory contains the following two subdirectoties.

    **models:** A directory containing the integrate functions for each of the models we use in the simulations, the algorithm is the same for all (4th-order Runge-Kutta), the various functions are only distinct in the output they give or if they are ready for single or multiple cells.

    **utils:** A directory containing all the miscellanous 'utility' functions used throughout the different experiments

**VoltageProfiling.ipynb:** This initial notebook showcases the work done in order to obtain the initial parameter values for each of the models used in this project



