from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots


#General plot style used in the project, and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 17})
plt.rcParams.update({"axes.labelsize": 15})


#Definition of the functions to integrate, that is the IF equation
def LIF(I, V, *, C = 1, gl = 0.1, El = -70):
  return  1, 1/C * (-gl*(V-El) + I)
	

#Creation of the phase diagram
phase_diagram = PhasePortrait2D(LIF, [[-20,20],[-70,30]],
    Density= 4,
	dF_args = {'C': 1, 'gl': 0.1, 'El': -70},
	MeshDim = 20,
	Title = 'LIF Phase Portrait',
	xlabel = r'Intensity$(\mu A)$',
	ylabel = 'Voltage(mV)',
	color= 'cool',
)


#Creation of the plot, the constant lines are representing the threshold and reset values
#phase_diagram.add_nullclines(xcolor='red', ycolor='green')
phase_diagram.plot()
phase_diagram.ax.hlines(-49.2,-20,20, color = 'green', label= 'V Threshold')
phase_diagram.ax.hlines(-66.9, -20,20, color = 'red', label= 'V Reset')
phase_diagram.ax.hlines(25,-20,20, color = 'blue', label = 'V Peak')
phase_diagram.ax.legend(loc='right', bbox_to_anchor=(0.27, 0.83),prop={'size':15},
          ncol=1,frameon=True, fancybox=True, shadow=False)

plt.show()