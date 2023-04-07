from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def LIF(I, V, *, C = 1, gl = 0.1, El = -70):
  return  1, 1/C * (-gl*(V-El) + I)
	



phase_diagram = PhasePortrait2D(LIF, [[-20,20],[-70,100]],
    Density= 4,
	dF_args = {'C': 1, 'gl': 0.1, 'El': -70},
	MeshDim = 20,
	Title = 'LIF Phase portrait',
	xlabel = 'Intensity(mA)',
	ylabel = r'Voltage$(\mu V)$',
	color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
#phase_diagram.add_slider('gl', valinit = 0.1, valinterval=[-0.5,0.6],valstep=0.1)
#phase_diagram.add_slider('El', valinit=-70, valinterval=[-80,-60],valstep=5)
phase_diagram.add_nullclines(xcolor='black', ycolor='green')

phase_diagram.plot()
plt.show()