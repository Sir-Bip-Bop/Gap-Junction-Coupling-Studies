from phaseportrait import *
import matplotlib.pyplot as plt
# import scienceplots
import numpy as np
import scipy as sp
import scipy.optimize as opt

# plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def func(x,y):
   return (x-1)*(x+1), (y-1)*(y+1)

phase_diagram = PhasePortrait2D(func, [[-2,2],[-2,2]],
      MeshDim = 20,
      Title = 'HH Phase portrait',
      xlabel = r'Voltage$(\mu V)$',
      ylabel = 'recovery variable',
      color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
# phase_diagram.add_slider('n', valinit = 0.5, valinterval=[0,1],valstep=0.05)
# phase_diagram.add_slider('h', valinit=0.5, valinterval=[0,1],valstep=0.05)
# phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
#phase_diagram.add_nullclines(xcolor='black', ycolor='green', precision=0.5)

func_X = lambda x, y: func(x,y)[0]
func_Y = lambda y, x: func(x,y)[1]

X = []
Y = []
ii = np.linspace(-2,2,100)
bb = np.linspace(-2,2,100)


for i in ii:
    solve_x = opt.fsolve(func_X,i,args=i)
    X.append(solve_x)

for i in bb:
    solve_y = opt.fsolve(func_Y,i,args=i)
    Y.append(solve_y)

phase_diagram.plot()

phase_diagram.ax.plot(X,ii, color= 'red')
phase_diagram.ax.plot(bb,Y, color = 'green')

plt.show()
