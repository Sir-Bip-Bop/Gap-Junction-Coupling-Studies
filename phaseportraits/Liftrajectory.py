from phaseportrait import Trajectory2D, Trajectory3D
import matplotlib.pyplot as plt

def LIF(I, V, *, C = 1, gl = 0.1, El = -70):
  return  1, 1/C * (-gl*(V-El) + I)

circle = Trajectory2D(LIF, n_points=628, size=2)
circle.initial_position(2,-70)

circle.plot()
plt.show()