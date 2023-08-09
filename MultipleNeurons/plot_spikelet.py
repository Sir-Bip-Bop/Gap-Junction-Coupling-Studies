import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
from scipy.sparse import dok_matrix
import multiprocessing as mp
import scipy as sp
import matplotlib.ticker as ticker
import project
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 18})
plt.rcParams.update({"axes.labelsize": 15})

#k = np.loadtxt('MultipleNeurons/k_plus_0.0007/k_data.txt')
#num = np.loadtxt('MultipleNeurons/k_plus_0.0007/num_data.txt')
#diff_HH = np.loadtxt('MultipleNeurons/k_plus_0.0007/diff_HH_data.txt')

k = np.loadtxt('MultipleNeurons/k_plus_divided/k_data.txt')
num = np.loadtxt('MultipleNeurons/k_plus_divided/num_data.txt')
diff_HH = np.loadtxt('MultipleNeurons/k_plus_divided/diff_HH_data.txt')

#discarding the values where diff_HH is zero
#diff_HH[diff_HH<1e-2] = np.nan

#Plot and fit of the resulting data
K, NUM = np.meshgrid(k,num)

def fit_fun(k, num, k_coef, num_coef,base):
    return (k * k_coef * (num**6)*num_coef + base*k)
    #return 2

fig = plt.figure()
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(K, NUM, diff_HH,cmap = 'cool')
ax.set_xlabel('Gap junction Strength')
ax.set_ylabel('Neuron Number')
ax.set_zlabel('Spikelet Height')
plt.title('3D surface of spikelet height',fontsize=22)
plt.show()

def _fit_fun(M, *args):
    k, num = M
    arr = np.zeros(k.shape)
    arr += fit_fun(k,num, *args)

    return arr 

guess_prms = [3, 0.2,2]
mask = ~np.isnan(diff_HH)
kdata = np.vstack((K[mask].ravel(),NUM[mask].ravel()))
popt, pcov = sp.optimize.curve_fit(_fit_fun,kdata,diff_HH[mask].ravel(),guess_prms)#,nan_policy = 'omit')



fig = plt.figure()
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(K, NUM, fit_fun(K,NUM,*popt),cmap = 'summer')
ax.plot_surface(K, NUM, diff_HH,cmap = 'cool')
cset = ax.contourf(K, NUM, diff_HH-fit_fun(K,NUM,*popt), zdir='z', offset=0, cmap='summer')
ax.set_xlabel('Gap junction Strength')
ax.set_ylabel('Neuron Number')
ax.set_zlabel('Spikelet Height')
plt.title('3D surface of spikelet height fit and residue',fontsize=22)
plt.show()

print('The final parameters:', popt)

# Process 2D inputs
poly = PolynomialFeatures(degree=3)
input_pts = np.stack([K[mask], NUM[mask]]).T
#assert(input_pts.shape == (400, 2))
in_features = poly.fit_transform(input_pts)

# Linear regression
model = LinearRegression()
model.fit(in_features, diff_HH[mask])

# Display coefficients
print(dict(zip(poly.get_feature_names_out(), model.coef_.round(4))))

# Check fit
print(f"R-squared: {model.score(poly.transform(input_pts), diff_HH[mask]):.3f}")


def fit_fun_2(k,num):
    return 65.0738*k + 0.0309*num -601.2423*k**2 -3.5194*k*num - 0.0034*num**2 + 5255.1221*k**3 + 122.9063*k**2*num + 0.1943*k*num**2 + 0.0001*num**3

fig = plt.figure()
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(K, NUM, fit_fun_2(K,NUM),cmap = 'summer')
ax.plot_surface(K, NUM, diff_HH,cmap = 'cool')
cset = ax.contourf(K, NUM, diff_HH-fit_fun_2(K,NUM), zdir='z', offset=0, cmap='summer')
ax.set_xlabel('Gap junction Strength')
ax.set_ylabel('Neuron Number')
ax.set_zlabel('Spikelet Height')
plt.title('3D surface of spikelet height fit and residue',fontsize=22)
plt.show()


#plt.grid()
#plt.pcolor(k,num,diff_HH)
#plt.colorbar()
#plt.show()


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.contour3D(k,num,diff_HH)
#plt.show()