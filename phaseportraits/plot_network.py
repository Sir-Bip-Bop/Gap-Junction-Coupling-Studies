import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import count
from pylab import rcParams
import networkx as nx
import imageio
import os

filenames=[]

name=("phaseportraits/datos_1000.txt")
print(name)
G = nx.read_edgelist(name,create_using=nx.Graph(), nodetype = int)
pos = nx.spring_layout(G)
#enfermos=[]
#recuperados=[]
#with open("Infectados/infectados_4.txt") as f:
#    for line in f:
#        enfermos.append(int(line))
#with open("Recuperados/recuperados_4.txt") as f:
#    for line in f:
#       recuperados.append(int(line))P
black_edges = [edge for edge in G.edges()]
#values=[]
#for node in G.nodes():
#    if node not in enfermos:
#        if node not in recuperados:
#            values.append('blue')
#        else:
#            values.append('green')
#    else:
#        values.append('red')
d= dict(G.degree)

color_lookup = {k:v for v,k in enumerate(sorted(set(G.degree())))}
low, *_, high = sorted(color_lookup.values())
norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
rcParams['figure.figsize'] = 7, 7

nx.draw_kamada_kawai(G,edgelist=black_edges, edge_color = 'grey',node_color=[mapper.to_rgba(i) 
                    for i in color_lookup.values()], node_size=[v * 10 for v in d.values()], arrows=False)
#plt.show()

mapper.set_array([])
ax=plt.gca() #get the current axes
#cbar = plt.colorbar(mapper,ax=ax,shrink=0.6)

filename = f'Red_BA.png'
filenames.append(filename)
plt.savefig(filename)
plt.close()