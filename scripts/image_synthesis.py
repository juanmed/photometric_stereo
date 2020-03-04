"""
Photometric method for determining surface orientation from multiple
images. R.J.Woodham, 1980

Synthesize an sphere z = - sqrt(r**2 - x**2 - y**2).
Assuming ortographic projection: x=p, y=q
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import reflectance_map as rm

def build_image(rmap, r):
	"""
	"""
	img = np.zeros(rmap.shape)
	for i in range(rmap.shape[0]):
		for j in range(rmap.shape[1]):
			pass


def get_sphere(x,y,r):
	z = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			z[i,j] = [r**2 -x[i,j]**2 - y[i,j]**2 if (x[i,j]**2 + y[i,j]**2) < r**2 else 0.001][0]

	return z

fig = plt.figure(figsize=(40,40))
ax1 = fig.add_subplot(2,2,1, projection='3d')
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

r = 3

# define sphere
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
x,y = np.meshgrid(x,y)
z = -np.sqrt(r**2 - x**2 - y**2)
cont1 = ax1.plot_surface(x,y,z,cmap=cm.viridis)

# gradient
p = -x/z
q = -y/z
normals = np.dstack((p,q,-np.ones_like(p)))

# We need to syntethize images for 3 lights
light1 = np.array([0.7,0.3,-1])
light2 = np.array([-0.610,0.456,-1])
light3 = np.array([-0.090,-0.756,-1])
Ra1 = rm.get_Ra(normals,light1,1)
Ra2 = rm.get_Ra(normals,light2,1)
Ra3 = rm.get_Ra(normals,light3,1)

im1 = np.zeros(Ra1.shape)
im1[np.where(z is not np.nan)] = Ra3[np.where(z is not np.nan)]
ax2.imshow(Ra1)
ax3.imshow(Ra2)
ax4.imshow(Ra3)



plt.show()




