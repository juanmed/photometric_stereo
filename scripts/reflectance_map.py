"""
Photometric method for determining surface orientation from multiple
images. R.J.Woodham, 1980
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_Ra(normal, light, q):
	"""
	Idealized model for prefectly diffuse (lambertian) surface 
	which is equally bright from all viewing directions
	"""
	lights = np.full(normal.shape, light)
	a = np.zeros(normal.shape[0:2])
	for i in range(normal.shape[0]):
		for j in range(normal.shape[1]):
			n = normal[i,j]
			a[i,j] = np.dot(n,light)/(np.linalg.norm(n)*np.linalg.norm(light))
	#a = np.dot(normal[:,:],light) 
	#a = a/np.linalg.norm(normal[:,:])
	#a = a/np.linalg.norm(light)
	return a

def get_Rb(normal, light, q):
	"""
	Idealized model for prefectly diffuse (lambertian) surface 
	which reflects equal amounts of light in all directions.
	"""
	lights = np.full(normal.shape, light)
	a = np.zeros(normal.shape[0:2])
	for i in range(normal.shape[0]):
		for j in range(normal.shape[1]):
			n = normal[i,j]
			a[i,j] = np.dot(n,light)/(np.linalg.norm(light))
	#a = np.dot(normal[:,:],light) 
	#a = a/np.linalg.norm(normal[:,:])
	#a = a/np.linalg.norm(light)
	return a

if __name__ == '__main__':
	# scene quantities
	light = np.array([0.7, 0.3, -1])
	qf = 1. # reflectance factor

	# object gradients and normals
	p = np.arange(-6,6,0.5)
	q = np.arange(-6,6,0.5)
	p, q = np.meshgrid(p,q)

	normals = np.dstack((p,q,-np.ones_like(p)))

	# Get reflectance maps
	Rb = get_Rb(normals, light, qf)
	Ra = get_Ra(normals, light, qf)


	# draw reflectance map
	fig = plt.figure(figsize=(20,20))
	ax = fig.add_subplot(111, projection='3d')
	cont1 = ax.plot_surface(p,q,Rb, cmap=cm.viridis)
	cont2 = ax.plot_surface(p,q,Ra, cmap=cm.coolwarm)
	# Add a color bar which maps values to colors.
	fig.colorbar(cont1)
	#fig.colorbar(cont2)

	# draw normals 
	light = light/(np.linalg.norm(light)*2)
	ax.quiver(0,0,0,light[0],light[1],light[2],  arrow_length_ratio=0.3, color='r')
	print(np.linalg.norm(normals[:,:]))
	normals_normalized = 10*normals[:,:]/np.linalg.norm(normals) #vector lengtjs
	starts = normals.copy() # vector starting points
	starts[:,:,2]=Ra
	point_vectors = np.dstack((starts,normals_normalized)).reshape(-1,6)
	#print(point_vectors)
	x,y,z,u,v,w = zip(*point_vectors)
	ax.quiver(x,y,z,u,v,w, pivot = "tail", color='b', arrow_length_ratio=0.3)


	Ra_contours = np.arange(np.min(Ra),np.max(Ra), (np.max(Ra)- np.min(Ra)) / 10)
	Ra_min_draw = np.min(Ra) - 4
	cset = ax.contour(p, q, Ra, zdir='z', offset=Ra_min_draw, cmap=cm.coolwarm, levels = Ra_contours)


	plt.show()