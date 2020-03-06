"""
Photometric method for determining surface orientation from multiple
images. R.J.Woodham, 1980

Synthesize an sphere z = - sqrt(r**2 - x**2 - y**2).
Assuming ortographic projection: x=p, y=q
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import reflectance_map as rm
import utils

def build_image(rmap, r,x,y):
	"""
	"""
	img = np.zeros(rmap.shape)
	for i in range(rmap.shape[0]):
		for j in range(rmap.shape[1]):
			img[i,j] = max(0,rmap[i,j])
	img[(x**2 + y**2) > r**2] = 0
	return img



def get_sphere(x,y,r):
	z = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			z[i,j] = [-np.sqrt(r**2 -x[i,j]**2 - y[i,j]**2) if (x[i,j]**2 + y[i,j]**2) < r**2 else 0][0]

	return z

def draw_contours(x,y,z,ax,cm):
	# draw contour plots
	z_contours = np.arange(np.min(z),np.max(z), (np.max(z)- np.min(z)) / 10)
	z_min_draw = np.min(z)
	ax.contour(x, y, z, zdir='z', offset=z_min_draw, cmap=cm, levels = z_contours)
	#return ax
 
def reconstruct_normals_3lights(imgs, lights, spacing = 1):
	"""
	imgs: nxnx3 matriz with n the size of images stacked in depth
	lights: 3x3 matrix of unitary incident light vectors
	"""
	linv = np.linalg.inv(lights)
	#linvs = np.full((16,16,3,3),linv)
	normals = np.zeros((imgs.shape[0],imgs.shape[1],3))
	albedo = np.zeros((imgs.shape[0],imgs.shape[1]))
	#normals[:,:] = np.dot(linvs,imgs)
	for i in range(0,imgs.shape[0],spacing):
		for j in range(0,imgs.shape[1],spacing):
			normals[i,j] = np.dot(linv,imgs[i,j])
			albedo[i,j] = np.linalg.norm(normals[i,j])
			normals[i,j] = normals[i,j]/albedo[i,j]
	return normals,albedo

def reconstruct_normals_nlights(imgs, lights):
	"""
	imgs: nxnx3 matriz with n the size of images stacked in depth
	lights: 3x3 matrix of unitary incident light vectors
	"""
	linv = np.linalg.pinv(lights)
	#print(linv)
	#linvs = np.full((16,16,3,3),linv)
	normals = np.zeros((imgs.shape[0],imgs.shape[1],3))
	albedo = np.zeros((imgs.shape[0],imgs.shape[1]))

	#normals[:,:] = np.dot(linvs,imgs)
	for i in range(imgs.shape[0]):
		for j in range(imgs.shape[1]):
			normals[i,j] = np.dot(linv,imgs[i,j])
			albedo[i,j] = np.linalg.norm(normals[i,j])
			normals[i,j] = normals[i,j]/albedo[i,j]
	return normals, albedo

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def getSurface(normals,method = 0):

	normal_x = normals[:,:,0]
	normal_y = normals[:,:,1]
	normal_z = normals[:,:,2]

	x_derivative = normal_x/normal_z
	y_derivative = normal_y/normal_z

	output = np.zeros(normals.shape[0:2])
	if method == 0:
		# columns first
		output = np.cumsum(x_derivative,axis=0)
		output = output + np.cumsum(y_derivative,axis=1) 
		return output
	elif method == 1:
		# rows first
		pass
	elif method == 2:
		# average
		pass
	else:
		return output

def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    imx = surface_normals.shape[1]
    imy = surface_normals.shape[0] #flipped for indexing
    #height_map = np.zeros((imx,imy)
    fx = surface_normals[:,:,0] / surface_normals[:,:,2]
    fy = surface_normals[:,:,1] / surface_normals[:,:,2]
    fy = np.nan_to_num(fy)
    fx = np.nan_to_num(fx)
    row = np.cumsum(fx,axis=1)
    column = np.cumsum(fy,axis=0)
    if integration_method == 'row':
        row_temp = np.vstack([row[0,:]]*imy)
        height_map = column + row_temp     
        #print(np.max(height_map))
    elif integration_method == 'column':
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = row + col_temp   
        #print(height_map.T)
    elif integration_method == 'average':
        row_temp = np.vstack([row[0,:]]*imy)
        col_temp = np.stack([column[:,0].T]*imx,axis=1)
        height_map = (row + column + row_temp + col_temp) / 2
        
    elif integration_method == 'random':
        iteration = 10
        height_map = np.zeros((imy,imx))
        for x in range(iteration):
            print(x)
            for i in range(imy):
                print(i)
                for j in range(imx):
                    id1 = 0
                    id2 = 0
                    val = 0
                    path = [0] * i + [1] * j
                    random.shuffle(path)
                    for move in path:
                        if move == 0:
                            id1 += 1
                            if id1 > imy - 1: id1 -= 1
                            val += fy[id1][id2]
                            #print(val,fx[id1][id2])
                        if move == 1:
                            id2 += 1
                            if id2 > imx - 1: id2 -= 1
                            val += fx[id1][id2]
                    height_map[i][j] += val
                    #print(i,j,val)
        height_map = height_map / iteration
        #print(np.max(height_map))
    else:
    	pass
    # print(height_map)
    return height_map

def find_surface(N, size, h_map, K = 2000, p = False):
    h, w = size
    ret = h_map
    fx = N[:,:,0] / N[:,:,2]
    fx = np.nan_to_num(fx)
    fy = N[:,:,1] / N[:,:,2]
    fy = np.nan_to_num(fy)
    fxp = np.roll(fx,1,axis = 1)
    fxp[:,0] = 0
    deriv_x = fxp - fx
    fyp = np.roll(fy,1,axis = 0)
    fyp[0,:] = 0
    deriv_y = fyp - fy

    for i in range(K):
        dx = 1
        if p:
            dx = (-15.0 / 49.0) * i + 16
        dy = dx
        left = np.roll(ret,1,axis=1)
        left[:,0] = 0
        right = np.roll(ret,-1,axis=1)
        right[:,-1] = 0
        up = np.roll(ret,1,axis=0)
        up[0,:] = 0
        down = np.roll(ret,-1,axis=0)
        down[-1,:] = 0
        neighbor = 1/4 * (up + left + right + down)
        deriv = 1/4 * (deriv_x * dx + deriv_y * dy)
        ret = neighbor + deriv

    return ret

def threshold(h_map):
    h_map += abs(np.min(h_map))
    # h_map /= np.max(h_map)
    h_map[h_map < np.max(h_map) * 0.6] = np.max(h_map) * 0.6
    return h_map 

if __name__ == '__main__':
	
	fig = plt.figure(figsize=(40,40))
	fig3 = plt.figure(figsize=(40,40))
	ax1 = fig3.add_subplot(1,1,1, projection='3d')
	ax2 = fig.add_subplot(2,3,2)
	ax3 = fig.add_subplot(2,3,4)
	ax4 = fig.add_subplot(2,3,5)
	ax5 = fig.add_subplot(2,3,3, projection='3d')
	ax6 = fig.add_subplot(2,3,6)

	r = 3

	# define sphere
	x = np.arange(-4, 4, 0.5)
	y = np.arange(-4, 4, 0.5)
	x,y = np.meshgrid(x,y)
	z = get_sphere(x,y,r)#-np.sqrt(r**2 - x**2 - y**2)
	cont1 = ax1.plot_surface(x,y,z)

	# gradient
	p = -x/z
	q = -y/z
	#p[p == np.inf]=0
	#p[p == -np.inf] = 0
	#q[q == np.inf]=0
	#q[q == -np.inf] = 0
	normals = np.dstack((p,q,-np.ones_like(p)))

	# draw sphere normals
	norm = np.linalg.norm(normals,axis=2)
	norm = np.dstack((norm, norm, norm))
	normals_normalized = np.divide(normals,norm) 
	starts = np.dstack((x,y,z))
	point_vectors = np.dstack((starts,normals_normalized)).reshape(-1,6)
	#print(point_vectors)
	a,b,c,u,v,w = zip(*point_vectors)
	ax1.quiver(a,b,c,u,v,w, pivot = "tail", color='r', arrow_length_ratio=0.3)

	observer = np.array([0,0,-1])

	# We need to syntethize images for 3 lights
	light1 = np.array([0.7,0.3,-1])
	light2 = np.array([-0.610,0.456,-1])
	light3 = np.array([-0.090,-0.756,-1])
	light4 = np.dot(utils.rotation_matrix(observer, np.pi/4),light1)

	# get reflection maps for sphere
	Ra1 = rm.get_Ra(normals,light1,1)
	Ra2 = rm.get_Ra(normals,light2,1)
	Ra3 = rm.get_Ra(normals,light3,1)
	Ra4 = rm.get_Ra(normals,light4,1)

	# generate images
	im1 = build_image(Ra1,r,x,y)
	im2 = build_image(Ra2,r,x,y)
	im3 = build_image(Ra3,r,x,y)
	im4 = build_image(Ra4,r,x,y)

	# show images
	ax2.imshow(im1)
	ax3.imshow(im2)
	ax4.imshow(im3)
	ax6.imshow(im4)


	# build reflection maps
	p = np.arange(-4,4,0.5)
	q = np.arange(-4,4,0.5)
	p,q = np.meshgrid(p,q)
	normals = np.dstack((p,q,-np.ones_like(p)))
	# get general reflection maps
	Ra1 = rm.get_Ra(normals,light1,1)
	Ra2 = rm.get_Ra(normals,light2,1)
	Ra3 = rm.get_Ra(normals,light3,1)
	Ra4 = rm.get_Ra(normals,light4,1)

	s1 = ax5.plot_surface(p,q,Ra1, cmap = cm.viridis)
	s2 = ax5.plot_surface(p,q,Ra2, cmap = cm.inferno)
	s3 = ax5.plot_surface(p,q,Ra3, cmap = cm.magma)	
	s4 = ax5.plot_surface(p,q,Ra4, cmap = cm.cividis)

	# draw reflection map contours
	fig2 = plt.figure(figsize=(20,20))
	fig2ax1 = fig2.add_subplot(1,1,1,projection='3d')
	draw_contours(p,q,Ra1,fig2ax1, cm.Blues)
	draw_contours(p,q,Ra2,fig2ax1, cm.Greens)
	draw_contours(p,q,Ra3,fig2ax1, cm.Reds)
	draw_contours(p,q,Ra4,fig2ax1, cm.RdPu)

	# reconstruct normals using 3 images and lights
	imgs = np.dstack((im1,im2,im3))
	lights = np.vstack((normalize(light1), normalize(light2), normalize(light3)))
	surf_normals, albedo = reconstruct_normals_3lights(imgs,lights)
	norm = np.linalg.norm(surf_normals,axis=2)
	norm = np.dstack((norm, norm, norm))
	normals_normalized = np.divide(surf_normals,norm)
	point_vectors = np.dstack((starts,normals_normalized)).reshape(-1,6)
	a,b,c,u,v,w = zip(*point_vectors)
	ax1.quiver(a,b,c,u,v,w, pivot = "tail", color='g', arrow_length_ratio=0.3)	

	# reconstruct normals using 4 images and lights
	imgs = np.dstack((im1,im2,im3,im4))
	lights = np.vstack((normalize(light1), normalize(light2), normalize(light3), normalize(light4)))
	surf_normals = reconstruct_normals_nlights(imgs,lights)
	norm = np.linalg.norm(surf_normals,axis=2)
	norm = np.dstack((norm, norm, norm))
	normals_normalized = np.divide(surf_normals,norm)
	point_vectors = np.dstack((starts,normals_normalized)).reshape(-1,6)
	a,b,c,u,v,w = zip(*point_vectors)
	ax1.quiver(a,b,c,u,v,w, pivot = "tail", color='b', arrow_length_ratio=0.3)	

	plt.tight_layout()
	plt.show()




