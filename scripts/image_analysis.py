import cv2
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils
import image_synthesis as isyn


if __name__ == '__main__':

	x = np.arange(-2,2,0.5)
	y = np.arange(-2,2,0.5)
	x,y = np.meshgrid(x,y)
	z = -x**2 -y**2
	
	fig1 = plt.figure(figsize=(20,20))
	fig1ax1 = fig1.add_subplot(1,1,1,projection='3d')
	#fig1ax2 = fig1.add_subplot(1,2,2)

	# define light vectors
	base_angle = np.pi/(2*5)
	light1 = np.array([0,0,-1])
	mx = np.array([-1,0,0])
	light2 = np.dot(utils.rotation_matrix(mx, base_angle),light1)
	light3 = np.array([1,0,0])
	light4 = np.dot(utils.rotation_matrix(-mx, base_angle),light1)
	light5 = np.dot(utils.rotation_matrix(-mx, base_angle*2),light1)

	lights = np.vstack((light1, light2, light3, light4, light5))
	starts = np.zeros((5,3))
	point_vectors = np.hstack((starts,lights))
	a,b,c,u,v,w = zip(*point_vectors)
	#fig1ax1.quiver(a,b,c,u,v,w, pivot = "tail", color='r', arrow_length_ratio=0.3)
	#fig1ax1.plot_surface(x,y,z, alpha=0.1)
	# read images
	im1 = cv2.imread('images/rgb_camera_001.png', cv2.IMREAD_GRAYSCALE)
	h,w= im1.shape
	im2 = cv2.imread('images/rgb_camera_002.png', cv2.IMREAD_GRAYSCALE)
	im3 = cv2.imread('images/rgb_camera_003.png', cv2.IMREAD_GRAYSCALE)
	im4 = cv2.imread('images/rgb_camera_004.png', cv2.IMREAD_GRAYSCALE)
	im5 = cv2.imread('images/rgb_camera_005.png', cv2.IMREAD_GRAYSCALE)
	fx = fy = 0.5
	im1 = cv2.resize(im1, None, fx=fx, fy=fy)
	im2 = cv2.resize(im2, None, fx=fx, fy=fy)
	im3 = cv2.resize(im3, None, fx=fx, fy=fy)
	im4 = cv2.resize(im4, None, fx=fx, fy=fy)
	im5 = cv2.resize(im5, None, fx=fx, fy=fy)
	
	# reconstruct normals
	imgs = np.dstack((im1, im2, im3, im4, im5))
	surf_normals, albedo = isyn.reconstruct_normals_nlights(imgs,lights)
	surf_normals = np.nan_to_num(surf_normals)
	print(surf_normals.shape)
	
	x = np.arange(0,w,2)
	y = np.arange(0,h,2)
	x,y = np.meshgrid(x,y)
	#starts = np.dstack((x,y,np.zeros_like(x)))
	#norm = np.linalg.norm(surf_normals,axis=2)
	#norm = np.dstack((norm, norm, norm))
	#normals_normalized = np.divide(surf_normals,norm)
	height_map = isyn.get_surface(surf_normals,'column')
	height_map = isyn.find_surface(surf_normals, imgs.shape[0:2], height_map)
	fig1ax1.plot_surface(x,y,height_map)

	#point_vectors = np.dstack((starts,normals_normalized)).reshape(-1,6)
	#a,b,c,u,v,w = zip(*point_vectors)
	#fig1ax1.quiver(a,b,c,u,v,w, pivot = "tail", color='g', arrow_length_ratio=0.3)	
	
	
	plt.show()