import cv2

from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray

def get_superpixels(image, megapixel_number, method, draw_megapixels):

    #change color space to HSV
    pimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if(method == 1):
        segments = slic(pimage, n_segments = megapixel_number, sigma = 5)

    if(method == 2):
        segments = felzenszwalb(pimage, scale=megapixel_number, sigma=2.0, min_size=500)

    if(method == 3):
        segments = quickshift(pimage, kernel_size=11, max_dist=megapixel_number/1.0, ratio=0.1)

    if(method == 4):
        gradient = sobel(rgb2gray(pimage))
        segments = watershed(gradient, markers=megapixel_number, compactness=0.0001)

    #if(draw_megapixels):
    megapixel_image = mark_boundaries(image,segments)
    #    cv2.imwrite("superpixeled.jpg",megapixel_image)

        #return megapixel_image

    #print("Segments Size: "  + str(type(segments)) + "Segments Dim: " + str(segments.shape))
    return segments, megapixel_image

def applyBlur(image, method):

    if(method == 1):
        kernel = 3
        img_smooth = cv2.GaussianBlur(image,(kernel,kernel),0,0,borderType=cv2.BORDER_REPLICATE) 
    elif(method == 2):
        bl_sigmacolor = 9
        bl_neighbor = 5
        bl_sigmaspace = 10
        img_smooth = cv2.bilateralFilter(image, bl_neighbor, bl_sigmacolor,bl_sigmaspace, borderType=cv2.BORDER_REPLICATE) 
    elif(method == 3):
        median_kernel = 5
        img_smooth = cv2.medianBlur(image, median_kernel)
    else:
        print ("NO BLUR APPLIED")

    #cv2.imwrite("blur"+args.input,img_smooth)#, [cv2.IMWRITE_JPEG_QUALITY, 0.9])
    return img_smooth