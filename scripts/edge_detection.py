import cv2
import numpy as np

if __name__ == '__main__':

    # load image
    im1 = cv2.imread('images/rgb_camera_001.png', cv2.IMREAD_GRAYSCALE)
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

    lt = 100
    ht = 200
    k = 3

    itera = 0
    # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    
    line_image1 = np.copy(im1) * 0  # creating a blank to draw lines on

    im1  = cv2.GaussianBlur(im1, (5,5),0) 
    edges1 = cv2.Canny(im1, lt, ht, k)
    edges1 = cv2.dilate(edges1, None, iterations = itera) 

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges1, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image1,(x1,y1),(x2,y2),(255,0,0),5)  

    cv2.imshow('window1', line_image1)
    cv2.waitKey()

    line_image2 = np.copy(im2) * 0  # creating a blank to draw lines on
    im2  = cv2.GaussianBlur(im2, (5,5),0)     
    edges2 = cv2.Canny(im2, lt, ht, k)
    edges2 = cv2.dilate(edges2, None, iterations = itera)    
    lines = cv2.HoughLinesP(edges2, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image2,(x1,y1),(x2,y2),(255,0,0),5)  

    cv2.imshow('window2', line_image2)
    cv2.waitKey

    line_image3 = np.copy(im3) * 0
    im3  = cv2.GaussianBlur(im3, (5,5),0) 
    edges3 = cv2.Canny(im3, lt, ht, k)
    edges3 = cv2.dilate(edges3, None, iterations = itera)  
    lines = cv2.HoughLinesP(edges3, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image3,(x1,y1),(x2,y2),(255,0,0),5)  

    cv2.imshow('window3', line_image3)
    cv2.waitKey()

    line_image4 = np.copy(im4) * 0
    im4  = cv2.GaussianBlur(im4, (5,5),0) 
    edges4 = cv2.Canny(im4, lt, ht, k)
    edges4 = cv2.dilate(edges4, None, iterations = itera) 
    lines = cv2.HoughLinesP(edges4, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image4,(x1,y1),(x2,y2),(255,0,0),5)     
    cv2.imshow('window4', line_image4)
    cv2.waitKey()

    line_image5 = np.copy(im5) * 0
    im5  = cv2.GaussianBlur(im5, (5,5),0) 
    edges5 = cv2.Canny(im5, lt, ht, k)
    edges5 = cv2.dilate(edges5, None, iterations = itera) 
    lines = cv2.HoughLinesP(edges5, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image5,(x1,y1),(x2,y2),(255,0,0),5)     
 
    cv2.imshow('window5', line_image5)
    cv2.waitKey()
    
    andedges = line_image1 | line_image2 | line_image3 | line_image4 | line_image5

    edges1 = edges1/255.
    edges2 = edges2/255.
    edges3 = edges3/255.
    edges4 = edges4/255.
    edges5 = edges5/255.

    suma = edges1 + edges2 + edges3 + edges4 + edges5
    suma = suma/np.max(suma)
    suma[suma < 0.5] = 0
    suma[suma >= 0.5] = 255
    suma = suma.astype(np.int8)

    cv2.imshow('window', andedges)
    cv2.waitKey()    