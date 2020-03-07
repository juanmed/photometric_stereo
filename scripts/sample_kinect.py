#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from point_cloud_from_depth import Depth_to_pc_Converter
import vision_utils as utils

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


class Box_Registrar():

    def __init__(self):

        self.converter = 0

        rospy.Subscriber("/image", Image, self.rgb_img_cb)
        rospy.Subscriber("/depth", Image, self.depth_img_cb)
        rospy.Subscriber("/camera_info", CameraInfo, self.cam_info_cb)

        self.img_publisher = rospy.Publisher("/segmented_img", Image, queue_size = 10)
        self.pc_pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)

        self.bridge = CvBridge()
        self.fx = 0
        self.fy = 0
        self.cx = 0
        self.cy = 0
        self.cam_pams_ready = False

        self.color_img = np.zeros((1))
        self.mask = np.zeros((1))
        self.original = np.zeros((1))

    def depth_img_cb(self, image_msg):
        """ Docstring """
        # changing the encoding is just a hack for cv bridge to 
        # allow 16UC1 conversion to img.
        # read more: https://github.com/ros/common_msgs/pull/107
        image_msg.encoding = "mono8" 
        dimg = self.bridge.imgmsg_to_cv2(image_msg, "passthrough")
        print("Depth: ",dimg.shape)
        print("Color: ", self.color_img.shape)

        if (self.cam_pams_ready and (self.color_img.shape[0] > 1)):
            cv2.imwrite("/home/jfmy/rosws/ps_ws/src/depth_camera_003.png", dimg)
            cv2.imwrite("/home/jfmy/rosws/ps_ws/src/rgb_camera_003.png",self.original)
            """
            for i in np.unique(self.mask):
                #img_patch = self.color_img[self.mask == 0].reshape(-1,-1,3)
                #dimg_patch = dimg[self.mask == 0].reshape(-1,-1)
                #print(img_patch.shape, dimg_patch.shape)
                #img_patch = np.where(self.mask == 0, self.color_img, np.zeros_like(self.color_img))
                #dimg_patch = np.where(self.mask == 0, self.dimg, np.zeros_like(self.dimg))
                mask = np.zeros(self.color_img.shape[:2], dtype="uint8")
                mask[self.mask == i] = 255
                img_patch = cv2.bitwise_and(self.color_img, self.color_img, mask = mask)
                dimg_patch = cv2.bitwise_and(dimg, dimg, mask = mask)
                pc_msg = self.converter.get_pointCloud2_msg(dimg_patch, img_patch, 1500, 255)
                pc_msg.header.stamp = rospy.Time.now()
                pc_msg.header.frame_id = "world"
                self.pc_pub.publish(pc_msg)
            """ 
        else:
            rospy.loginfo("Camera params not received or image segmentation not ready.")

    def cam_info_cb(self, msg):
        if not self.cam_pams_ready:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.converter = Depth_to_pc_Converter(self.fx, self.fy, self.cx, self.cy)
            self.cam_pams_ready = True
            #del self.cam_pams_sub

    def rgb_img_cb(self, image_msg):
        """ Docstring
        fshw, scale = 400, sigma = 2, minsize = 500
        watershed, megapixel = 20, compactness = 0.0001
         """
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        self.original = img
        blur_img = utils.applyBlur(img, 1)
        self.mask, bound_img = utils.get_superpixels(blur_img, 400 , 2, True)  #fshw  = 400
        self.color_img = bound_img*255
        maxa = np.max(self.color_img)
        msg = self.bridge.cv2_to_imgmsg((self.color_img*255/maxa).astype(img.dtype), "rgb8")
        self.img_publisher.publish(msg)  

    def icp(self):
        R = np.diag([1,1,1])
        t = np.ones((3))
        return R,t

if __name__ == '__main__':
    try:
        rospy.init_node("pc_from_depth_image_publisher", anonymous = True)
        
        registrar = Box_Registrar()

        rospy.loginfo('Point Cloud from Depth Publisher Started')
        rospy.spin()
        rospy.loginfo('Point Cloud from Depth Publisher Terminated')     

    except rospy.ROSInterruptException:
        print("ROS Terminated.")
        pass    