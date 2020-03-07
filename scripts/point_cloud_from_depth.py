#!/usr/bin/env python
import numpy as np
import struct
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

def build_arg_parser():
    """
    Build command line argument parser
    Return:
        Dictionary containing arguments and their values
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap = ap.parse_args()
    return ap

class Depth_to_pc_Converter():



    def __init__(self, fx = 0, fy = 0, cx = 0, cy = 0):
        
        # camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.FIELDS =   [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        # PointField('rgb', 12, PointField.UINT32, 1),
                        PointField('rgba', 12, PointField.UINT32, 1),]

    def get_point_cloud(self, depth,maxval):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.
        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.
        Made by Gareth Rees. Available at:
        https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < maxval)
        z = np.where(valid, depth / (maxval+1), np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)
        print(z.shape, x.shape, y.shape)
        return np.dstack((x, y, z))

    def get_colorPointCloud(self, depth, img, maxval):
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0) & (depth < maxval)
        z = np.where(valid, depth / (maxval+1), np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)  
        return np.dstack((x,y,z,img))      

    def get_pointCloud2_msg(self, depth, img, H, maxval):
        """
        To generate pointCloud2 message refer to:
        https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
        """
        points = []
        points = self.get_colorPointCloud(depth, img, maxval)
        rows,cols,channels = points.shape
        points = points.reshape(rows*cols,channels)
        alpha = 255
        points = list(map(lambda a,b,c,d,e,f: [c*H,-a*H,-b*H, struct.unpack('I', struct.pack('BBBB', f, e, d, alpha))[0]], points[:,0], points[:,1], points[:,2], points[:,3], points[:,4], points[:,5]))
        header = Header()
        pc2 = point_cloud2.create_cloud(header, self.FIELDS, points)
        return pc2



def main(args):

    dimg = cv2.imread(args.image)

    converter = Depth_to_pc_Converter(fx, fy, cx, cy)
    pc = converter.convert(dimg)





if __name__ == '__main__':
    args = build_arg_parser()
    main(args)