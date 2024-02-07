import numpy as np
import os
import cv2
import glob
from libs.visualization import Visualization

class Experiments:
    rois = []
    def __init__(self, input, export_depth_data, npy):
        self.input = input
        self.npy = npy
        self.export_depth_data = export_depth_data
        self.start = None
        self.end = None
        self.ROI = None
        self.all_depth_data = None

    @classmethod
    def crop_pallette(cls, rois):
        scaled_roi = []
        for roi in rois:
            individual_pallete = []
            for num in roi:
                num_x = num[0]-rois[0][0][0]
                num_y = num[1]-rois[0][0][1]
                individual_pallete.append(list([num_x, num_y]))
            scaled_roi.append(list(individual_pallete))
        return scaled_roi

    def bounding_box(self, image):
        '''input: original image
        output: cropped image with selected region of interest s.t. the depth values are > 0'''
        self.start = self.ROI[0][0]
        self.end = self.ROI[-1][-1]
        return image[self.start[1]:self.end[1],self.start[0]:self.end[0]]

    def get_depth_data(self):
        #TO DO: Make this function dynamic s.t. user can give his own ROI
        all_color = glob.glob(os.path.join(self.input[0], "*.jpg"))    # for color data
        self.ROI = Visualization.get_cropped_rois(image_path=all_color[0])
            
        if self.export_depth_data:
            all_depth = glob.glob(os.path.join(self.input[1], "*.png"))    # for depth data
            x1, y1 = self.ROI[0][0]
            x2, y2 = self.ROI[-1][-1]
            c = abs(x2 - x1)
            r = abs(y2 - y1)
            n_frames = len(all_depth)
            self.all_depth_data = np.ones((n_frames, r, c))
            for i, image in enumerate(all_depth):
                depth_image = cv2.imread(image, -1)
                cropped_image = self.bounding_box(depth_image)
                depth_data = cropped_image #np.cos(6*np.pi/180) * cropped_image #for 6 degrees tilt of AzureKinect depth sensor wrt RGB camera
                self.all_depth_data[i] = depth_data
                print(f'Getting distance from {i}th depth frame')

            path_splitted = os.path.split(self.input[1])
            kinect_data_path = path_splitted[0]
            if isinstance(self.npy, list):
                self.npy = self.npy[0]
            np.save(os.path.join(kinect_data_path, self.npy), np.float32(self.all_depth_data))
            print(f'{self.npy} saved at {kinect_data_path}')
        return self.ROI, self.all_depth_data
