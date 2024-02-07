import numpy as np 
import os
import argparse
import matplotlib.pyplot as plt
from libs.experiments import Experiments
from libs.visualization import Visualization


def color_depth_overlap(data, export_depth):
    realsense_roi = [[541,240], [591,290]]
    kinect_roi = [[500,225], [550,275]]
    realsense = Experiments(data[1], export_depth, npy=None, start= np.asarray(realsense_roi[0],dtype=int), end= np.asarray(realsense_roi[1],dtype = int))   
    kinect = Experiments(data[0], export_depth,npy= None, start= np.asarray(kinect_roi[0], dtype= int), end= np.asarray(kinect_roi[1], dtype= int))  
    cropped_rs = realsense.crop_roi(cv2.imread(data[1])) 
    cropped_k4a = kinect.crop_roi(cv2.imread(data[0])) 
    return cropped_rs, cropped_k4a

def plots(obj_list, title_list, suptitle, camera, path, label_list, heatmap=True):
    Visualization.subplots(obj_list, title_list, suptitle, camera, path, label_list, heatmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=
            "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--input",
                        nargs = '+',
                        required=False,
                        help="set input depth folder")
    parser.add_argument("--npy",
                        required=False,
                        help="npy filename to export data or npy path to read data")
    parser.add_argument("--export_depth",
                        action='store_true',
                        help="option to view bounding box")
    parser.add_argument("--camera_id",
                        type=int,
                        required=False,
                        help="RealSense = 0 or AzureKinect = 1")
    parser.add_argument("--fig_path",
                        required=False,
                        help="path to save plots")

    args = parser.parse_args()

    ROI_coordinates = []
    data = None
    experiment3 = Experiments(args.input, args.export_depth, args.npy)
    if len(args.input)>1:        
        ROI_coordinates, depth_data = experiment3.get_depth_data()  # Depth values can be obtained from depth images only for Azure Kinect
        print(f'depth_data.shape = {depth_data.shape}, ROI: {ROI_coordinates}')
        path_splitted = os.path.split(args.input[1])
        kinect_data_path = path_splitted[0]
        data = np.load(os.path.join(kinect_data_path, args.npy))
        # data = (data + (0.0084))*1000
        # data = (data + (0.0084))

    else: 
        print(f"input: {args.input}")                      
        ROI_coordinates, _ = experiment3.get_depth_data()   # For RealSense Camera, only ROI coordinates can be obtained from color images
        print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
        data = np.load(args.npy)
        # data = (data + (0.0084))*1000
        # data = (data + (0.0084))

    scaled_roi = Experiments.crop_pallette(ROI_coordinates)
    # print(f"scaled ROI: {scaled_roi}")
    object_list = []
    data_list = []
    for i, roi in enumerate(scaled_roi):
        object_list.append(Visualization(data[:, roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]))
        data_list.append(data)

    obj_list = [object_list, data_list]
    textures_list = ["(1, 1)", "(1, 2)", "(1, 3)", "(2, 1)", "(2, 2)", "(2, 3)","(3, 1)", "(3, 2)", "(3, 3)"]
    camera=["RealSense","AzureKinect"]
    label_list=['m','mm']

    plots(obj_list, title_list=textures_list, suptitle="Effect of Texture", camera=camera[args.camera_id], path=args.fig_path, label_list=label_list[args.camera_id])
