import numpy as np
import argparse 
import os
import matplotlib.pyplot as plt
from libs.experiments import Experiments
from libs.visualization import Visualization

def plots(obj_list, title_list, suptitle, camera, path, label_list, heatmap=True):
    if camera=="AzureKinect":
        Visualization.subplots(obj_list, title_list[::-1], suptitle, camera, path, label_list, heatmap)
    else:
        Visualization.subplots(obj_list, title_list, suptitle, camera, path, label_list, heatmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=
            "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--input",
                        required=False,
                        nargs = '+',
                        help="set path to input color and depth directory. Give color path as 1st argument")
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
        if args.camera_id==0:
            data = (data + (0.0084))*1000 #For RealSense

    else:                       
        ROI_coordinates, _ = experiment3.get_depth_data()   # For RealSense Camera, only ROI coordinates can be obtained from color images
        print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
        data = np.load(args.npy)
        data = (data + (0.0084))*1000
        # data = data+0.0084

    scaled_roi = Experiments.crop_pallette(ROI_coordinates)
    print(f"scaled ROI: {scaled_roi}")
    object_list = []
    data_list = []
    for i, roi in enumerate(scaled_roi):
        object_list.append(Visualization(data, roi[0], roi[1]))
        data_list.append(data)

    obj_list = [object_list, data_list]
    textures_list = ["Blue", "Green", "Red", "Black", "Yellow", "White"]
    camera=["RealSense","AzureKinect"]
    label_list=['mm','mm']

    plots(obj_list, title_list=textures_list, suptitle="Effect of Color (Absorptivity)", camera=camera[args.camera_id], path=args.fig_path, label_list=label_list[args.camera_id])
