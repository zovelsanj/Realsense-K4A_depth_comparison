import numpy as np 
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import os
from libs.experiments import Experiments
from libs.visualization import Visualization
from libs.type_convert import convert_float64_to_32

def plots(data_list, title_list, suptitle, camera, path, label_list, heatmap, warmup, offset, check_distribution):
    if offset:
        Visualization.subplots(data_list, title_list[1], suptitle[1], camera, path, label_list, heatmap, offset)
        return
    if warmup:
        Visualization.subplots(data_list, title_list[0], suptitle[0], camera, path, label_list, heatmap)
    else:
        Visualization.subplots(data_list, title_list[1], suptitle[1], camera, path, label_list, heatmap, offset, check_distribution)

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
                        nargs = '*',
                        help="npy filename to export data or npy path to load data")
    parser.add_argument("--export_depth",
                        action='store_true',
                        help="option to export depth data after selection ROIs")
    parser.add_argument("--camera_id",
                        type=int,
                        help="RealSense = 0 or AzureKinect = 1")
    parser.add_argument("--fig_path",
                        required=False,
                        help="path to save plots")

    args = parser.parse_args()

    object_list = []
    data_list = []
    if len(args.npy)>1:
        for i, file_name in enumerate(args.npy):
            print(f'Loading {file_name}')
            data=np.load(file_name)
            convert_float64_to_32(data, out_dir=os.path.dirname(file_name), idx=[args.camera_id, i])
            if args.camera_id==0:
                data = (data+0.0082)*1000 #for RealSense

            object_list.append(Visualization(data,[50,50],[100,100]))
            data_list.append(data)
        obj_list = [object_list, data_list]
        title_list = [["Before Warmup", "After Warmup"], ["0.5m", "1m", "1.5m", "2m", "2.5m"]]
        suptitles = ["Effect of Warmup Time", "Effect of Camera Distance"]
        camera=["RealSense","AzureKinect"]
        label_list=['mm','mm']  #Replace mm by m in case metric required as meters.

        plots(obj_list, title_list, suptitles, camera=camera[args.camera_id], path=args.fig_path, label_list=label_list[args.camera_id], heatmap=False, warmup=False, offset=False, check_distribution=False)
    
    else:
        ROI_coordinates = None
        experiment1 = Experiments(args.input, args.export_depth, args.npy)
        if len(args.input)>1:
            ROI_coordinates, depth_data = experiment1.get_depth_data()
            print(f'depth_data.shape = {depth_data.shape}, ROI: {ROI_coordinates}')
        else:
            ROI_coordinates, _ = experiment1.get_depth_data()
            print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
