import numpy as np
import argparse 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from libs.experiments import Experiments
from libs.visualization import Visualization

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
                        nargs="*",
                        help="npy filename to export data or npy path to read data")
    parser.add_argument("--export_depth",
                        action='store_true',
                        help="option to view bounding box")
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
        for file_name in args.npy:
            data=np.load(file_name)
            if args.camera_id==0: #for RealSense
                data = (data + (0.0084))*1000   #in case not added in realsense_recorder.py
            object_list.append(Visualization(data,data.shape[1],data.shape[2]))
            data_list.append(data)

        obj_list = [object_list, data_list]
        title_list = ["Static", "Motion"]
        camera=["RealSense","AzureKinect"]
        label_list='mm'
        suptitle = "Effect of Motion"

        Visualization.subplots(obj_list, title_list, suptitle, camera=camera[args.camera_id], path=None, label_list=label_list, heatmap=False)    
    else:
        ROI_coordinates = None
        experiment2 = Experiments(args.input, args.export_depth, args.npy)
        if len(args.input)>1:
            ROI_coordinates, depth_data = experiment2.get_depth_data()
            print(f'depth_data.shape = {depth_data.shape}, ROI: {ROI_coordinates}')
        else:
            ROI_coordinates, _ = experiment2.get_depth_data()
            print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
