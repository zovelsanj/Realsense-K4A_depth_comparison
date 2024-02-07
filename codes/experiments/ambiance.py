import numpy as np 
import argparse
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from libs.experiments import Experiments
from libs.visualization import Visualization

def subplots(data_list, title, camera, label_list, path, heatmap=True):
    num_rows, num_cols = Visualization.get_grid(len(data_list))
    print(f"Rows: {num_rows}, Columns: {num_cols}")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Per-Pixel Mean Distance Distribution')
    subfigs = fig.subfigures(nrows=num_rows, ncols=1)

    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"({camera[row]})", fontsize=12)
        axs = subfig.subplots(nrows=1, ncols=num_cols)
        for col, ax in enumerate(axs):
            if heatmap:
                im = data_list[col+len(axs)*row].noise_visualization(ax=ax)
                cbar = fig.colorbar(im, ax=ax,label=f"distance ({label_list[row]})")
                cbar.ax.tick_params(labelsize=8)
            else:
                Visualization.temporal_depth_variation(data_list[i])
            if row==0:
                ax.set_title(title[col])      
    if path is not None:
        plt.savefig(path)
    plt.show()

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
    parser.add_argument("--fig_path",
                        required=False,
                        help="path to save plots")

    args = parser.parse_args()

    obj_list = []
    # title_list = ["Indoor Ambient", "Indoor Controlled", "Outdoor Shadow", "Outdoor Sunlight"]
    title_list = ["Controlled Environment", "Uncontrolled Environment"]
    if len(args.npy)>1:
        for i,file_name in enumerate(args.npy):
            data=np.load(file_name)
            if i<len(title_list):   #pass RealSense .npy files first then AzureKinect
                data=(data+0.0082)*1000
            obj_list.append(Visualization(data,data.shape[1],data.shape[2]))
        # print(len(obj_list))
        subplots(obj_list, title_list, camera=["RealSense","AzureKinect"],label_list=['mm','mm'], path=args.fig_path)

    else:
        ROI_coordinates = None
        experiment5 = Experiments(args.input, args.export_depth, args.npy)
        if len(args.input)>1:
            ROI_coordinates, depth_data = experiment5.get_depth_data()
            print(f'depth_data.shape = {depth_data.shape}, ROI: {ROI_coordinates}')
        else:
            ROI_coordinates, _ = experiment5.get_depth_data()
            print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
