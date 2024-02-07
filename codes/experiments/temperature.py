import numpy as np 
import argparse
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from libs.experiments import Experiments
from libs.visualization import Visualization

def subplots(data_list, title, camera, label_list, path, heatmap):
    num_rows, num_cols = Visualization.get_grid(len(data_list[0]))
    print(num_rows, num_cols)
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Influence of Integration Time')
    subfigs = fig.subfigures(nrows=num_rows, ncols=1)
    
    for row, subfig in enumerate(subfigs):
        # subfig.suptitle(f"{camera[row]}", fontsize=12)
        axs = subfig.subplots(nrows=1, ncols=num_cols)
        for col, ax in enumerate(axs):
            if col+len(axs)*row >= len(title_list): return
            if heatmap:
                im = data_list[0][col+len(axs)*row].noise_visualization(ax=ax)
                cbar = fig.colorbar(im, ax=ax,label=f"distance ({label_list[row]})")
                cbar.ax.tick_params(labelsize=8)
            else:
                # print(data_list[1][col+len(axs)*row].shape)
                Visualization.temporal_depth_variation(data_list[1][col+len(axs)*row])
            ax.set_title(title[col+len(axs)*row])   
   
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
                        help="npy filename only to export data or npy path to load data")
    parser.add_argument("--export_depth",
                        action='store_true',
                        help="option to export depth data after selection ROIs")
    parser.add_argument("--fig_path",
                        required=False,
                        help="path to save plots")
    parser.add_argument("--camera_id",
                            type=int,
                            help="RealSense = 0 or AzureKinect = 1")
    args = parser.parse_args()

    object_list = []
    data_list = []
    # title_list = ["20$^{o}C$", "25$^{o}C$", "30$^{o}C$"]
    title_list = ["5$\mu$s", "25$\mu$s", "50$\mu$s", "100$\mu$s", "250$\mu$s", "500$\mu$s", "1ms", "1.5ms", "2.5ms", "5ms", "10ms", "50ms", "100ms", "500ms", "1000ms"]
    camera=["RealSense", "AzureKinect"]
    if len(args.npy)>1:
        for i,file_name in enumerate(args.npy):
            data=np.load(file_name)
            if args.camera_id==0:
                data=(data+0.0082)*1000   
            data_list.append(data)
            object_list.append(Visualization(data,data.shape[1],data.shape[2]))
        
        obj_list = [object_list, data_list]
        subplots(obj_list, title_list, camera[args.camera_id],label_list=['mm','mm', 'mm'], path=args.fig_path, heatmap=False)
        # sd = obj_list[0].noise_visualization(show_window=False)
        # Visualization.uncertainty_plot(data_list[0], sd)

    else:
        ROI_coordinates = None
        experiment6 = Experiments(args.input, args.export_depth, args.npy)
        if len(args.input)>1:
            ROI_coordinates, depth_data = experiment6.get_depth_data()
            print(f'depth_data.shape = {depth_data.shape}, ROI: {ROI_coordinates}')
        else:
            ROI_coordinates, _ = experiment6.get_depth_data()
            print(f'Starting and ending coordinates of ROI: {ROI_coordinates}')
