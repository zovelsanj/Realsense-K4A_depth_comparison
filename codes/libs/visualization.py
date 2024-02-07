import cv2 
import copy
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
import seaborn as sns

class Statistics():
    def __init__(self, data, num_x, num_y):
        self.data = data
        if isinstance(num_x, int) and isinstance(num_y, int):   # sometimes num_x and num_y are just integers and sometimes coordinates of ROIs
            self.num_x = np.asarray([0, 0]) 
            self.num_y = np.asarray([num_x, num_y])
        else:
            self.num_x = num_x
            self.num_y = num_y                                                                                         

    def check_negative_depth(self):
        '''input: depth array 
        output: checks for pixels with negative depth values (due to holes) and returns 0 instead'''
        self.data[self.data < 0] = 0
        return self.data[:, self.num_x[1]:self.num_y[1], self.num_x[0]:self.num_y[0]]         #for all data including pallets
        # return self.data[:, self.num_x[1]:self.num_x[1]+50, self.num_x[0]:self.num_x[0]+50]     #for uniformity of pallets

    def z_score_outlier_removal(self, gt=500):
        depth_array = self.check_negative_depth()
        median = np.median(depth_array, axis=0)
        mad = np.median(np.abs(depth_array - median), axis=0) * 1.4826  # scaling factor for consistency with standard deviation
        zero_indices = np.where(mad == 0)
        mad[zero_indices] = 10e-9
        
        z_scores = 0.6745 * (depth_array - median) / mad  # scaling factor for consistency with z-score
        z_score_threshold = 3.5  # Set a threshold for z-scores (3sigma)
        outliers = depth_array[np.abs(z_scores) > z_score_threshold]
        clean_data = depth_array[np.abs(z_scores) <= z_score_threshold]
        if gt is not None:
            clean_data = clean_data[clean_data > (gt - 500)]
        print(f'min data: {np.min(clean_data)}, max data: {np.max(clean_data)}')
        return clean_data, median

    def standard_deviation(self, gt):
        '''input: array of ith pixel in all the depth images
        output: standard deviation of ith pixel depth across the frames'''
        depth_array = self.check_negative_depth()
        sd = np.std(depth_array, axis=0)
        cleaned_data, _ = self.z_score_outlier_removal(gt)
        limiting_sd = np.std(cleaned_data, axis=0)
        return sd, limiting_sd 
    
    def mean(self):
        '''input: array of ith pixel in all the depth images
        output: mean of ith pixel depth across the frames'''
        depth_array = self.check_negative_depth()#*1000
        return np.mean(depth_array, axis=0) #take mean of individual pixels across the frames
    
    def median(self):
        '''input: array of ith pixel in all the depth images
        output: median of ith pixel depth across the frames'''
        depth_array = self.check_negative_depth()
        return np.median(depth_array, axis=0)

class Visualization(Statistics):
    ROIS = []
    start = None
    num_images=0
    pixel_data=None
    def __init__(self, data=None, num_x=50, num_y=50):
        super().__init__(data, num_x, num_y)
        self.end = None
        self.image = None
        self.temp_image = None

    def noise_visualization(self, ax=None, show_window=True, gt=None):
        """Computes per-pixel standard deviation (SD) and mean.
        Also return plot object based on show_window flag"""
        # instance = cls(None, num_x=0, num_y=0)
        sd, _ = super().standard_deviation(gt)
        limiting_sd = 15
        sd[sd > limiting_sd]=0     #replace standard deviation > limiting_sigma by 0 to indicate noise
        mean = super().mean()
        median = super().median()
        if show_window:
            if ax is None:    #for all other plots
                plot = plt.pcolormesh(median)
                plt.xlabel("pixels", fontsize=16)
                plt.ylabel("pixels", fontsize=16)
                plt.tick_params(axis='x', labelsize=12)
                plt.tick_params(axis='y', labelsize=12)

            else:  #for ambience and temperature plots
                plot = ax.pcolormesh(mean)
                ax.set_xlabel("pixels", fontsize=16)
                ax.set_ylabel("pixels", fontsize=16)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
            
            return plot
        else:
            return mean

    def offset_visualization(self, ground_truth):
        '''offset from ground truth distance i.e. random error'''
        ground_truth = ground_truth*1000
        median = super().median() - ground_truth
        plot = plt.pcolormesh(median)
        plt.xlabel("pixels", fontsize=16)
        plt.ylabel("pixels", fontsize=16)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        return plot

    @classmethod
    def check_distribution(cls, data_list, title_list):
        '''Check the distribution of data first and then determine which outlier removal method to use.
        Our data are normally distrubuted, so we are using z-score outlier removal.'''

        labels = [f'{i}' for i in title_list]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
        num_rows, num_cols = cls.get_grid(len(data_list))
        for i, data in enumerate(data_list):
            plt.subplot(num_rows, num_cols, i+1)
            color = colors[i % len(colors)] 
            plt.hist(data.flatten(), bins=30, density=True, edgecolor='black', color=color, label=labels[i])
            mu, std = norm.fit(data)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            
            plt.title(labels[i], fontsize=20)
            plt.xlabel('Distance(mm)', fontsize=16)
            plt.ylabel('Density of Data', fontsize=16)
            # plt.legend(labels=labels)
        plt.show()

    @classmethod
    def violin_plot(cls, data_list, title_list):
        num_rows, num_cols = cls.get_grid(len(data_list))
        fig, axes = plt.subplots(num_rows, num_cols)
        for i, ax in enumerate(axes.flat):
            data = data_list[i]
            data = data[:100, :, :]
            sns.violinplot(data=data, ax=ax, color='red')
            ax.set_xlabel('Distribution of Data', fontsize=16)
            ax.set_ylabel('Distance(mm)', fontsize=16)
            ax.set_title(title_list[i], fontsize=20)

        plt.tight_layout()
        plt.show()

    @classmethod
    def get_grid(cls, num_images):
        num_rows = int(np.sqrt(num_images))
        num_cols = int(np.ceil(num_images/num_rows))
        while num_cols*num_rows - num_images >= num_cols:
            num_rows -= 1
            num_cols = int(np.ceil(num_images/num_rows))
        return num_rows, num_cols
    
    @classmethod
    def subplots(cls, data_list, title_list, suptitle, camera, path, label_list, heatmap, offset=False, check_distribution=False):
        """subplots such as violin_plot, heatmap, offset plot, and temporal plot"""
        print(f'Flag Status: heatmap is {heatmap}, offset is {offset}, check_distribution is {check_distribution}')
        if check_distribution:
            fig = plt.figure(constrained_layout=True)
            plt.suptitle(f"Distribution of Data({camera})", fontsize=20)
            # cls.check_distribution(data_list[1], title_list)
            cls.violin_plot(data_list[1], title_list)
            print('Execute the code with check_distribution=False for other plots')
            return

        fig = plt.figure(constrained_layout=True)
        plt.suptitle(f"{suptitle}({camera})", fontsize=20)
        # plt.suptitle(f"Offset Distances({camera})")
        num_rows, num_cols = cls.get_grid(len(data_list[0]))
        im = None
        print(f'No.of Rows: {num_rows}, No. of Columns: {num_cols}')
        if not offset:
            for i in range(len(data_list[0])):
                plt.subplot(num_rows, num_cols, i+1)
                value_str = title_list[i]
                ground_truth = 0# float(value_str[:-1])   #For camera distance only
                if heatmap:
                    im = data_list[0][i].noise_visualization()
                    cbar = plt.colorbar(im, label=f"distance ({label_list})", format='%.1f')
                    cbar.ax.tick_params(labelsize=12)

                else:
                    print("temporal")
                    cls.temporal_depth_variation(data_list[1][i])
                plt.title(title_list[i], fontsize=20)  
            
        else:
            for i in range(len(data_list[0])):
                plt.subplot(num_rows, num_cols, i+1)
                value_str = title_list[i]
                ground_truth =  float(value_str[:-1]) 
                im = data_list[0][i].offset_visualization(ground_truth)
                cbar = plt.colorbar(im, label=f"distance ({label_list})", format='%.1f')
                cbar.ax.tick_params(labelsize=12)

        if path is not None:
            plt.savefig(path)
        plt.show()
    
    @classmethod
    def uncertainty_plot(cls, data, std):
        circle_colors = ['r', 'g', 'b']
        fig, ax = plt.subplots()
        im = ax.imshow(std, cmap='viridis')
        fig.colorbar(im, label='Standard deviation')
        # Add circles and extract points inside each circle
        circle_points = []
        r_x, r_y = int(data.shape[1]/2), int(data.shape[2]/2)
        # Define the colors for each circle
        circle_radii = (np.arange(3) + 1) * std[r_x, r_y]

        # Compute the distances between each pixel and the center of the circle
        x, y = np.indices(data.shape[:2])
        distances = np.sqrt((x - r_x)**2 + (y - r_y)**2)

        # Define the colors for the circles and data points
        circle_colors = ['r', 'g', 'b']

        # Create the plot
        fig, ax = plt.subplots()

        # Plot the circles and data points within each circle
        circle_points = []
        for i, circle_radius in enumerate(circle_radii):
            circle = Circle((r_x, r_y), circle_radius, fill=False, linewidth=2, edgecolor=circle_colors[i])
            ax.add_patch(circle)
            indices = np.where(distances <= circle_radius)
            points = data[indices[0], indices[1], :].reshape(-1, data.shape[2])
            circle_points.append(points)
            ax.scatter(indices[1], indices[0], s=10, c=circle_colors[i], alpha=0.5)
            
        # Set the title and axis labels
        ax.set_title('Standard Deviation Circles')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Show the plot
        plt.show()

    @classmethod
    def temporal_depth_variation(cls, pixel_data, ground_truth=0):
        """Temporal plot for center pixel or mean/median of all pixels"""
        # plt.plot(np.arange(pixel_data.shape[0]), np.mean(pixel_data, axis=(1,2)))   # mean over each frame for all pixels
        # plt.plot(np.arange(pixel_data.shape[0]), np.mean(pixel_data, axis=(1,2))-ground_truth*1000)   
        plt.plot(np.arange(pixel_data.shape[0]), pixel_data[:, int(pixel_data.shape[1]/2), int(pixel_data.shape[2]/2)])  #center pixel data  
        plt.xlabel("#Frames", fontsize=16)
        plt.ylabel("Distance(mm)", fontsize=16)
        # plt.ylabel("Offset Distance(mm)")
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)


    @classmethod
    def click_and_crop(cls, event, x, y, flags, param):
        """Draw rectangular ROIs with mouse callbacks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            Visualization.start = [x, y]
        elif event == cv2.EVENT_MOUSEMOVE and Visualization.start is not None: # for drawing rectangle as you move the mouse until release
            cls.temp_image = cls.image.copy()
            cls.end = [x, y]
            cv2.rectangle(cls.temp_image, Visualization.start, cls.end,(0, 0, 0), 1)
            cv2.imshow("image", cls.temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            cls.end = [x, y]
            Visualization.ROIS.append([Visualization.start, cls.end])
            cls.image = cls.temp_image.copy()
            Visualization.start = None
            cls.end = None
            
    @classmethod
    def get_cropped_rois(cls, image_path=None):
        """Get data of cropped ROIs using click_and_crop().
        Press 'R' to record data and 'Q' to quit the process."""
        Visualization.ROIS = []
        image = cv2.imread(image_path)
        cls.image = image
        cls.temp_image = cls.image.copy()
            
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", cls.click_and_crop)
        while True:
            cv2.imshow("image", cls.temp_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                cls.temp_image = cls.image.copy()
                Visualization.ROIS = []
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        return Visualization.ROIS
        