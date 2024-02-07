## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import argparse
import numpy as np
import cv2
import os
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

def write_video(filename, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    # path = '/media/suruchi/ba37ba79-2116-4c22-84f9-81ed294628fa/suruchi/Documents/venv_realsense/'
    path = "./images/"

    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, "color"))
        os.mkdir(os.path.join(path, "depth"))

    w, h,_= frames[0][0].shape
    print("Writing frames")
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
    for i in range(len(frames)):
        cv2.imwrite(os.path.join(os.path.join(path, "color"), str(i)+".png"), frames[i][0])
        cv2.imwrite(os.path.join(os.path.join(path, "depth"), str(i)+".png"), frames[i][1])
        # writer.write(frames[i])
    
    writer.release() 
# Start streaming
pipeline.start(config)


def main(video_length):
    frames_to_write = []
    try:
        while True:
            align = rs.align(rs.stream.color)
            
            # Wait for a coherent pair of frames: depth and color
            frameset = pipeline.wait_for_frames()
            frames = align.process(frameset)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            gray_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
                
            else:
                images = np.hstack((color_image, depth_colormap))
            frames_to_write.append([images[:, :1280, :], images[:, 1280:, :]])
            print(color_image.shape, depth_colormap.shape, images.shape)

            # if cv2.waitKey(33) == 27:
            if len(frames_to_write) == 30*video_length:
                cv2.destroyAllWindows()
                return frames_to_write
                
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, required=True, help="length of video")
    args = parser.parse_args()

    frames_written = []
    frames_written = main(args.len)
    write_video("output.mkv", frames_written, 30)


