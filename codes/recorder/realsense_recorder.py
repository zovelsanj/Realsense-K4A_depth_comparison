# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join
import os
import shutil
import json
from enum import IntEnum

try:
    # Python 2 compatible
    input = raw_input
except NameError:
    pass


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            return


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

def set_exposure(rs, pipeline, sensor, exp):
    # Query minimum and maximum supported values
    max_exp = sensor.get_option_range(rs.option.exposure).max
    min_exp = sensor.get_option_range(rs.option.exposure).min
    print(max_exp, min_exp)
    sensor.set_option(rs.option.exposure, exp)
    # After applying new exposure value its recommended to wait for several frames to make sure the change takes effect
    # Alternatively, the user can know when exposure was changed using per-frame metadata
    
    # for j in range(1, 10):
    #     frames = pipeline.wait_for_frames()
    #     ir = frames.get_infrared_frame(1)
    #     if ir.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
    #         if ir.get_frame_metadata(rs.frame_metadata_value.actual_exposure) == min_exp:
    #             break # Exposure change took place, no need to keep waiting

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")
    parser.add_argument("--output_folder",
                        default='data/realsense/',
                        help="set output folder")
    parser.add_argument("--record_rosbag",
                        action='store_true',
                        required=False,
                        help="Recording rgbd stream into realsense.bag")
    parser.add_argument(
        "--record_imgs",
        action='store_true',
        help="Recording save color and depth images into realsense folder")
    parser.add_argument("--playback_rosbag",
                        action='store_true',
                        required=False,
                        help="Play recorded realsense.bag file")
    parser.add_argument("--iteration",
                        required=True,
                        type=str,
                        help="kth iteration for scene capture")
    parser.add_argument("--len",
                        required=True,
                        type=int,
                        help="capture length in secs")
    parser.add_argument("--config",
                        required=True,
                        type=str,
                        help="path to config file")
    parser.add_argument("--npy",
                        required=False,
                        type=str,
                        help="path to npy file")
    parser.add_argument("--start",
                        required=False,
                        type=int,
                        nargs='+',
                        default=None,
                        help="starting corner of checkerboard (in pixels)")
    parser.add_argument("--end",
                        required=False,
                        type=int,
                        nargs='+',
                        default=None,
                        help="end corner of checkerboard (in pixels)")
    parser.add_argument("--exp",
                        type=float,
                        required=False,
                        default=None,
                        help="change the exposure")

    args = parser.parse_args()

    # if sum(o is not False for o in vars(args).values()) != 2:
    #     parser.print_help()
    #     exit()

    path_output = args.output_folder
    path_iteration = os.path.join(path_output, args.iteration)
    path_depth = join(path_iteration, "depth")
    path_color = join(path_iteration, "color")
    if args.record_imgs:
        make_clean_folder(path_output)
        if not os.path.exists(path_iteration):
            make_clean_folder(path_iteration)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    path_bag = join(args.output_folder, "realsense.bag")
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    f = open(args.config)
    custom_config = json.load(f)

    if args.record_imgs or args.record_rosbag:
        # note: using 640 x 480 depth resolution produces smooth depth boundaries
        #       using rs.format.bgr8 for color image format for OpenCV based image visualization
        config.enable_stream(rs.stream.depth, custom_config["resolution"][0], custom_config["resolution"][1], rs.format.z16, custom_config["camera_fps"])
        config.enable_stream(rs.stream.color, custom_config["resolution"][0], custom_config["resolution"][1], rs.format.bgr8, custom_config["camera_fps"])
        if args.record_rosbag:
            config.enable_record_to_file(path_bag)
    if args.playback_rosbag:
        config.enable_device_from_file(path_bag, repeat_playback=True)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    if args.exp is not None:
        set_exposure(rs, pipeline, depth_sensor, args.exp)

    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    if args.start is None:
        x1 = int(custom_config["resolution"][0]/2)
        y1 = int(custom_config["resolution"][1]/2)
        x2 = x1+100
        y2 = y1+100
    else:
        x1, y1 = args.start
        x2, y2 = args.end
    c = abs(x2 - x1)
    r = abs(y2 - y1)

    frame_count = 0
    dist = np.zeros((custom_config["camera_fps"] * args.len, r, c))
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            roi_depth_values = np.ones((r,c))
            
            for i in range(r):
                for j in range(c):
                    pixel_value = aligned_depth_frame.get_distance(x1+j, y1+i)    # gives distance in meters
                    # print(x+i, y+j) 
                    pixel_value = np.float32(pixel_value) #convert to float32 as loading float64 can be resource extensive in case of large npy files
                    roi_depth_values[i][j] = pixel_value+0.0042  #add the depth camera location i.e. |-4.2mm| from the front glass face

            dist[frame_count-1] = roi_depth_values
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | \
                    (depth_image_3d <= 0), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            if args.record_imgs:
                if frame_count == 0:
                    save_intrinsic_as_json(
                        join(args.output_folder, "camera_intrinsic.json"),
                        color_frame)
                cv2.imwrite("%s/%06d.png" % \
                        (path_depth, frame_count), depth_colormap)
                cv2.imwrite("%s/%06d.jpg" % \
                        (path_color, frame_count), color_image)
                print(frame_count)
                frame_count += 1

            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
            if frame_count > custom_config["camera_fps"] * args.len:
                    break
        np.save(os.path.join(path_iteration, args.npy), dist)     
        print(f"Saved {args.npy} at {path_iteration}")
    finally:
        pipeline.stop()
