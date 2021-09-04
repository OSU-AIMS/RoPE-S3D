# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import pyrealsense2 as rs
import numpy as np


class LiveCamera():

    def __init__(self, width = 1280, height = 720, fps = 30) -> None:

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)    # Enable Depth
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)    # Enable Color

        self.align = rs.align(rs.stream.color)

    def start(self):
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        opt = rs.option
        # Define Filters
        self.deci_filter = rs.decimation_filter()
        self.deci_filter.set_option(opt.filter_magnitude, 2) #default value (2)

        self.spat_filter = rs.spatial_filter()
        #default values (2,0.5,20,0)
        self.spat_filter.set_option(opt.filter_magnitude, 2)
        self.spat_filter.set_option(opt.filter_smooth_alpha, 0.5)
        self.spat_filter.set_option(opt.filter_smooth_delta, 20)
        self.spat_filter.set_option(opt.holes_fill, 0)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(opt.filter_smooth_alpha, 0.5)

    def stop(self):
        self.pipeline.stop()

    def filter(self, frames):
        frames_filtered = self.deci_filter.process(frames).as_frameset()
        frames_filtered = self.spat_filter.process(frames_filtered).as_frameset()
        frames_filtered = self.temporal_filter.process(frames_filtered).as_frameset()
        return frames_filtered

    def get(self):

        depth, color = False, False

        # Wait until a pair is found
        while not depth or not color:
            frames = self.pipeline.wait_for_frames()
            filtered = self.filter(frames)
            frames_aligned = self.align.process(filtered)
            depth = frames_aligned.get_depth_frame()
            color = frames_aligned.get_color_frame()

        # Return color, depth frame
        return np.array(color.get_data()), np.array(depth.get_data(),dtype=float) * self.depth_scale

    def get_average(self):

        num = 20

        depth, color = False, False

        # Wait until a pair is found
        while not depth or not color:
            frames = self.pipeline.wait_for_frames()
            filtered = self.filter(frames)
            frames_aligned = self.align.process(filtered)
            depth = frames_aligned.get_depth_frame()
            color = frames_aligned.get_color_frame()

        depth_sum = np.array(depth.get_data())

        for i in range(num - 1):
            depth = False
            while not depth:
                frames = self.pipeline.wait_for_frames()
                filtered = self.filter(frames)
                frames_aligned = self.align.process(filtered)
                depth = frames_aligned.get_depth_frame()
            depth_sum += np.array(depth.get_data())

        

        # Return color, depth frame
        return np.array(color.get_data()), depth_sum * self.depth_scale / num

