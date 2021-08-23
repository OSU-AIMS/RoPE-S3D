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

    def stop(self):
        self.pipeline.stop()

    def get(self):

        depth, color = False, False

        # Wait until a pair is found
        while not depth or not color:
            frames = self.pipeline.wait_for_frames()
            frames_aligned = self.align.process(frames)
            depth = frames_aligned.get_depth_frame()
            color = frames_aligned.get_color_frame()

        # Return color, depth frame
        return np.array(color.get_data()), np.array(depth.get_data(),dtype=float) * self.depth_scale

