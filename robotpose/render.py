# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import numpy as np

import cv2
import PySimpleGUI as sg

from .data import Dataset

from .simulation import DatasetRenderer


class Aligner():
    """
    Used to manually find the position of camera relative to robot.

    W/S - Move forward/backward
    A/D - Move left/right
    Z/X - Move down/up
    Q/E - Roll
    R/F - Tilt down/up
    G/H - Pan left/right
    +/- - Increase/Decrease Step size
    """

    def __init__(self, dataset, start_idx = None, end_idx = None):
        # Load dataset
        self.ds = Dataset(dataset, permissions='a')

        self.renderer = DatasetRenderer(dataset, None, mode='seg_full')

        if start_idx is not None:
            self.start_idx = start_idx
        else:
            self.start_idx = 0

        self.idx = self.start_idx

        if end_idx is not None:
            self.end_idx = end_idx
        else:
            self.end_idx = self.ds.length - 1

        self.inc = int((self.end_idx - self.start_idx + 1)/20)
        if self.inc > 10:
            self.inc = 10
        if self.inc < 1:
            self.inc = 1

        self.c_pose = self.ds.camera_pose[self.start_idx]

        # Movement steps
        self.xyz_steps = [.001,.005,.01,.05,.1,.25,.5]
        self.ang_steps = [.0005,.001,.005,.01,.025,.05,.1]
        self.step_loc = len(self.xyz_steps) - 4

        self._findSections()
        self.section_idx = 0

        print("Copying Image Array...")
        self.real_arr = np.copy(self.ds.og_img)

        self.gui = AlignerGUI()


    def run(self):
        ret = True
        move = True

        while ret:
            event, values = self.gui.update(self.section_starts, self.section_idx)
            if event == 'quit':
                print("Quit by user.")
                ret = False
                continue
            elif event == 'new_section':
                self._newSection(values)
            elif event == 'goto':
                if 0 <= self.idx and self.idx < self.ds.length:
                    self.idx = values
                    move = True

            self._getSection()
            self.readCameraPose()

            if move:
                real = self.real_arr[self.idx]
                self.renderer.setPosesFromDS(self.idx)
                render, depth = self.renderer.render()
                image = self.combineImages(real, render)
                move = False
            image = self.addOverlay(image)
            cv2.imshow("Aligner", image)

            inp = cv2.waitKey(1)
            ret, move = self.moveCamera(inp)

        self.gui.close()
        cv2.destroyAllWindows()


    def moveCamera(self,inp):
        xyz_step = self.xyz_steps[self.step_loc]
        ang_step = self.ang_steps[self.step_loc]

        if inp == ord('0'):
            return False, False
        elif inp == ord('='):
            self.step_loc += 1
            if self.step_loc >= len(self.xyz_steps):
                self.step_loc = len(self.xyz_steps) - 1
            return True, False
        elif inp == ord('-'):
            self.step_loc -= 1
            if self.step_loc < 0:
                self.step_loc = 0
            return True, False
        elif inp == ord('k'):
            self.increment(-self.inc)
            return True, True
        elif inp == ord('l'):
            self.increment(self.inc)
            return True, True

        if inp == ord('d'):
            self.c_pose[0] -= xyz_step
        elif inp == ord('a'):
            self.c_pose[0] += xyz_step
        elif inp == ord('w'):
            self.c_pose[1] -= xyz_step
        elif inp == ord('s'):
            self.c_pose[1] += xyz_step
        elif inp == ord('z'):
            self.c_pose[2] += xyz_step
        elif inp == ord('x'):
            self.c_pose[2] -= xyz_step
        elif inp == ord('q'):
            self.c_pose[3] -= ang_step
        elif inp == ord('e'):
            self.c_pose[3] += ang_step
        elif inp == ord('r'):
            self.c_pose[4] -= ang_step
        elif inp == ord('f'):
            self.c_pose[4] += ang_step
        elif inp == ord('g'):
            self.c_pose[5] += ang_step
        elif inp == ord('h'):
            self.c_pose[5] -= ang_step

        self.saveCameraPose()
        return True, True


    def addOverlay(self, image):
        pose_str = "["
        for num in self.c_pose:
            pose_str += f"{num:.3f}, "
        pose_str +="]"
        image = cv2.putText(image, pose_str,(10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.xyz_steps[self.step_loc]),(10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.ang_steps[self.step_loc]),(10,150), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.idx),(10,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return image


    def combineImages(self,image_a, image_b, weight = .5):
        return np.array(image_a * weight + image_b *(1-weight), dtype=np.uint8)


    def increment(self, step):
        if (self.idx + step >= 0) and (self.idx + step < self.ds.length):
            self.idx += step


    def saveCameraPose(self):
        for idx in range(self.start_idx, self.end_idx + 1):
            self.ds.camera_pose[idx,:] = self.c_pose

    def readCameraPose(self):
        self.c_pose = self.ds.camera_pose[self.idx,:]

    def _findSections(self):
        self.section_starts = []
        p = [0,0,0,0,0,0]
        for idx in range(self.ds.length):
            if not np.array_equal(self.ds.camera_pose[idx], p):
                self.section_starts.append(idx)
                p = self.ds.camera_pose[idx,:]
        self.section_starts.append(self.ds.length)
        self.section_starts

    def _newSection(self, idx):
        self.section_starts.append(idx)
        self.section_starts.sort()

    def _getSection(self):
        section_start = max([x for x in self.section_starts if x <= self.idx])
        self.section_idx = self.section_starts.index(section_start)
        self.start_idx = section_start
        self.end_idx = self.section_starts[self.section_idx + 1] - 1



class AlignerGUI():

    def __init__(self):
        control_str = "W/S - Move forward/backward\n"+\
            "A/D - Move left/right\nZ/X - Move up/down\nQ/E - Roll\n"+\
            "R/F - Tilt down/up\nG/H - Pan left/right\n+/- - Increase/Decrease Step size\nK/L - Last/Next image"
        self.layout = [[sg.Text("Currently Editing:"), sg.Text(size=(40,1), key='editing')],
                        [sg.Input(size=(5,1),key='num_input'),sg.Button('Go To',key='num_goto'), sg.Button('New Section',key='new_section')],
                        [sg.Text("",key='warn',text_color="red", size=(22,1))],
                        [sg.Table([[["Sections:"]],[[1,1]]], key='sections'),sg.Text(control_str)],
                        [sg.Button('Quit',key='quit')]]

        self.window = sg.Window('Aligner Controls', self.layout, return_keyboard_events = True, use_default_focus=False)

    def update(self, section_starts, section_idx):
        event, values = self.window.read(timeout=1, timeout_key='tm')
        section_table = []
        for idx in range(len(section_starts)-1):
            section_table.append([[f"{section_starts[idx]} - {section_starts[idx+1]-1}"]])

        self.window['editing'].update(f"{section_starts[section_idx]} - {section_starts[section_idx+1]-1}")
        self.window['sections'].update(section_table)

        try:
            if event == 'new_section':
                return ['new_section',int(values['num_input'])]
            elif event == 'num_goto':
                return ['goto',int(values['num_input'])]
            if len(values['num_input']) > 0:
                if int(values['num_input']) is not None:
                    self.window['warn'].update("")
        except ValueError:
            self.window['warn'].update("Please input a number.")

        if event == 'quit':
            self.close()
            return ['quit',None]

        return [None,None]

    def close(self):
        self.window.close()
