# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os

import cv2
import numpy as np
import PySimpleGUI as sg

from .data import Dataset, DatasetInfo
from .simulation import Aligner, Renderer
from .urdf import URDFReader
from .utils import expandRegion


class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.urdf_reader = URDFReader()
        self.valid_urdf = self.urdf_reader.internal_path != None

        urdf_menu = [
            [sg.Txt("Current URDF:")],
            [sg.Txt(self.urdf_reader.internal_path,key='-current_urdf-'),
                sg.Button("Change",key='-browse_urdf-',tooltip='Select URDF path')]
        ]

        dataset_menu = [
            [sg.Txt("Dataset:"),sg.Combo(self.compiled_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-details-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')]]

        self.layout = [
            [sg.Frame("URDF Options", urdf_menu)],
            [sg.Frame("Dataset Options", dataset_menu)],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Dataset Wizard')]
            ]


    def run(self):
        self.window = sg.Window('Dataset Wizard', self.layout.copy(), finalize=True)
        self.window.bring_to_front()

        event = ''
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(10)
            if event not in (sg.WIN_CLOSED,'-quit-'):
                self._updateButtons(values)
                self._runEvent(event, values)

        self.window.close()


    def _updateButtons(self,values):

        if values['-dataset-'] in self.unique_sets():
            for button in ['-details-','-align-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-details-','-align-']:
                self.window[button].update(disabled = True)

        if not self.valid_urdf:
            for button in ['-align-']:
                self.window[button].update(disabled = True)
                

    def _runEvent(self,event,values):
        if event =='-align-':
            self._runAligner(values['-dataset-'])
        elif event == '-details-':
            self._showDetails(values['-dataset-'])
        elif event == '-edit_skele-':
            self._runKeypointWizard(values['-skeleton-'])
        elif event == '-browse_urdf-':
            self._changeURDF()


    def _changeURDF(self):
        self.valid_urdf = False
        path = sg.popup_get_file("Select new URDF",
            title="URDF Selection",
            file_types=(("URDF Files", ".urdf"),), 
            initial_folder=os.getcwd())
        if path is not None:
            if os.path.isfile(path) and path.endswith('.urdf'):
                path = os.path.relpath(path, os.path.commonprefix([path,os.getcwd()]))
                self.urdf_reader.path = path.replace('\\','/')
                self.valid_urdf = True
            else:
                sg.popup_ok("Error:","Invalid URDF file selection.")
        self.window['-current_urdf-'].update(self.urdf_reader.path)
           

    def _showDetails(self, dataset):
        ds = Dataset(dataset)
        sg.popup_ok(str(ds), title=f"{dataset} Details")

    def _runAligner(self, dataset):
        print(f'Aligning {dataset}')
        align = Aligner(dataset)
        align.run()
        print(f'Alignment Complete')

    def _runMeshWizard(self):
        self.window.disable()
        self.window.disappear()
        wiz = MeshWizard()
        wiz.run()
        cv2.destroyAllWindows()
        self.window.enable()
        self.window.reappear()
        self.window.bring_to_front()



ANGLES = ['-pi','-pi/2','0','pi/2','pi']
ANGLE_DICT = {'-pi':-np.pi,'-pi/2':-np.pi/2,'0':0,'pi/2':np.pi/2,'pi':np.pi}

class MeshWizard():

    def __init__(self):

        self.rend = Renderer(suppress_warnings=True)
        self.base_pose = [1.5,-1.5,.35, 0,np.pi/2,0]

        self.u_reader = URDFReader()
        lims = self.u_reader.joint_limits
        lims *= (180/np.pi) 
        lims = np.round(lims)

        self.rend.setJointAngles([0,0,0,0,0,0])

        self.mode = 'seg'
        self.crop = True
        self.use_cv = False
        self.last_joint_selected = ''

        def jointSlider(name, lower, upper):
            return [sg.Txt(f"{name}:"),
                sg.Slider(range=(lower, upper),
                    orientation='h', tick_interval=90, 
                    size=(20, 20), default_value=0, key=f'-{name}-')]


        render_modes = [
            [sg.Radio("Segmented Joints","render",default=True, key="-render_seg-")],
            [sg.Radio("Realistic Metallic","render", key="-render_real-")],
            [sg.Checkbox("Crop To Fit", default=True,key='-crop-')],
            [sg.Checkbox("Display in New Window", default=False, key='-disp_cv-')],
            [sg.Checkbox("Highlight Selected", default=True, key='-highlight-')]
        ]

        column1 = [
            [sg.Frame('View Settings',[
                [sg.Slider(range=(-30, 30), orientation='v', size=(5, 20), default_value=0,key='-vert_slider-'),
                    sg.VerticalSeparator(),
                    sg.Column(render_modes)],
                [sg.Slider(range=(-180, 180), orientation='h', size=(20, 20), default_value=0, key='-horiz_slider-'),
                    sg.Button("Reset",key='-view_reset-')]
            ]
            )],
            [sg.Frame('Robot Joints',[
                jointSlider("S",*lims[0]),
                jointSlider("L",*lims[1]),
                jointSlider("U",*lims[2]),
                jointSlider("R",*lims[3]),
                jointSlider("B",*lims[4]),
                [sg.Button("Reset",key='-joint_reset-')]
            ]
            )]
            ]


        view_column = [
            [sg.Frame('Preview',[[sg.Image(key='-preview_img-',pad=(2,2))]], pad=(1,1))]
        ]

        self.layout = [          
            [sg.Column(column1),sg.Column(view_column,key='-preview_column-',pad=(1,1))],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Skeleton Wizard')],
            ]
        

    
    def run(self):
        self.window = sg.Window('Skeleton Wizard', self.layout)
        event, values = self.window.read(1)
        self.window.bring_to_front()

        event = ''
        prev_values = {}
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(5, timeout_key='TIMEOUT')
            if event not in (sg.WIN_CLOSED,'-quit-'):
                if event != 'TIMEOUT' or values != prev_values:
                    self._updateInputs(values)
                    self._runEvent(event, values)
                    self._setRotation(values)
                    self._setJointAngles(values)
                    self._setViewMode(values)
                    self.show(self.render())
                    prev_values = values

        self.window.close()


    def _updateInputs(self, values):
        self.window['-preview_column-'].update(visible=(not values['-disp_cv-']))


    def _runEvent(self, event, values):
        self.crop = values['-crop-']
        self.use_cv = values['-disp_cv-']

        if event == '-view_reset-':
            self._resetRotation()
        if event == '-joint_reset-':
            self._resetJointAngles()


    def _setViewMode(self, values):
        modes = ['seg','real']
        mode_keys = ['-render_seg-','-render_real-']
        mode = [modes[x] for x in range(len(modes)) if values[mode_keys[x]]][0]
        if self.mode == mode:
            return
        else:
            self.rend.setMode(mode)
            self.mode = mode



    def _resetRotation(self):
        self._setRotation({'-horiz_slider-':0,'-vert_slider-':0})
        for slider in ['-horiz_slider-','-vert_slider-']:
            self.window[slider].update(0)

    def _setRotation(self, values):
        rotation_h = values['-horiz_slider-']
        rotation_v = values['-vert_slider-']
        self.rotation_h = (rotation_h/180) * np.pi
        self.rotation_v = (rotation_v/180) * np.pi

        self.c_pose = np.copy(self.base_pose)
        self.c_pose[1] *= (1 - np.sin(self.rotation_v) * np.tan(self.rotation_v)) * np.cos(self.rotation_h)
        self.c_pose[0] *= np.sin(self.rotation_h)
        self.c_pose[2] = np.sin(self.rotation_v) * 1 + .15
        self.c_pose[4] = np.pi/2 - self.rotation_v
        self.c_pose[5] = self.rotation_h
        self.rend.setCameraPose(self.c_pose)


    def _resetJointAngles(self):
        self.rend.setJointAngles([0,0,0,0,0,0])
        for joint in ['-S-','-L-','-U-','-R-','-B-']:
            self.window[joint].update(0)

    def _setJointAngles(self, values):
        joint_angles = [0,0,0,0,0,0]
        for joint, idx in zip(['-S-','-L-','-U-','-R-','-B-'], range(5)):
            joint_angles[idx] = values[joint] * np.pi/180 

        self.rend.setJointAngles(joint_angles)

    def render(self):
        color, depth = self.rend.render()
        if self.crop:
            color = self._cropImage(color)
        return color
    
    def show(self, image):
        if self.use_cv:
            cv2.imshow("Mesh Wizard",image)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            imgbytes = cv2.imencode('.png', image)[1].tobytes()
            self.window['-preview_img-'].update(data=imgbytes)

    def _cropImage(self, image, pad = 10):
        # Aim for 700 w by 720 h
        occupied = np.any(image,-1)
        rows, columns = np.where(occupied)
        min_row = max(0,min(rows)-pad)
        max_row = min(image.shape[0]-1,max(rows)+pad)
        min_col = max(0,min(columns)-pad)
        max_col = min(image.shape[1]-1,max(columns)+pad)
        while max_col - min_col < 699:
            if max_col < image.shape[1]:
                max_col +=1
            if min_col > 0:
                min_col -=1
        while max_row - min_row < 705:
            if max_row < image.shape[0]:
                max_row +=1
            if min_row > 0:
                min_row -=1
        return image[min_row:max_row,min_col:max_col]


