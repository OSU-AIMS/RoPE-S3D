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

from deepposekit import Annotator
import PySimpleGUI as sg

from .data import DatasetInfo, Dataset
from .render import Aligner
from .skeleton import Skeleton, SkeletonInfo
from .simulation import SkeletonRenderer


class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.skinf = SkeletonInfo()


        dataset_menu = [
            [sg.Text("Dataset:"),sg.InputCombo(self.compiled_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-details-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')]]

        keypoint_menu = [
            [sg.Text("Keypoint Skeleton:"),
                sg.InputCombo([x for x in self.skinf.valid() if x != 'BASE'],key='-skeleton-', size=(10, 1))],
            [sg.Button("Edit",key='-edit_skele-',tooltip='Edit Skeleton with Skeleton Wizard')],
            [sg.Button("View Annotations",key='-manual_annotate-', disabled=True)],
            [sg.Button("Create New Skeleton",key='-new_skele-'),sg.Button("Finish Skeleton Creation",key='-finish_skele-',visible=False)]
            ]

        self.layout = [
            [sg.Frame("Dataset Options", dataset_menu)],
            [sg.Frame("Keypoint Options", keypoint_menu)],
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
            if values['-skeleton-'] in self.skinf.valid():
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = False)
            else:
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = True)
        else:
            for button in ['-details-','-align-']:
                self.window[button].update(disabled = True)

        if values['-skeleton-'] in self.skinf.valid():
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = True)
        
        if self.skinf.num_incomplete() > 0:
            self.window['-finish_skele-'].update(visible = True)
        else:
            self.window['-finish_skele-'].update(visible = False)
                


    def _runEvent(self,event,values):
        if event =='-align-':
            self._runAligner(values['-dataset-'])
        elif event == '-details-':
            self._showDetails(values['-dataset-'])
        elif event == '-manual_annotate-':
            self._manualAnnotate(values['-dataset-'], values['-skeleton-'])
        elif event == '-edit_skele-':
            self._runKeypointWizard(values['-skeleton-'])
        elif event == '-new_skele-':
            self._makeNewSkeleton()
        elif event == '-finish_skele-':
            self._finishSkeleton()

           
    def _showDetails(self, dataset):
        ds = Dataset(dataset)
        sg.popup_ok(str(ds), title=f"{dataset} Details")


    def _finishSkeleton(self):
        layout = [[sg.Text("Skeleton To Finish:"),
                sg.InputCombo(self.skinf.incomplete(),key='-skeleton-', size=(10, 1))],
                [sg.Submit(),sg.Cancel()]]
        window = sg.Window("Finish Skeleton",layout)
        event, values = window.read()
        window.close()
        if event != 'Submit':
            return
        else:
            skele = Skeleton(values['-skeleton-'], create=True)
            sg.popup_ok(f"Please edit the skeleton JSON to include keypoint positions along with angle prediction relations.",
                "This can be done completely manually or by selecting the skeleton and clicking 'Edit' in the wizard.",
                )

        


    def _makeNewSkeleton(self):
        name = sg.popup_get_text("Name for new keypoint skeleton:",title='Create Keypoint Skeleton')

        if name is not None:
            name = name.upper()
            confirm = sg.popup_ok_cancel(f"Create new skeleton with name {name} ?")
            if confirm == 'OK':
                path = self.skinf.create_csv(name)
                sg.popup_ok(f"Please edit {path} to include the needed keypoints along with their parent keypoints.",
                "Then return and select 'Finish Skeleton Creation' to create a JSON config file for the skeleton.",
                "More keypoints may be added later (by editing the skeleton), but it is suggested to add them now to save time"
                )


    def _manualAnnotate(self,dataset,skeleton):
        ds = Dataset(dataset,skeleton)
        ds.makeDeepPoseDS()
        app = Annotator(datapath=os.path.abspath(ds.deepposeds_path),
                dataset='images',
                skeleton=ds.skele.csv_path,
                shuffle_colors=False,
                text_scale=1)

        app.run()


    def _runAligner(self, dataset):
        print(f'Aligning {dataset}')
        align = Aligner(dataset)
        align.run()
        print(f'Alignment Complete')

    def _runKeypointWizard(self, skeleton):
        self.window.disable()
        self.window.disappear()
        wiz = SkeletonWizard(skeleton)
        wiz.run()
        cv2.destroyAllWindows()
        self.window.enable()
        self.window.reappear()
        self.window.bring_to_front()




class SkeletonWizard(Skeleton):

    def __init__(self, name):
        super().__init__(name)

        self.rend = SkeletonRenderer(name, suppress_warnings=True)
        self.base_pose = [1.5,-1.5,.35, 0,np.pi/2,0]
        self._setRotation(0,0)
        self.mode = 0

        self.rend.setJointAngles([0,0,0,0,0,0])

        self.current_mode = 'key'

        def jointSlider(name, lower, upper):
            return [sg.Text(f"{name}:"),
                sg.Slider(range=(lower, upper),
                    orientation='h', tick_interval=90, 
                    size=(20, 20), default_value=0, key=f'-{name}-')]


        render_modes = [
            [sg.Radio("Keypoint","render", default=True, key="-render_key-")],
            [sg.Radio("Segmented Joints","render", key="-render_seg-")],
            [sg.Radio("Segmented Body","render", key="-render_body-")],
            [sg.Radio("Realistic Metallic","render", key="-render_real-")]
        ]


        column1 = [
            [sg.Frame('View Settings',[
                [sg.Slider(range=(-30, 30), orientation='v', size=(5, 20), default_value=0,key='-vert_slider-'),
                    sg.VerticalSeparator(),
                    sg.Column(render_modes),
                    sg.VerticalSeparator(),
                    sg.Button("Reset",key='-view_reset-')],
                [sg.Slider(range=(-180, 180), orientation='h', size=(20, 20), default_value=0, key='-horiz_slider-')]
            ]
            )],
            [sg.Frame('Robot Joints',[
                jointSlider("S",-170,170),
                jointSlider("L",-65,150),
                jointSlider("U",-135,255),
                jointSlider("R",-190,190),
                jointSlider("B",-135,135),
                [sg.Button("Reset",key='-joint_reset-')]
            ]
            )]
            ]

        self.layout = [          
            [sg.Text(f"Keypoint Skeleton: {name}")],
            [sg.Column(column1)],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Skeleton Wizard')]
            ]
        

    
    def run(self):
        self.window = sg.Window('Skeleton Wizard', self.layout)

        event = ''
        prev_values = {}
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(1, timeout_key='TIMEOUT')
            if event not in (sg.WIN_CLOSED,'-quit-'):
                if event != 'TIMEOUT' or values != prev_values:
                    self._runEvent(event, values)
                    self._setViewMode(values)
                    self.render()
                    prev_values = values
            self.window.bring_to_front()

        self.window.close()



    def _runEvent(self, event, values):
        self._setRotation(values['-horiz_slider-'],values['-vert_slider-'])
        self._setJointAngles(values)

        if event == '-view_reset-':
            self._resetRotation()
        if event == '-joint_reset-':
            self._resetJointAngles()


    def _setViewMode(self, values):
        modes = ['key','seg','seg_full','real']
        mode_keys = ['-render_key-','-render_seg-','-render_body-','-render_real-']
        mode = [modes[x] for x in range(len(modes)) if values[mode_keys[x]]][0]
        if self.current_mode == mode:
            return
        else:
            self.rend.setMode(mode)
            self.current_mode = mode



    def _resetRotation(self):
        self._setRotation(0,0)
        for slider in ['-horiz_slider-','-vert_slider-']:
            self.window[slider].update(0)

    def _setRotation(self, rotation_h, rotation_v):
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
        cv2.imshow("Keypoint Wizard",color)
        cv2.waitKey(1)

