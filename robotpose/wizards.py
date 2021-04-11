# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import PySimpleGUI as sg
import os
import cv2
import numpy as np

from .dataset import DatasetInfo, Dataset
from .render import Aligner, Renderer
from .skeleton import Skeleton, valid_skeletons
from .autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator
from .simulation.rendering import SkeletonRenderer
from deepposekit import Annotator



def annotate(dataset,skeleton):
    print(f'Annotating {dataset} using skeleton {skeleton}')
    rend = Renderer(dataset, skeleton)
    key = AutomaticKeypointAnnotator(dataset, skeleton, renderer = rend)
    key.run()
    del key
    seg = AutomaticSegmentationAnnotator(dataset, skeleton, renderer = rend)
    seg.run()
    print('Annotation Complete')


class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.layout = [          
            [sg.Text("Dataset:"),sg.InputCombo(self.unique_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("Load",key='-load-',tooltip='Load dataset, building if required.'), 
                sg.Button("Recompile",key='-recompile-',tooltip='Reprocess raw data from dataset.'),
                sg.Button("Build",key='-build-',tooltip='Build dataset from scratch.')],
            [sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')],
            [sg.Text("Skeleton:"),
                sg.InputCombo(valid_skeletons(),key='-skeleton-', size=(20, 1)),
                sg.Button("Edit Skeleton",key='-edit_skele-',tooltip='Edit Skeleton with Skeleton Wizard')],
            [sg.Button("AutoAnnotate",key='-annotate-',tooltip='In dev', disabled=True),
                sg.Button("View Annotations",key='-manual_annotate-', disabled=True)],
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
            for button in ['-load-','-recompile-','-build-','-align-']:
                self.window[button].update(disabled = False)
            if values['-skeleton-'] in valid_skeletons():
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = False)
            else:
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = True)
        else:
            for button in ['-load-','-recompile-','-build-','-align-','-annotate-']:
                self.window[button].update(disabled = True)

        if values['-skeleton-'] in valid_skeletons():
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = True)
                


    def _runEvent(self,event,values):
        if event =='-align-':
            self._runAligner(values['-dataset-'])
        elif event == '-load-':
            pass
        elif event == '-recompile-':
            self._recompile(values['-dataset-'])
        elif event == '-build-':
            self._rebuild(values['-dataset-'])
        elif event == '-annotate-':
            self._annotate(values['-dataset-'], values['-skeleton-'])
        elif event == '-manual_annotate-':
            self._manualAnnotate(values['-dataset-'], values['-skeleton-'])
        elif event == '-edit_skele-':
            self._runKeypointWizard(values['-skeleton-'])
           
    def _manualAnnotate(self,dataset,skeleton):
        ds = Dataset(dataset,skeleton)
        ds.makeDeepPoseDS()
        app = Annotator(datapath=os.path.abspath(ds.deepposeds_path),
                dataset='images',
                skeleton=ds.skele.csv_path,
                shuffle_colors=False,
                text_scale=1)

        app.run()

    def _annotate(self,dataset,skeleton):
        annotate(dataset,skeleton)

    def _recompile(self,dataset):
        print(f'Recompiling {dataset}')
        ds = Dataset(dataset, recompile=True)
        print('Recompilation Complete')

    def _rebuild(self,dataset):
        print(f'Building {dataset}')
        ds = Dataset(dataset,rebuild=True)
        print('Build Complete')

    def _runAligner(self, dataset):
        print(f'Building {dataset}')
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

        self.rend = SkeletonRenderer(name)
        self.base_pose = [1.5,1.5,.35, 0,np.pi/2,np.pi/2]
        self.rotation = 0
        self._rotate(0)

        self.rend.setJointAngles([0,0,0,0,0,0])

        self.layout = [          
        [sg.Text(f"Skeleton: {name}")],
        [sg.Text("Current Rotation: 0 Degrees", key='curr_rot')],
        [sg.Button("Rotate Left",key='-rot_left-'),sg.Button("Rotate Right",key='-rot_right-')],
        [sg.Button("Quit",key='-quit-',tooltip='Quit Skeleton Wizard')]
        ]
        

    
    def run(self):
        self.window = sg.Window('Skeleton Wizard', self.layout)

        event = ''
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(10)
            if event not in (sg.WIN_CLOSED,'-quit-'):
                #self._updateButtons(values)
                self._runEvent(event, values)
                self.render()
                self.window['curr_rot'].update(f"Current Rotation: {round(self.rotation * 180 / np.pi)} Degrees")

        self.window.close()



    def _runEvent(self, event, values):
        if event == '-rot_left-':
            self._rotate(-np.pi/6)
        elif event == '-rot_right-':
            self._rotate(np.pi/6)

    def _rotate(self, amount):
        self.rotation += amount
        self.c_pose = np.copy(self.base_pose)
        self.c_pose[0] *= np.cos(self.rotation)
        self.c_pose[1] *= np.sin(self.rotation)
        self.c_pose[5] = self.rotation + np.pi/2
        self.rend.setCameraPose(self.c_pose)

    def render(self):
        color, depth = self.rend.render()
        cv2.imshow("Keypoint Wizard",color)
        cv2.waitKey(1)

