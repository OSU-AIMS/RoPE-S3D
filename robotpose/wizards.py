# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import PySimpleGUI as sg

from .dataset import DatasetInfo, Dataset
from .render import Aligner, Renderer
from .skeleton import valid_skeletons
from .autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator



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
            [sg.Text("Skeleton:"),sg.InputCombo(valid_skeletons(),key='-skeleton-', size=(20, 1))],
            [sg.Button("AutoAnnotate",key='-annotate-',tooltip='In dev', disabled=True)],
            [],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Dataset Wizard')]
            ]


    def run(self):
        self.window = sg.Window('Dataset Wizard', self.layout)

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
                self.window['-annotate-'].update(disabled = False)
            else:
                self.window['-annotate-'].update(disabled = True)
        else:
            for button in ['-load-','-recompile-','-build-','-align-','-annotate-']:
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




