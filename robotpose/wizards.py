# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import logging
import os
import shutil

import cv2
import numpy as np
import PySimpleGUI as sg

from .CompactJSONEncoder import CompactJSONEncoder
from .constants import WIZARD_DATASET_PREVIEW as PREVIEW
from .data import Dataset, DatasetInfo, Splitter, Verifier
from .paths import Paths as p
from .simulation import Aligner, Renderer
from .training.models import ModelTree
from .urdf import URDFReader


class Wizard(DatasetInfo):
    """A GUI that allows for easy interaction with data and settings."""

    def __init__(self):
        super().__init__()

        # URDF Reading 
        self.urdf_reader = URDFReader()
        self.valid_urdf = self.urdf_reader.internal_path != None

        # Dataset Splitting
        self.last_split_data = {}
        self.last_applied_split_data = {}
        self.applied_split_data = {}
        self.current_dataset = self.compiled_sets[0]

        if self.current_dataset is None:
            # Raise error if no datasets have been compiled
            raise SystemError(
                'No datasets have been compiled. Please build a dataset before attempting to use the wizard.'+
                "\nThis is done with 'python wizard.py DATASET_NAME'")
                
        self._getNewSplitData()

        if PREVIEW:
            # Dataset Previewing
            self._changeThumbnails()
            self.preview_idx = 0

        # GUI Layout Setup 

        ########################################################################################
        # Tab responsible for Dataset selection and viewing
        data_tab_layout = [
            [sg.Txt("Dataset:"),sg.Combo(self.compiled_sets,key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-details-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer'),
                sg.Button("Verify",key='-verify-',tooltip='Remove images of incorrect poses from dataset')],
            [sg.Image(k='-preview-')],
            
        ]

        ########################################################################################
        ds_split_left_pad = 7

        # Slider config for dataset split selection
        def SplitSlider(key,**kwargs):
            return sg.Slider((0,1),self.applied_split_data[key],orientation='h',resolution=.05,k=f'-{key}_prop-',**kwargs)

        # Base right-justified text for dataset split selection
        def RightText(text,**kwargs):
            return sg.Txt(text,size=(ds_split_left_pad,1),justification='right',**kwargs)

        split_inputs = [
            [RightText('Train:'), SplitSlider('train'),sg.Button('Update',k='-update_split-')],
            [RightText('Validate:'), SplitSlider('validate')],
            [RightText('Ignore:'), SplitSlider('ignore',disabled=True,trough_color='dim gray')],
        ]

        ds_split_menu = [
            *split_inputs,
            [RightText(''),sg.Txt('Training',text_color='green',background_color='white'),
                sg.Txt('Validation',text_color='blue',background_color='white'),
                sg.Txt('Ignored',text_color='red',background_color='white')],
            [RightText('New:'), sg.Graph((200,20),(0,0),(1,1),background_color='white',k='-new_split_graph-')],
            [RightText('Current:'), sg.Graph((200,20),(0,0),(1,1),background_color='white',k='-current_split_graph-')]]

        training_tab_layout = [
            [sg.Txt('Current Dataset:',tooltip='Synced with "Data" Tab'),sg.Txt(self.current_dataset,size=(20,1),k='-split_screen_ds-',tooltip='Synced with "Data" Tab')],
            [sg.Frame('Data Split', ds_split_menu)]
        ]

        ########################################################################################
        prediction_tab_layout = [
            [ModelTree()()],
            [sg.Button('Delete Selected',k='-delete_model-')]
        ]

        ########################################################################################
        urdf_tab_layout = [
            [sg.Txt("URDF:"),
                sg.Combo(self.urdf_reader.available_names,self.urdf_reader.name,key='-urdf-', size=(10, 1))],
            [sg.Txt("Active:"), sg.Txt(self.urdf_reader.name,key='-active_urdf-')],
            [sg.Button("View Robot",key='-view-',tooltip='View robot in MeshViewer')]
        ]

        ########################################################################################
        data_tab = sg.Tab('Data', data_tab_layout)
        training_tab = sg.Tab('Training', training_tab_layout)
        prediction_tab = sg.Tab('Prediction', prediction_tab_layout)
        urdf_tab = sg.Tab('URDF', urdf_tab_layout)

        tabgroup = sg.TabGroup([[data_tab,training_tab,prediction_tab,urdf_tab]],k='-tabgroup-')

        self.layout = [
            [tabgroup],
            [sg.Button("Quit",key='-quit-',tooltip='Quit Wizard')]
            ]


    def run(self):
        """Run Wizard"""
        # Create window
        self.window = sg.Window('Dataset Wizard', self.layout.copy(), finalize=True)
        self.window.bring_to_front()

        event = ''
        while event not in (sg.WIN_CLOSED,'-quit-'):
            event, values = self.window.read(10)
            if event not in (sg.WIN_CLOSED,'-quit-'):
                self._updateValues(values)
                self._runEvent(event, values)

        self.window.close()


    def _updateValues(self, values:dict):
        """Update app behavior based on the values present on inputs"""

        self.updateDatasetSplit(values)

        if values['-dataset-'] in self.unique_sets:
            # If dataset is valid, show preview, enable buttons
            if values['-tabgroup-'] == "Data":
                if PREVIEW:
                    self._showPreview()
            if values['-dataset-'] != self.current_dataset:
                self.current_dataset = values['-dataset-']
                self.updateDatasetSplit(values)
                self.window['-split_screen_ds-'].update(self.current_dataset)
                if PREVIEW:
                    self._changeThumbnails()
            for button in ['-details-','-align-','-verify-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-details-','-align-','-verify-']:
                self.window[button].update(disabled = True)

        # Enable/Disable model delete button if on correct tab
        if values['-tabgroup-'] == "Prediction":
            # Only allow deletion of models, not all of a dataset's models
            if values['-model_tree-'] == [] or sum([x in self.unique_sets for x in values['-model_tree-']]) > 0:
                self.window['-delete_model-'].update(disabled=True)
            else:
                self.window['-delete_model-'].update(disabled=False)

        if not self.valid_urdf:
            # Only allow alignment of dataset if valid URDF is loaded
            for button in ['-align-']:
                self.window[button].update(disabled = True)

        # Update URDF if changed
        if values['-urdf-'] in self.urdf_reader.available_names and values['-urdf-'] != self.urdf_reader.name:
            self.urdf_reader.path = self.urdf_reader.available_paths[self.urdf_reader.available_names.index(values['-urdf-'] )]
            self.window['-active_urdf-'].update(self.urdf_reader.name)


    def _runEvent(self, event:str, values:dict):
        """Run a specific subprocess based on user input"""
        if event =='-align-':
            self._runAligner(values['-dataset-'])
        elif event == '-verify-':
            self._runVerifier(values['-dataset-'])
        elif event == '-details-':
            self._showDetails(values['-dataset-'])
        elif event == '-view-':
            self._runMeshViewer()
        elif event == '-update_split-':
            t,v = self.updateDatasetSplit(values)
            self._writeDatasetSplit(t,v)
        elif event == '-delete_model-':
            self._deleteModel(values)



    def _deleteModel(self, values):
        response = sg.popup_ok_cancel(f"Delete the following model(s)?\n{values['-model_tree-']}",title="Model Deletion")
        if response == "OK":
            for model in values['-model_tree-']:
                shutil.rmtree(os.path.join(p().MODELS,model))
            self.window['-model_tree-'].update(ModelTree().data)



    def updateDatasetSplit(self, values:dict):
        """Update dataset split sliders and graphics"""
        
        try:
            train = float(values['-train_prop-'])
            valid = float(values['-validate_prop-'])
        except ValueError:
            return

        # If train prop is over 1, set it to 1 and everything else to 0
        if train > 1:
            self.window['-train_prop-'].update(value=1)
            train = 1
            self.window['-validate_prop-'].update(value=0)
            valid = 0
        
        # If total prop is over 1, bring down validation to conform with training
        if train + valid > 1:
            self.window['-validate_prop-'].update(value=(1-train))
            valid = 1 - train

        # Update ignore prop
        self.window['-ignore_prop-'].update(value=f'{1-train-valid:0.2f}')

        # Update current graphics if values have changed
        current = {'train':train,'valid':valid}
        if current != self.last_split_data:
            self._updateSplitGraph(train, valid, '-new_split_graph-')
            self.last_split_data = current
        
        # Get actual applied data and update graphics if it has changed
        self._getNewSplitData()
        if self.last_applied_split_data != self.applied_split_data:
            self.last_applied_split_data = self.applied_split_data
            self._updateSplitGraph(self.applied_split_data['train'],self.applied_split_data['validate'],'-current_split_graph-')

        return train, valid


    def _getNewSplitData(self):
        """Retrive the current dataset split applied"""

        def writeData(data):
            with open(p().SPLIT_CONFIG,'w') as f:
                f.write(CompactJSONEncoder(indent=4).encode(data))

        if not os.path.isfile(p().SPLIT_CONFIG):
            data = {}
            writeData(data)
        else:
            with open(p().SPLIT_CONFIG,'r') as f:
                data = json.load(f)

        if self.current_dataset not in data:
            data[self.current_dataset] = {
                'train':0.7,
                'validate':0.2,
                'ignore':0.1
            }
            writeData(data)

        self.applied_split_data = data[self.current_dataset]


    def _writeDatasetSplit(self, train:float, validate:float):
        # Write new split config
        with open(p().SPLIT_CONFIG,'r') as f:
            data = json.load(f)
        data[self.current_dataset] = {'train':train,'validate':validate,'ignore':1-train-validate}
        with open(p().SPLIT_CONFIG,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(data))

        # Actually split if data is present
        expected_anno_dir = os.path.join(Dataset(self.current_dataset).dataset_dir,"link_annotations")
        if os.path.isdir(expected_anno_dir):
            self.window.set_cursor('wait')
            self.window.read(1)
            s = Splitter(expected_anno_dir)
            s.resplit(train,validate)
            self.window.set_cursor('arrow')
            self.window.read(1)


    def _updateSplitGraph(self, train:float, validate:float, key:str):
        """Update split graphics

        Parameters
        ----------
        train, validate : floats
            Ratio of type
        key : str
            Window key of the graphic item to change
        """
        self.window[key].erase()    # Clear old drawing

        # Update with new
        self.window[key].draw_rectangle((0,1),(train,0),fill_color='green')
        self.window[key].draw_rectangle((train,1),(train+validate,0),fill_color='blue')
        self.window[key].draw_rectangle((train+validate,1),(1,0),fill_color='red')

    def _showPreview(self):
        """Show thumbnail preview of dataset"""
        # Go to next thumbnail
        self.preview_idx += 1
        if self.preview_idx >= self.thumbnails.shape[0]:
            self.preview_idx = 0

        # Set dimensions for thumbnail
        dims = [x * 2 for x in self.thumbnails.shape[1:3]]
        dims.reverse()

        # Resize and show
        image = cv2.resize(self.thumbnails[self.preview_idx],tuple(dims))
        imgbytes = cv2.imencode('.png', image)[1].tobytes()
        self.window['-preview-'].update(data=imgbytes)

    def _changeThumbnails(self):
        """Load new set of thumbnails into memory"""
        ds = Dataset(self.current_dataset)
        self.thumbnails = np.copy(ds.preview_img)
        self.preview_idx = 0

    def _showDetails(self, dataset:str):
        """Show dataset attributes"""
        ds = Dataset(dataset)
        sg.popup_ok(str(ds), title=f"{dataset} Details")

    def _runVerifier(self, dataset:str):
        """Run the dataset Verifier Tool"""
        self._hideWindow()
        v = Verifier(dataset)
        v.run()
        self._showWindow()

    def _runAligner(self, dataset:str):
        """Run the dataset Aligner Tool"""
        self._hideWindow()
        align = Aligner(dataset)
        align.run()
        self._showWindow()

    def _runMeshViewer(self):
        """Run the MeshViewer tool"""
        self._hideWindow()
        wiz = MeshViewer()
        wiz.run()
        cv2.destroyAllWindows()
        self._showWindow()

    def _hideWindow(self):
        """Hide main window during subprocess"""
        self.window.disable()
        self.window.disappear()
    
    def _showWindow(self):
        """Show main window after subprocess"""
        self.window.enable()
        self.window.reappear()
        self.window.bring_to_front()







class MeshViewer():
    """Allows for the viewing and interaction with the meshes defined in a robot's URDF"""

    def __init__(self):

        self.rend = Renderer(suppress_warnings=True)
        self.crop = False
        self._findBasePose()
        
        # Get limits in degrees and rounded to integers
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
                [sg.Column(render_modes)],
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
            [sg.Button("Quit",key='-quit-',tooltip='Quit Mesh Wizard')],
            ]
        

    def run(self):
        """Run Mesh Wizard"""
        self.window = sg.Window('Mesh Wizard', self.layout)
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
        """Set the style of the 3D model"""
        modes = ['seg','real']
        mode_keys = ['-render_seg-','-render_real-']
        mode = [modes[x] for x in range(len(modes)) if values[mode_keys[x]]][0]
        if self.mode == mode:
            return
        else:
            self.rend.setMode(mode)
            self.mode = mode

    def _resetRotation(self):
        """Go back to base rotation value"""
        self._setRotation({'-horiz_slider-':0})
        for slider in ['-horiz_slider-']:
            self.window[slider].update(0)

    def _setRotation(self, values):
        """Set the position of the camera around the robot model"""
        rotation_h = values['-horiz_slider-']
        self.rotation_h = (rotation_h/180) * np.pi

        self.c_pose = np.copy(self.base_pose)
        self.c_pose[1] *= np.cos(self.rotation_h)
        self.c_pose[0] *= np.sin(self.rotation_h)
        self.c_pose[4] = 0
        self.c_pose[5] = self.rotation_h
        self.rend.setCameraPose(self.c_pose)


    def _resetJointAngles(self):
        """Put robot back in base pose"""
        self.rend.setJointAngles([0,0,0,0,0,0])
        for joint in ['-S-','-L-','-U-','-R-','-B-']:
            self.window[joint].update(0)

    def _setJointAngles(self, values):
        """Update the robot pose"""
        joint_angles = [0,0,0,0,0,0]
        for joint, idx in zip(['-S-','-L-','-U-','-R-','-B-'], range(5)):
            joint_angles[idx] = values[joint] * np.pi/180 

        self.rend.setJointAngles(joint_angles)

    def render(self):
        """Render and crop image if needed"""
        color, depth = self.rend.render()
        if self.crop:
            color = self._cropImage(color)
        return color
    
    def show(self, image):
        """Show render either in PySimpleGUI or in an OpenCV window"""
        if self.use_cv:
            cv2.imshow("Mesh Wizard",image)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            imgbytes = cv2.imencode('.png', image)[1].tobytes()
            self.window['-preview_img-'].update(data=imgbytes)

    def _cropImage(self, image, pad = 10):
        """Remove black space on edges of render"""
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

    
    def _findBasePose(self):
        """Find the closest camera position where the entire robot is always in frame"""
        self.rend.setJointAngles([0,0,np.pi/2,0,0,0])

        def set_render_and_process(r,z):
            self.base_pose = [0,-r,z, 0,0,0]
            self.rend.setCameraPose(self.base_pose)
            self.base_pose[0] = r
            frame = self.render()
            return np.any(frame,-1)

        r = 1.5
        z = 0.75

        frame = set_render_and_process(r,z)

        for inc in [1,0.5,0.25,0.1,0.05,0.01]:

            # Back away until blackspace on top and bottom
            while frame[0].any() or frame[-1].any():
                r += inc
                frame = set_render_and_process(r,z)

            # Used to determine max/min row
            def r_val(frame, x):
                # x is either 0 (min) or -1 (max)
                f = frame.any(1)
                return np.where(f)[0][x]

            # Center down
            while r_val(frame, 0) < (frame.shape[0] - r_val(frame, -1)):
                z += inc
                frame = set_render_and_process(r,z)
            # Center up
            while r_val(frame, 0) > (frame.shape[0] - r_val(frame, -1)):
                z -= inc
                frame = set_render_and_process(r,z)
            k = 10 # Move towards, leaving k pixels above and/or below
            while r_val(frame, 0) > k and (frame.shape[0] - r_val(frame, -1)) > k:
                r -= inc
                frame = set_render_and_process(r,z)
        self.rend.setJointAngles([0,0,0,0,0,0])
        set_render_and_process(r,z)
        logging.info(f'\n\nFor reference, the base camera position for this robot is:\n{self.base_pose}\n\n')

