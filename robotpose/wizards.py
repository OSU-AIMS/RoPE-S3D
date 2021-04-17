# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from logging import disable
import os
from PySimpleGUI.PySimpleGUI import TreeData
import cv2
import numpy as np
from numpy.core.numeric import True_

from deepposekit import Annotator
import PySimpleGUI as sg

from .data import DatasetInfo, Dataset
from .render import Aligner
from .skeleton import Skeleton, SkeletonInfo
from .simulation import SkeletonRenderer
from .urdf import URDFReader


class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.sk_inf = SkeletonInfo()
        self.urdf_reader = URDFReader()

        self.valid_urdf = self.urdf_reader.return_path() != None

        urdf_menu = [
            [sg.Text("Current URDF:")],
            [sg.Text(self.urdf_reader.return_path(),key='-current_urdf-'),
                sg.Button("Change",key='-browse_urdf-',tooltip='Select URDF path')]
        ]

        dataset_menu = [
            [sg.Text("Dataset:"),sg.InputCombo(self.compiled_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-details-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')]]

        keypoint_menu = [
            [sg.Text("Keypoint Skeleton:"),
                sg.InputCombo([x for x in self.sk_inf.valid() if x != 'BASE'],key='-skeleton-', size=(10, 1))],
            [sg.Button("Edit",key='-edit_skele-',tooltip='Edit Skeleton with Skeleton Wizard')],
            [sg.Button("View Annotations",key='-manual_annotate-', disabled=True)],
            [sg.Button("Create New Skeleton",key='-new_skele-'),sg.Button("Finish Skeleton Creation",key='-finish_skele-',visible=False)]
            ]

        self.layout = [
            [sg.Frame("URDF Options", urdf_menu)],
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
            if values['-skeleton-'] in self.sk_inf.valid():
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = False)
            else:
                for button in ['-manual_annotate-']:
                    self.window[button].update(disabled = True)
        else:
            for button in ['-details-','-align-']:
                self.window[button].update(disabled = True)

        if values['-skeleton-'] in self.sk_inf.valid():
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = False)
        else:
            for button in ['-edit_skele-']:
                self.window[button].update(disabled = True)
        
        if self.sk_inf.num_incomplete() > 0:
            self.window['-finish_skele-'].update(visible = True)
        else:
            self.window['-finish_skele-'].update(visible = False)

        if not self.valid_urdf:
            for button in ['-align-','-edit_skele-']:
                self.window[button].update(disabled = True)
                


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
                self.urdf_reader.store_path(path.replace('\\','/'))
                self.valid_urdf = True
            else:
                sg.popup_ok("Error:","Invalid URDF file selection.")
        self.window['-current_urdf-'].update(self.urdf_reader.return_path())
           
    def _showDetails(self, dataset):
        ds = Dataset(dataset)
        sg.popup_ok(str(ds), title=f"{dataset} Details")

    def _finishSkeleton(self):
        layout = [[sg.Text("Skeleton To Finish:"),
                sg.InputCombo(self.sk_inf.incomplete(),key='-skeleton-', size=(10, 1))],
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
                path = self.sk_inf.create_csv(name)
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

        u_reader = URDFReader()
        lims = u_reader.joint_limits
        lims *= (180/np.pi) 
        lims = np.round(lims)

        self.rend.setJointAngles([0,0,0,0,0,0])

        self.current_mode = 'key'
        self.crop = True
        self.use_cv = False

        self.jointTree = JointTree(name)
        self.meshTree = MeshTree(name)

        def jointSlider(name, lower, upper):
            return [sg.Text(f"{name}:"),
                sg.Slider(range=(lower, upper),
                    orientation='h', tick_interval=90, 
                    size=(20, 20), default_value=0, key=f'-{name}-')]


        render_modes = [
            [sg.Radio("Keypoint","render", default=True, key="-render_key-")],
            [sg.Radio("Segmented Joints","render", key="-render_seg-")],
            [sg.Radio("Realistic Metallic","render", key="-render_real-")],
            [sg.Checkbox("Crop To Fit", default=True,key='-crop-'), sg.Text('Pad:'),sg.Input('10',key='-crop_pad-',size=(3,None))],
            [sg.Checkbox("Display in New Window", default=False, key='-disp_cv-')]
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

        column2 = [
            [sg.Frame("Mesh Tree",[[self.meshTree()]])],
            [sg.Frame("Joint Tree",[[self.jointTree()]])]
        ]

        view_column = [
            [sg.Frame('Preview',[[sg.Image(key='-preview_img-')]])]
        ]

        self.layout = [          
            [sg.Text(f"Keypoint Skeleton: {name}")],
            [sg.Column(column1),sg.Column(column2),sg.Column(view_column,key='-preview_column-')],
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
                    self._updateDisabled(values)
                    self._runEvent(event, values)
                    self._setViewMode(values)
                    self.show(self.render())
                    prev_values = values
            self.window.bring_to_front()

        self.window.close()

    def _updateDisabled(self, values):
        self.window['-crop_pad-'].update(disabled=(not values['-crop-']))
        self.window['-preview_column-'].update(visible=(not values['-disp_cv-']))


    def _runEvent(self, event, values):
        self._setRotation(values['-horiz_slider-'],values['-vert_slider-'])
        self._setJointAngles(values)

        self.crop = values['-crop-']
        try:
            self.pad = int(values['-crop_pad-'])
        except ValueError:
            self.pad = 10

        self.use_cv = values['-disp_cv-']

        if event == '-view_reset-':
            self._resetRotation()
        if event == '-joint_reset-':
            self._resetJointAngles()


    def _setViewMode(self, values):
        modes = ['key','seg','real']
        mode_keys = ['-render_key-','-render_seg-','-render_real-']
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
        if self.crop:
            color = self._cropImage(color)
        return color
    
    def show(self, image):
        if self.use_cv:
            cv2.imshow("Keypoint Wizard",image)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            imgbytes = cv2.imencode('.png', image)[1].tobytes()
            self.window['-preview_img-'].update(data=imgbytes)



    def _cropImage(self, image):
        # Aim for 700 w by 720 h
        occupied = np.any(image,-1)
        rows, columns = np.where(occupied)
        min_row = max(0,min(rows)-self.pad)
        max_row = min(image.shape[0]-1,max(rows)+self.pad)
        min_col = max(0,min(columns)-self.pad)
        max_col = min(image.shape[1]-1,max(columns)+self.pad)
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




class MeshTree(Skeleton):
    def __init__(self, name):
        super().__init__(name)
        self.update()
        self.treedata = sg.TreeData()
        self.urdf_reader = URDFReader()
        self._addMeshes()
        self._addKeypointsToTree()

    def _addMeshes(self):
        for mesh in self.urdf_reader.mesh_names:
            self.treedata.insert('',mesh,mesh,[])

    def _addKeypointsToTree(self):
        for keypoint in self.keypoints:
            dat = self.keypoint_data[keypoint]
            pose = list(np.round(np.array(dat['pose']),2))
            self.treedata.insert(
                dat['parent_link'],
                keypoint,
                keypoint,
                pose
                )
            

    def __call__(self):
        return sg.Tree(self.treedata,('X','Y','Z','R','P','Y'), def_col_width=3, auto_size_columns=False, num_rows=8, key='-mesh_tree-')



class JointTree(Skeleton):
    def __init__(self, name):
        super().__init__(name)
        self.update()
        self.treedata = sg.TreeData()
        self._addJoints()
        self._addJointPredictors()

    def _addJoints(self):
        for joint in self.joint_data.keys():
            self.treedata.insert('',joint,joint,[])
        
    def _addJointPredictors(self):
        for joint in self.joint_data.keys():
            for pred in self.joint_data[joint]['predictors'].keys():
                pred_data = self.joint_data[joint]['predictors'][pred]
                self.treedata.insert(
                    joint,
                    f'{joint}-{pred}',
                    f'pred_{pred}',
                    [pred_data['from'],pred_data['to'],pred_data['length'],pred_data['offset']]
                    )

    def __call__(self):
        return sg.Tree(self.treedata,('From','To','Length','Offset'),key='-joint_tree-')
