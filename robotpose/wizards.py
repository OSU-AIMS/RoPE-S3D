# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os
from PySimpleGUI.PySimpleGUI import Window
import cv2
import numpy as np


from deepposekit import Annotator
import PySimpleGUI as sg

from .data import DatasetInfo, Dataset
from .skeleton import Skeleton, SkeletonInfo
from .simulation import SkeletonRenderer, Aligner
from .urdf import URDFReader
from .utils import expandRegion



class DatasetWizard(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.get()

        self.sk_inf = SkeletonInfo()
        self.urdf_reader = URDFReader()

        self.valid_urdf = self.urdf_reader.return_path() != None

        urdf_menu = [
            [sg.Txt("Current URDF:")],
            [sg.Txt(self.urdf_reader.return_path(),key='-current_urdf-'),
                sg.Button("Change",key='-browse_urdf-',tooltip='Select URDF path')]
        ]

        dataset_menu = [
            [sg.Txt("Dataset:"),sg.Combo(self.compiled_sets(),key='-dataset-', size=(20, 1))],
            [sg.Button("View Details",key='-details-',tooltip='View dataset details'),
                sg.Button("Align",key='-align-',tooltip='Align Dataset images with renderer')]]

        keypoint_menu = [
            [sg.Txt("Keypoint Skeleton:"),
                sg.Combo([x for x in self.sk_inf.valid() if x != 'BASE'],key='-skeleton-', size=(10, 1))],
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
        layout = [[sg.Txt("Skeleton To Finish:"),
                sg.Combo(self.sk_inf.incomplete(),key='-skeleton-', size=(10, 1))],
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



ANGLES = ['-pi','-pi/2','0','pi/2','pi']
ANGLE_DICT = {'-pi':-np.pi,'-pi/2':-np.pi/2,'0':0,'pi/2':np.pi/2,'pi':np.pi}

class SkeletonWizard(Skeleton):

    def __init__(self, name):
        super().__init__(name)
        self.update()

        self.rend = SkeletonRenderer(name, suppress_warnings=True)
        self.base_pose = [1.5,-1.5,.35, 0,np.pi/2,0]

        self.u_reader = URDFReader()
        lims = self.u_reader.joint_limits
        lims *= (180/np.pi) 
        lims = np.round(lims)

        self.rend.setJointAngles([0,0,0,0,0,0])

        self.mode = 'key'
        self.crop = True
        self.use_cv = False
        self.last_joint_selected = ''
        self.last_keypoint_selected = ''
        self.last_predictor_selected = ''

        self.jointTree = JointTree(name)
        self.meshTree = MeshTree(name)

        def jointSlider(name, lower, upper):
            return [sg.Txt(f"{name}:"),
                sg.Slider(range=(lower, upper),
                    orientation='h', tick_interval=90, 
                    size=(20, 20), default_value=0, key=f'-{name}-')]


        render_modes = [
            [sg.Radio("Keypoint","render", default=True, key="-render_key-")],
            [sg.Radio("Segmented Joints","render", key="-render_seg-")],
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

        edit_keypoint = [
            [sg.Txt('Name:'),
                sg.Combo(self.keypoints,key='-keypoint_name-')],
            [sg.Txt('Parent Mesh:'),sg.Combo(self.u_reader.mesh_names[:-1],key='-keypoint_parent_mesh-'),
                sg.Txt('Parent Keypoint:'),sg.Combo(self.keypoints,key='-keypoint_parent_key-')],
            [sg.T('X:'),sg.Input('',size=(5,None),key='-keypoint_x-'),
                sg.T('Y:'),sg.Input('',size=(5,None),key='-keypoint_y-'),
                sg.T('Z:'),sg.Input('',size=(5,None),key='-keypoint_z-')],
            [sg.T('Roll:'),sg.Spin(ANGLES,size=(5,None),key='-keypoint_roll-'),
                sg.T('Pitch:'),sg.Spin(ANGLES,size=(5,None),key='-keypoint_pitch-'),
                sg.T('Yaw:'),sg.Spin(ANGLES,size=(5,None),key='-keypoint_yaw-')],
            [sg.Button('New',key='-new_keypoint-'),
                sg.Button('Rename',key='-rename_keypoint-'),
                sg.Button('Remove',key='-remove_keypoint-')]
        ]

        edit_predictor =[
            [sg.Txt('Joint:'),
                sg.Combo([j for j in self.joint_data.keys()],key='-joint_name-',auto_size_text=False,size=(3,None)),
                sg.Txt('Predictor:'),sg.Combo([],key='-predictor_name-',auto_size_text=False,size=(3,None))],
            [sg.Txt('From:'),sg.Combo(self.keypoints,key='-predictor_from-'),
                sg.Txt('To:'),sg.Combo(self.keypoints,key='-predictor_to-')],
            [sg.Txt('Length:'),sg.Input('',size=(5,None),key='-predictor_length-'),
                sg.Txt('Angle Offset:'),sg.Input('',key='-predictor_offset-',size=(5,None)),
                sg.Button('Estimate',key='-est_pred_length-',disabled=True)],
            [sg.Button('New',key='-new_predictor-'),
                sg.Button('Remove',key='-remove_predictor-')]
        ]

        column2 = [
            [sg.Frame("Mesh Tree",[[self.meshTree()]])],
            [sg.Frame("Edit Keypoint",edit_keypoint)],
            [sg.Frame("Joint Tree",[[self.jointTree()]])],
            [sg.Frame("Edit Predictor",edit_predictor)],
        ]

        view_column = [
            [sg.Frame('Preview',[[sg.Image(key='-preview_img-',pad=(2,2))]], pad=(1,1))]
        ]

        self.layout = [          
            [sg.Txt(f"Keypoint Skeleton: {name}",font="Any 20")],
            [sg.Column(column1),sg.Column(column2),sg.Column(view_column,key='-preview_column-',pad=(1,1))],
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
                    self._changeKeypointLocation(values)
                    self._changePredictorValues(values)
                    self._runEvent(event, values)
                    self._setRotation(values)
                    self._setJointAngles(values)
                    self._setViewMode(values)
                    self.show(self.render(values))
                    prev_values = values

        self.window.close()


    def _updateInputs(self, values):
        self.window['-preview_column-'].update(visible=(not values['-disp_cv-']))

        def updateKeypointEditor():
            a = values['-keypoint_name-'] in self.keypoints
            for item in ['-keypoint_x-','-keypoint_y-','-keypoint_z-','-keypoint_roll-','-keypoint_pitch-','-keypoint_yaw-','-keypoint_parent_mesh-','-keypoint_parent_key-']:
                self.window[item].update(disabled=not a)

            if a:
                #Load values
                if self.last_keypoint_selected != values['-keypoint_name-']:
                    self.window['-keypoint_parent_key-'].update(values=[x for x in self.keypoints if x != values['-keypoint_name-']])
                    self.window['-keypoint_parent_key-'].update(value=self.keypoint_data[values['-keypoint_name-']]['parent_keypoint'])
                    self.window['-keypoint_parent_mesh-'].update(value=self.keypoint_data[values['-keypoint_name-']]['parent_link'])
                    for item, idx in zip(['-keypoint_x-','-keypoint_y-','-keypoint_z-'],range(3)):
                        self.window[item].update(value=self.keypoint_data[values['-keypoint_name-']]['pose'][idx])
                    for item, idx in zip(['-keypoint_roll-','-keypoint_pitch-','-keypoint_yaw-'],range(3,6)):
                        for ang in ANGLE_DICT:
                            if abs(ANGLE_DICT[ang] - self.keypoint_data[values['-keypoint_name-']]['pose'][idx]) < .1:
                                self.window[item].update(value=ang)

                    self.last_keypoint_selected = values['-keypoint_name-']

        def updatePredEditor():
            a = values['-joint_name-'] in self.joint_data.keys()
            self.window['-predictor_name-'].update(disabled = not a)
            if a:
                if values['-joint_name-'] != self.last_joint_selected:
                    self.window['-predictor_name-'].update(values=[x for x in self.joint_data[values['-joint_name-']]['predictors'].keys()])
                    self.last_joint_selected = values['-joint_name-']
                    self.last_predictor_selected = ''

            try:
                b = (values['-predictor_name-'] in [x for x in self.joint_data[values['-joint_name-']]['predictors'].keys()]) and a
            except KeyError:
                b = False
            for item in ['-predictor_from-','-predictor_to-','-predictor_length-','-predictor_offset-','-est_pred_length-','-remove_predictor-']:
                self.window[item].update(disabled = not b)
            
            if a:
                c = values['-predictor_name-'] in [x for x in self.joint_data[values['-joint_name-']]['predictors'].keys()]
                if c:
                    sel_joint = values['-joint_name-']
                    sel_predictor = values['-predictor_name-']
                    if self.last_predictor_selected != sel_predictor:
                        for key, val in zip(['-predictor_from-','-predictor_to-','-predictor_length-','-predictor_offset-'],['from','to','length','offset']):
                            self.window[key].update(value = self.joint_data[sel_joint]['predictors'][sel_predictor][val])
                        self.last_predictor_selected = sel_predictor

        updatePredEditor()
        updateKeypointEditor()


    def _changeKeypointLocation(self, values):
        sel_keypoint = values['-keypoint_name-']
        # Change Parent if different
        if sel_keypoint in self.keypoints and sel_keypoint == self.last_keypoint_selected:
            if values['-keypoint_parent_mesh-'] != self.data['keypoints'][sel_keypoint]['parent_link']:
                if values['-keypoint_parent_mesh-'] in self.u_reader.mesh_names[:-1]:
                    self._changeKeypointParentLink(sel_keypoint,values['-keypoint_parent_mesh-'])
                    self._refreshTrees()

            if values['-keypoint_parent_key-'] != self.data['keypoints'][sel_keypoint]['parent_keypoint']:
                acceptable = [x for x in self.keypoints if x != sel_keypoint]
                acceptable.append('')
                if values['-keypoint_parent_key-'] in acceptable:
                    self._changeKeypointParentPoint(sel_keypoint,values['-keypoint_parent_key-'])

            current_pose = self.data['keypoints'][sel_keypoint]['pose']
            new_pose = current_pose.copy()
            for name, idx in zip(['-keypoint_x-','-keypoint_y-','-keypoint_z-'], range(3)):
                try:
                    val = float(values[name])
                    new_pose[idx] = val
                except ValueError:
                    pass

            for name, idx in zip(['-keypoint_roll-','-keypoint_pitch-','-keypoint_yaw-'], range(3,6)):
                val = ANGLE_DICT[values[name]]
                new_pose[idx] = val
            
            if current_pose != new_pose:
                self._changeKeypointPose(sel_keypoint, new_pose)
                self.window['-mesh_tree-'].update(key=sel_keypoint,value=new_pose)
            

    def _changePredictorValues(self, values):
        sel_joint = values['-joint_name-']
        sel_predictor = values['-predictor_name-']
        if sel_joint in self.joint_data.keys():
            if sel_predictor in self.joint_data[sel_joint]['predictors'].keys():
                prev_data = self.joint_data[sel_joint]['predictors'][sel_predictor]
                new_data = {}
                if values['-predictor_from-'] in self.keypoints:
                    new_data['from'] = values['-predictor_from-']
                else:
                    new_data['from'] = prev_data['from']
                if values['-predictor_to-'] in self.keypoints:
                    new_data['to'] = values['-predictor_to-']
                else:
                    new_data['to'] = prev_data['to']

                for key_json, key_gui  in zip(['length','offset'],['-predictor_length-','-predictor_offset-']):
                    try:
                        new_data[key_json] = float(values[key_gui])
                    except ValueError:
                        new_data[key_json] = prev_data[key_json]

                if new_data != prev_data:
                    self.data['joints'][sel_joint]['predictors'][sel_predictor] = new_data
                    self._writeJSON()
                    self.window['-joint_tree-'].update(key=f'{sel_joint}-{sel_predictor}',value=(new_data['from'],new_data['to'],new_data['length'],new_data['offset']))




    def _runEvent(self, event, values):
        self.crop = values['-crop-']
        self.use_cv = values['-disp_cv-']

        if event == '-view_reset-':
            self._resetRotation()
        if event == '-joint_reset-':
            self._resetJointAngles()
        if event == '-new_keypoint-':
            self._newKeypoint()
        if event == '-rename_keypoint-':
            self._renameKeypointGUI(values)
        if event == '-remove_keypoint-':
            self._removeKeypointGUI(values)
        if event == '-new_predictor-':
            self._newPredictor(values)
        if event == '-remove_predictor-':
            self._deletePredictor(values)


    def _deletePredictor(self, values):
        layout = [
            [sg.Txt(f"Remove predictor {values['-predictor_name-']} from joint {values['-joint_name-']}?")],
            [sg.Button('Remove',key='-remove-'), sg.Button('Cancel',key='-cancel-')]
        ]
        window = sg.Window('Remove Predictor', layout)
        event, v = window.read()
        if event == '-remove-':
            self._removePredictor(values['-joint_name-'],values['-predictor_name-'])
            sg.popup_auto_close("Predictor removed.", auto_close_duration=2)
            self._refreshTrees()
        window.close()


    def _newPredictor(self, values):
        if values['-joint_name-'] in ['S','L','U','R','B']:
            inital = values['-joint_name-']
        else:
            inital = None
        layout = [
            [sg.Txt('Joint to Add to:'),sg.Spin(['S','L','U','R','B'],inital,key='-in-',auto_size_text=False,size=(5,None))],
            [sg.Button('Create',key='-create-'), sg.Button('Cancel',key='-cancel-')]
        ]
        window = sg.Window("New Predictor",layout)
        event, values = window.read()
        if event == '-create-':
            self._addPredictor(values['-in-'])
            sg.popup_auto_close(f"Predictor added to joint {values['-in-']}", auto_close_duration=2)
            self._refreshTrees()
        window.close()


    def _newKeypoint(self):
        while True:
            new_name = sg.popup_get_text('Enter new keypoint name:',title='Keypoint Creation',size=(20,None))
            if new_name is None:
                break
            if new_name in self.keypoints:
                sg.popup_ok("Keypoint Name Invalid: Already Used")
            elif sum([str(x) in new_name for x in range(10)]) > 0:
                sg.popup_ok("Keypoint Name Invalid: Cannot Contain Numbers")
            else:
                if new_name is None or new_name == '':
                    break
                else:
                    self._addKeypoint(new_name)
                    self._refreshTrees()
                    self.window['-keypoint_name-'].update(values=self.keypoints)
                    break

    def _renameKeypointGUI(self, values):
        if values['-keypoint_name-'] in self.keypoints:
            inital = values['-keypoint_name-']
        else:
            inital = None
        layout = [
            [sg.Txt('Keypoint to Rename:'),sg.Spin(self.keypoints,inital,key='-in-',auto_size_text=False,size=(10,None))],
            [sg.Txt('New Name:'),sg.Input(size=(10,None),key='-new-')],
            [sg.Button('Rename',key='-rename-'), sg.Button('Cancel',key='-cancel-')]
        ]
        window = sg.Window('Keypoint Renaming', layout)
        event = ''
        while event not in ['-cancel-',sg.WINDOW_CLOSED]:
            event, values = window.read()
            if event == '-rename-':
                if values['-new-'] is None:
                    break
                if values['-new-'] in self.keypoints:
                    sg.popup_ok("Keypoint Name Invalid: Already Used")
                elif sum([str(x) in values['-new-'] for x in range(10)]) > 0:
                    sg.popup_ok("Keypoint Name Invalid: Cannot Contain Numbers")
                else:
                    if values['-new-'] is None or values['-new-'] == '':
                        break
                    else:
                        self._renameKeypoint(values['-in-'],values['-new-'])
                        self._refreshTrees()
                        self.window['-keypoint_name-'].update(values=self.keypoints)
                        break
        window.close()

    def _removeKeypointGUI(self, values):
        if values['-keypoint_name-'] in self.keypoints:
            inital = values['-keypoint_name-']
        else:
            inital = None
        layout = [
            [sg.Txt('Keypoint to Remove:'),sg.Spin(self.keypoints,inital,key='-in-',auto_size_text=False,size=(10,None))],
            [sg.Button('Remove',key='-rm-'), sg.Button('Cancel',key='-cancel-')]
        ]
        window = sg.Window('Keypoint Removal', layout)
        event, values = window.read()
        if event == '-rm-':
            keypoint = values['-in-']
            val = sg.popup_ok_cancel(f"Confirm removal of keypoint {keypoint}",
            "This will delete the keypoint and all predictors assocaited with it",
            title="Confirm Removal")
            if val == 'OK':
                self._removeKeypoint(keypoint)
                self._refreshTrees()
                self.window['-keypoint_name-'].update(values=self.keypoints)
                sg.popup_auto_close(f"Keypoint {keypoint} removed.", title="Keypoint Removed", auto_close_duration=2)
        window.close()


    def _setViewMode(self, values):
        modes = ['key','seg','real']
        mode_keys = ['-render_key-','-render_seg-','-render_real-']
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
    
    def _refreshTrees(self):
        self.window['-mesh_tree-'].update(values = self.meshTree.refresh())
        self.window['-joint_tree-'].update(values = self.jointTree.refresh())

    def render(self, values):
        if values['-highlight-']:
            color, depth = self.rend.render_highlight(values['-mesh_tree-'],(30,230,240))
        else:
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


    def _colorVisible(self, image, color):
        return len(np.where(np.all(image == color, axis=-1))[0]) > 0

    def _outlineColorContour(self, image, color, offset=5, outline_color=(30,255,250), thickness=2, detect_from = None):
        if detect_from is None:
            detect_from = image
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        mask[np.where(np.all(detect_from == color, axis=-1))] = 255
        mask = expandRegion(mask, offset)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = np.array(image)
        cv2.drawContours(image, contours, -1, outline_color, thickness)
        return image

    def _circleColor(self, image, color, radius=5, outline_color=(30,255,250), thickness=2, detect_from = None):
        if detect_from is None:
            detect_from = image
        coords = np.where(np.all(detect_from == color, axis=-1))
        avg_y = int(np.mean(coords[0]))
        avg_x = int(np.mean(coords[1]))
        image = np.array(image)
        cv2.circle(image, (avg_x,avg_y), radius, outline_color, thickness)
        return image





class MeshTree(Skeleton):
    def __init__(self, name):
        super().__init__(name)
        self.update()
        self.treedata = sg.TreeData()
        self.urdf_reader = URDFReader()
        self._addMeshes()
        self._addKeypointsToTree()

    def _addMeshes(self):
        for mesh in self.urdf_reader.mesh_names[:-1]:
            self.treedata.insert('',mesh,mesh,[])

    def _addKeypointsToTree(self):
        for keypoint in self.keypoints:
            dat = self.keypoint_data[keypoint]
            pose = list(np.round(np.array(dat['pose']),2))
            parent = dat['parent_link']
            if parent not in self.urdf_reader.mesh_names[:-1]:
                parent = ''
            self.treedata.insert(
                parent,
                keypoint,
                keypoint,
                pose
                )
            
    def refresh(self):
        self.treedata = sg.TreeData()
        self.update()
        self._addMeshes()
        self._addKeypointsToTree()
        return self.treedata

    def __call__(self):
        return sg.Tree(self.treedata,('X','Y','Z','Ro','Pi','Ya'), def_col_width=3, auto_size_columns=False, num_rows=8, key='-mesh_tree-')



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

    def refresh(self):
        self.treedata = sg.TreeData()
        self.update()
        self._addJoints()
        self._addJointPredictors()
        return self.treedata

    def __call__(self):
        return sg.Tree(self.treedata,('From','To','Length','Offset'),key='-joint_tree-',num_rows=6)
