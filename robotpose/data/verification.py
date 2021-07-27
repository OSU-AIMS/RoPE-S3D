# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import re

import cv2
import numpy as np
import PySimpleGUI as sg

from ..constants import (VERIFIER_ALPHA, VERIFIER_COLUMNS, VERIFIER_ROWS,
                         VERIFIER_SCALER, VERIFIER_SELECTED_GAMMA)
from ..simulation import DatasetRenderer
from .building import Builder
from .dataset import Dataset
from typing import List


class Verifier():
    """Allows images to be compared againt their claimed robot pose to elimate faulty data"""


    def __init__(
            self, 
            dataset: str, 
            selected: List[int] = None, 
            thumbnails: np.ndarray = None, 
            overlays: np.ndarray = None
            ):
        """ Create Verifier
        Parameters
        ----------
        dataset : str
            Dataset to verify
        selected, thumbnails, overlays : List[int], np.ndarray, np.ndarray, optional
            If supplied, will run in 'child' mode to confirm the poses to be deleted
        """

        self.ds = Dataset(dataset)
        self.name = dataset
        self.ds_path = self.ds.dataset_path
        
        img_size = [int(x * VERIFIER_SCALER) for x in self.ds.preview_img.shape[1:3]]
        dims = tuple(img_size[::-1])

        # Set parent/child mode
        self.mode = 'parent' if selected is None else 'child'
        if selected is not None: assert thumbnails is not None and overlays is not None

        if self.mode == 'parent':
            # Requires loading all data from scratch
            self.length = self.ds.length
            self.ds_renderer = DatasetRenderer(dataset, 'seg_full')
            self.selected = set()
            og_thumbnails = np.copy(self.ds.preview_img)
            self.thumbnails = np.zeros((self.length,*img_size,3,),np.uint8)
            self.overlays = np.copy(self.thumbnails)

            for idx in range(self.length):
                color,depth = self.ds_renderer.render_at(idx)
                self.overlays[idx] = cv2.resize(color,dims)
                self.thumbnails[idx] = cv2.resize(og_thumbnails[idx],dims)

            del self.ds_renderer
        else:
            # Copy data, do not reload
            self.length = len(thumbnails)
            self.selected = set([x for x in range(len(selected))])
            self.thumbnails = np.copy(thumbnails)
            self.overlays = np.copy(overlays)
            self.parent_selected = selected

        def img_frame(row,column):
            return sg.Image(size=dims,k=f'-img_{row}_{column}-',enable_events=True)

        top = sg.Column(
            [[sg.Text('',k='-range-',size=(15,1),justification='right'),
                sg.Button('Prev',k='-prev-'),
                sg.Button('Next',k='-next-')]],
            justification='right',element_justification='right')

        bottom = sg.Column(
            [[sg.Button('Apply' if selected is None else 'Confirm',k='-apply-'),
                sg.Button('Cancel',k='-cancel-')]]
            ,justification='left',element_justification='left')

        self.layout = [
            [sg.Text('Select Images to Remove' if selected is None else 'Confirm Removal',font='Any 36 bold')],
            *[[img_frame(r,c) for c in range(VERIFIER_COLUMNS)] for r in range(VERIFIER_ROWS)],
            [top],
            [bottom]
        ]

        del self.ds
        


    def run(self):
        self.window = sg.Window('Data Verification', self.layout.copy(), finalize=True)
        self.window.bring_to_front()

        self.start_idx = 0
        self._updateImgs()
        self._updateRange()

        event = ''
        while event not in (sg.WIN_CLOSED,'-cancel-','-apply-'):
            event, values = self.window.read(10)
            if event not in (sg.WIN_CLOSED,'-cancel-','-apply-'):
                self._runEvent(event)
            if event == '-apply-':
                if self.mode == 'parent':
                    ret = self._runChild()
                    if ret is not None:
                        self.window.close()
                        return ret
                    else:
                        event = ''
                else:
                    return self._returnSelected()

        if event in (sg.WIN_CLOSED,'-cancel-'):
            self.window.close()
            return None

        


    def _runEvent(self, event):
        match = re.search(r'img_([0-9]+)_([0-9]+)',event)
        if match:
            r,c = int(match.group(1)), int(match.group(2))

            idx = self.start_idx + r * VERIFIER_COLUMNS + c

            if idx in self.selected:
                self.selected.remove(idx)
            else:
                self.selected.add(idx)

            self._updateImg(r,c)

        elif event == '-next-':
            self._next()
        elif event == '-prev-':
            self._prev()


    def _updateRange(self):
        start = self.start_idx
        if self.mode == 'parent':
            end = min(self.start_idx + VERIFIER_ROWS * VERIFIER_COLUMNS, self.length) - 1
        else:
            end = min(self.start_idx + VERIFIER_ROWS * VERIFIER_COLUMNS, len(self.parent_selected)) - 1
        self.window['-range-'].update(f'{start}-{end}')


    def _runChild(self):

        child = Verifier(self.name, list(self.selected), self.thumbnails[list(self.selected)], self.overlays[list(self.selected)])
        final_selections = child.run()

        if final_selections is not None:
            del child
            bob = Builder()
            bob.remove_idxs(self.ds_path, final_selections)
            return final_selections
        else:
            return None

    def _returnSelected(self):
        self.window.close()
        return np.array(self.parent_selected)[list(self.selected)]


    def _next(self):
        if self.start_idx + VERIFIER_COLUMNS * VERIFIER_ROWS <= self.length:
            self.start_idx += VERIFIER_COLUMNS * VERIFIER_ROWS  

        self._updateImgs()
        self._updateRange()


    def _prev(self):
        self.start_idx -= VERIFIER_COLUMNS * VERIFIER_ROWS
        if self.start_idx < 0:
            self.start_idx = 0

        self._updateImgs()
        self._updateRange()


    def _updateImgs(self):
        if self.mode == 'parent':
            for r in range(VERIFIER_ROWS):
                for c in range(VERIFIER_COLUMNS):
                    self._updateImg(r,c)
        else:
            idxs = list(self.selected)
            idx = self.start_idx
            for r in range(VERIFIER_ROWS):
                for c in range(VERIFIER_COLUMNS):
                    if idx < len(idxs):
                        self._updateImg(r, c, idxs[idx])
                        idx += 1

    def _updateImg(self, row, column, idx = None):
        if idx is None:
            idx = self.start_idx + row * VERIFIER_COLUMNS + column

        if idx < self.length:
            if idx in self.selected:
                g = VERIFIER_SELECTED_GAMMA
            else:
                g = 0

            img = cv2.addWeighted(self.thumbnails[idx], VERIFIER_ALPHA, self.overlays[idx],1-VERIFIER_ALPHA,g)

            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            self.window[f'-img_{row}_{column}-'].update(data=imgbytes)

