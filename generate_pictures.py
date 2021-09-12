# Move out of assets to use



from typing import Union
import cv2
import numpy as np
import tensorflow as tf
from pixellib.instance import custom_segmentation
from scipy.interpolate import interp1d


from robotpose.training.models import ModelManager

from robotpose.utils import str_to_arr, color_array
from robotpose import Dataset, DatasetRenderer
from robotpose.urdf import URDFReader





class Extractor():
    def __init__(self, ds, idx) -> None:

        self.idx = idx

        self.u_reader = URDFReader()

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])

        mm = ModelManager()
        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model(mm.dynamicLoad(dataset=ds))

        

        self.ds = Dataset(ds)
        self.color, self.depth = self.ds.og_img[idx], self.ds.depthmaps[idx]

        self.ds_render = DatasetRenderer(ds,'real')

        self._segment()
        self._underlayColor()
        self._render()
        self.save()


    def _render(self):
        self.render_color, self.render_depth = self.ds_render.render_at(self.idx)


    def _underlayColor(self):
        ALPHA = 0.7
        self.seg_depth_colored = cv2.addWeighted(color_array(self.seg_depth),ALPHA,self.color,1-ALPHA,0)


    def _segment(self):
        r, self.segmented = self.seg.segmentImage(self.color.copy(), process_frame=True)
        segmentation_data = self._reorganize_by_link(r)

        dilate_by = 8
        erode_by = 7

        dilate_by, erode_by = np.ones((dilate_by,dilate_by)), np.ones((erode_by,erode_by))

        # Isolate depth to be only where robot body is
        new = np.zeros((self.depth.shape))
        for k in segmentation_data:
            new += segmentation_data[k]['mask']
        new = cv2.erode(cv2.dilate(new,dilate_by),erode_by)
        self.seg_depth = self.depth * new.astype(bool).astype(float)

    def save(self):
        cv2.imwrite('assets/01_color.png',self.color)
        cv2.imwrite('assets/02_depth.png',color_array(self.depth))
        cv2.imwrite('assets/03_seg_color.png',self.segmented)
        cv2.imwrite('assets/04_seg_depth.png',self.seg_depth_colored)
        cv2.imwrite('assets/05_rend_color.png',self.render_color)
        cv2.imwrite('assets/06_rend_depth.png',color_array(self.render_depth))

    def _reorganize_by_link(self, data: dict) -> dict:
        out = {}
        for idx in range(len(data['class_ids'])):
            id = data['class_ids'][idx]
            if id not in data['class_ids'][:idx]:
                out[self.classes[id]] = {
                    'confidence':data['scores'][idx],
                    'mask':data['masks'][...,idx]
                    }
            else:
                out[self.classes[id]]['mask'] += data['masks'][...,idx]
                out[self.classes[id]]['confidence'] = max(out[self.classes[id]]['confidence'], data['scores'][idx])
        return out



if __name__ == "__main__":
    a = Extractor('set91',51)