import os
import cv2
import numpy as np
from labelme.label_file import LabelFile
import tempfile
import os

def makeMask(image):
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    mask[np.where(np.all(image != (0,0,0), axis=-1))] = 255
    return mask


def maskImg(image):
    mask = makeMask(image)
    mask_ = np.zeros(image.shape, bool)
    for idx in range(image.shape[2]):
        mask_[:,:,idx] = mask
    mask_img = np.ones(image.shape, np.uint8) * 255
    return mask_img


def makeContours(image):
    thresh = makeMask(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    np.save('contours.npy', contours)
    img = np.copy(image)

    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    return contours



def contourImg(image):
    contours = makeContours(image)
    img = np.copy(image)
    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    return img





def labelSegmentation(act_img, render_img, path = None):

    f = LabelFile()

    if type(act_img) is not str:

        assert path is not None, "Path must be specified if an image file is not used."

        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir,'img.png'), act_img)
            imageData = f.load_image_file(os.path.join(tmpdir,'img.png'))

        act_image_path = 'img.png'
        json_path = path
        if not json_path.endswith('.json'):
            json_path += '.json'
    else:
        imageData = f.load_image_file(act_img)
        act_image_path = act_img
        json_path = act_image_path.replace('.png','.json')

    contour = makeContours(render_img)

    contourlist = np.asarray(contour[0]).tolist()
    contour_data = []
    for point in contourlist:
        contour_data.append(point[0])
    

    shape = {
        "label": "mh5",
        "points": contour_data,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }

    f.save(
        filename = json_path,
        shapes = [shape],
        imagePath = act_image_path,
        imageHeight = 720,
        imageWidth = 1280,
        imageData = imageData
    )