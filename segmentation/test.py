import pixellib
from pixellib.instance import custom_segmentation, instance_segmentation
import cv2
import numpy as np


segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 1, class_names= ["BG", "mh5"])
segment_image.load_model(r'mask_rcnn_models\mask_rcnn_model.044-0.034999.h5') 
r, output = segment_image.segmentImage("sample.png", extract_segmented_objects=True)
#segment_image.process_video('test_vid.avi', show_bboxes=True, output_video_name='vid_out.avi', frames_per_second=12.5)
cv2.imshow("test",output)
cv2.waitKey(0)

cv2.imshow(f"test",r['extracted_objects'])
cv2.waitKey(0)
print(np.asarray(r['extracted_objects']).shape)

