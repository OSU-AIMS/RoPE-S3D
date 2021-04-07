import pixellib
from pixellib.custom_train import instance_custom_training


train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 1)
train_maskrcnn.load_dataset("Robot")
for iou in [0.9,0.95,0.99,0.999,0.9999,0.99999]:
    train_maskrcnn.evaluate_model(r'mask_rcnn_models\mask_rcnn_model.044-0.034999.h5',iou)