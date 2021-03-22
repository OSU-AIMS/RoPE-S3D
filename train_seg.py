#Show sample
import pixellib
from pixellib.custom_train import instance_custom_training
import requests
import os
from robotpose.dataset import Dataset
import robotpose.paths as p
import argparse



def train(dataset, skeleton, batch):
    ds = Dataset(dataset, skeleton, load_seg=False, load_ply=False)

    default_model_path = r'models/segmentation/mask_rcnn_coco.h5'

    if not os.path.isfile(default_model_path):
        print("Base model not found.\nDownloading...")
        url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"
        r = requests.get(url, allow_redirects=True)
        open(default_model_path, 'wb').write(r.content)

    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes = 1, batch_size = batch)
    train_maskrcnn.load_pretrained_model(default_model_path)

    #Train
    train_maskrcnn.load_dataset(ds.seg_anno_path)
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = p.SEG_MODELS)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('--batch',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    args = parser.parse_args()

    train(args.dataset, args.skeleton, args.batch)