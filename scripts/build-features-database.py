import cv2
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import json

from src.utils import create_cfg, get_catalogs, register_dataset, extract_features
from src.arguments import get_parser
from src.dataloader import get_dataloader

from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.structures.boxes import Boxes


parser = get_parser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

args.data_augmentation = "none"
args.sampler = "TrainingSampler"

register_dataset(args)
cfg = create_cfg(args)

cfg.MODEL.WEIGHTS = args.model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
model = predictor.model

def extract_dataset(dataset_name, dataset_file, database):
    dataset_size = len(json.load(open(dataset_file))["images"])
    data_loader = get_dataloader(cfg, args, dataset_name)
    extracted = 0
    for batch in data_loader:
        for elem in batch:
            file_name = elem["file_name"].split("/")[-1]
            if file_name not in database:
                database[file_name] = list()
                features = extract_features(model, elem["image"], elem["instances"].gt_boxes.to(model.device))
                for roi, features_vector, type in zip(elem["instances"].gt_boxes, features, elem["instances"].gt_classes):
                    database[file_name].append(dict(
                        roi=roi.tolist(),
                        features=features_vector.tolist(),
                        type=type.item()
                    ))
                extracted += 1
                print("Extracted: %s  progress: %.3f%%" % (elem["file_name"], 100*extracted/dataset_size))
            if extracted == dataset_size:
                return

train_database = dict()
test_database = dict()
for dataset_name, dataset_file, database in zip(["train", "test"], ["datasets/lesions/train.json", "datasets/lesions/test.json"], [train_database, test_database]):
    print("Extrating dataset: %s" % (dataset_name, ))
    extract_dataset(dataset_name, dataset_file, database)


print("Saving %s" % (args.output, ))

json.dump(train_database, open(os.path.join(args.output, "train-features.json"), "w"))
json.dump(test_database, open(os.path.join(args.output, "test-features.json"), "w"))
json.dump({**train_database, **test_database}, open(os.path.join(args.output, "features.json"), "w"))

sys.exit()

