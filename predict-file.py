import cv2
import sys
import argparse
import numpy as np
import os

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import Metadata


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--th", type=float, default=.5)

args = parser.parse_args()

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = args.model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th


metadata = Metadata()
metadata.set(
    evaluator_type="coco",
    thing_classes=["neoplastic", "aphthous", "traumatic"],
    thing_dataset_id_to_contiguous_id={"1": 0, "2": 1, "3": 2}
)

predictor = DefaultPredictor(cfg)


def predict(file):
    img = cv2.imread(file)
    output = predictor(img)
    output["instances"].remove("pred_masks")
    pred_v = Visualizer(img[:, :, ::-1], metadata, scale=1)
    pred_v = pred_v.draw_instance_predictions(output["instances"].to("cpu"))

    pred = pred_v.get_image()[:, :, ::-1]
    pred = cv2.resize(pred, (1024, 1024))

    
    if args.output == "":
        cv2.imshow("pred", pred)
        if cv2.waitKey(0) == ord("q"):
            sys.exit(0)
    else:
        cv2.imwrite(args.output, pred)


predict(args.file)