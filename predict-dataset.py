import cv2
import sys
import argparse
import numpy as np
import os

from src.utils import create_cfg, get_catalogs, register_dataset
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from src.arguments import get_parser

parser = get_parser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset-name", type=str, default="test_dataset")
parser.add_argument("--th", type=float, default=.5)
parser.add_argument("--dump", type=str, default="")

args = parser.parse_args()

register_dataset(args)
cfg = create_cfg(args)

cfg.MODEL.WEIGHTS = args.model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th

predictor = DefaultPredictor(cfg)

counter = 0

def predict(elem, dump=""):
    global counter
    img = cv2.imread(elem["file_name"])

    output = predictor(img)

    pred_v = Visualizer(img[:, :, ::-1], catalogs["metadata"], scale=1)
    gt_v = Visualizer(img[:, :, ::-1], catalogs["metadata"], scale=1)

    pred_v = pred_v.draw_instance_predictions(output["instances"].to("cpu"))
    gt_v = gt_v.draw_dataset_dict(elem)

    pred = pred_v.get_image()[:, :, ::-1]
    gt = gt_v.get_image()[:, :, ::-1]

    pred = cv2.resize(pred, (800, 800))
    gt = cv2.resize(gt, (800, 800))
    
    img = np.hstack((pred, gt))

    if dump == "":
        cv2.imshow("img", img)

        if cv2.waitKey(0) == ord("q"):
            sys.exit(0)
    else:
        cv2.imwrite(os.path.join(args.dump, str(counter)+".jpg"), img)
        print("wrote %d" % (counter))
        counter += 1
    

catalogs = get_catalogs(args.dataset_name)
for d in catalogs["dicts"]:
    print(d)
    predict(d, dump=args.dump)
