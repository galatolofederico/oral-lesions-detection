import cv2
import sys
import argparse
import numpy as np
import os
import torch
import copy

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.utils import create_cfg, get_catalogs, register_dataset, get_dataset_name, get_predictor
from src.arguments import get_parser
from src.dataloader import get_dataloader

from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data.detection_utils import annotations_to_instances

def get_polygon(bbox, style="coco"):
    if style == "coco":
        x0, y0, w, h = bbox
        x0, y0, x1, y1 = x0, y0, x0+w, y0+h
    elif style == "coords":
        x0, y0, x1, y1 = bbox
    else:
        raise Exception("Unknown bbox style %s" % (style))
    
    return Polygon([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])



def get_matching_prediction(bbox, output, debug=False):
    if debug: plt.clf()
    gt_bbox = get_polygon(bbox, "coords")
    classes = []
    for i, (pred_bbox, pred_class, score, tag) in enumerate(zip(output["instances"].pred_boxes, output["instances"].pred_classes, output["instances"].scores, output["instances"].tags)):
        pred_class = pred_class.item()
        pred = get_polygon(pred_bbox, "coords")
        if debug: plt.plot(*gt_bbox.exterior.xy, c="g")
        if debug: plt.plot(*pred.exterior.xy, c="r")
        if gt_bbox.intersection(pred).area > gt_bbox.area*0.1:
            classes.append(pred_class)
            output["instances"].tags[i] = 1
    if debug:
        plt.draw()
        plt.pause(0.001)
    if len(classes) == 0:
        return None
    elif len(classes) == 1:
        return classes[0]
    else:
        for c in classes[1:]:
            if c != classes[0]:
                return None
        return classes[0]


def eval_one(instances, output, metrics, file_name=None, debug=False):
    output["instances"].tags = np.zeros(len(output["instances"]))

    for gt_class, gt_bbox in zip(instances.gt_classes, instances.gt_boxes):

        pred_class = get_matching_prediction(gt_bbox, output, debug)

        if pred_class is None:
            if debug: print("ERRORE DETECTION (LESIONE NON RICONOSCIUTA) (was: %d)" % (gt_class, ))
            metrics["detection_matrix"][1] += 1
            metrics["errors_matrix"][gt_class] += 1
        else:
            if debug: print("DETECTION CORRETTA (classe effettiva: %d classe predetta: %d)" % (gt_class, pred_class))
            metrics["detection_matrix"][0] += 1
            metrics["classification_lists"]["pred"].append(pred_class)
            metrics["classification_lists"]["true"].append(gt_class)
    
    if (output["instances"].tags == 0).any():
        for c, t in zip(output["instances"].pred_classes, output["instances"].tags):
            if t == 0:
                if debug: print("DETECTION FALSO POSITIVO")
                metrics["false_positives"][c] += 1


    if debug:
        cfg = create_cfg(args)
        cfg.MODEL.WEIGHTS = args.model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th
        catalogs = get_catalogs("test_dataset")

        img_dict = None
        for d in catalogs["dicts"]:
            if d["file_name"] == file_name:
                img_dict = d
                break
        img = cv2.imread(file_name)

        pred_v = Visualizer(img[:, :, ::-1], catalogs["metadata"], scale=1)
        gt_v = Visualizer(img[:, :, ::-1], catalogs["metadata"], scale=1)
        
        pred_v = pred_v.draw_instance_predictions(output["instances"].to("cpu"))
        gt_v = gt_v.draw_dataset_dict(img_dict)

        pred = pred_v.get_image()[:, :, ::-1]
        gt = gt_v.get_image()[:, :, ::-1]

        pred = cv2.resize(pred, (800, 800))
        gt = cv2.resize(gt, (800, 800))
        
        img = np.hstack((pred, gt))

        cv2.imshow("img", img)

        print("")

    if cv2.waitKey(0) == ord("q"):
        sys.exit(0)


def evaluate(args):
    args.data_augmentation = "none"
    register_dataset(args)

    cfg = create_cfg(args)

    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.th
    
    predictor_cls = get_predictor(args)
    predictor = predictor_cls(cfg)
    
    #dataloader = get_dataloader(cfg, args, args.dataset)
    catalogs = get_catalogs(args.dataset)

    metrics = dict(
        detection_matrix=np.zeros(2),
        classification_lists=dict(pred=[], true=[]),
        errors_matrix=np.zeros(len(catalogs["metadata"].thing_classes)),
        false_positives=np.zeros(len(catalogs["metadata"].thing_classes))
    )

    if args.debug:
        plt.ion()
        plt.show()

    for elem in catalogs["dicts"]:
        if args.skip_test: break
        image = cv2.imread(elem["file_name"])
        
        prediction = predictor(image)
        
        instances = annotations_to_instances(elem["annotations"], (elem["height"], elem["width"]))
        eval_one(instances, prediction, metrics, elem["file_name"], args.debug)
    
    return metrics

if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="test_dataset")
    parser.add_argument("--th", type=float, default=.5)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    metrics = evaluate(args)

    detection_matrix = metrics["detection_matrix"]
    classification_lists = metrics["classification_lists"]
    errors_matrix = metrics["errors_matrix"]
    false_positives = metrics["false_positives"]

    classification_matrix = confusion_matrix(classification_lists["true"], classification_lists["pred"])

    print("Detections T/F")
    print(detection_matrix)
    print("Detected classification")
    print(classification_matrix)
    print("GT Detection Errors")
    print(errors_matrix)
    print("Class false positives")
    print(false_positives)