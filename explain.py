import cv2
import sys
import argparse
import numpy as np
import os
import json
import torch
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from scipy import spatial
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import Metadata
from detectron2.structures.boxes import Boxes
from detectron2.structures import Instances

from src.utils import extract_features, forward_model_full
from plots.plot_pca_point import plot_pca_point
from plots.plot_histogram_dist import plot_histogram_dist

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--pca-model", type=str, required=True)
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--features-database", type=str, required=True)
parser.add_argument("--dataset-folder", type=str, required=True)
parser.add_argument("--output", type=str, default="diagnosis.pdf")
parser.add_argument("--tmp-folder", type=str, default="/tmp/build_diagnosis")
parser.add_argument("--th", type=float, default=.5)
parser.add_argument("--distance", type=str, default="cosine")
parser.add_argument("--top-k", type=int, default=3)


args = parser.parse_args()

if not os.path.exists(args.tmp_folder):
    os.mkdir(args.tmp_folder)


def load_model(args):
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
    model = predictor.model

    return dict(
        predictor=predictor,
        model=model,
        metadata=metadata
    )




def compute_similarities(features, database):
    similarities = dict()
    dist_fn = getattr(spatial.distance, args.distance)
    for file_name, elems in  database.items():
        for elem in elems:
            similarities[file_name] = dict(
                dist=dist_fn(elem["features"], features),
                file_name=file_name,
                box=elem["roi"],
                type=elem["type"]
            )
    similarities = OrderedDict(sorted(similarities.items(), key=lambda e: e[1]["dist"]))
    return similarities


def draw_box(file_name, box, type, model):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (800, 800))

    pred_v = Visualizer(img[:, :, ::-1], model["metadata"], scale=1)
    instances = Instances((800, 800), pred_boxes=Boxes(torch.tensor(box).unsqueeze(0)), pred_classes=torch.tensor([type]))
    pred_v = pred_v.draw_instance_predictions(instances)

    pred = pred_v.get_image()[:, :, ::-1]
    pred = cv2.resize(pred, (800, 800))

    return pred


def explain(file, model):
    database = json.load(open(args.features_database))
    cv_img = cv2.imread(file)
    cv_img = cv2.resize(cv_img, (800, 800))

    img = torch.as_tensor(cv_img.astype("float32").transpose(2, 0, 1))
    instances = forward_model_full(model["model"], img)
    
    instances.remove("pred_masks")
    pred_v = Visualizer(cv_img[:, :, ::-1], model["metadata"], scale=1)
    pred_v = pred_v.draw_instance_predictions(instances.to("cpu"))

    pred = pred_v.get_image()[:, :, ::-1]
    pred = cv2.resize(pred, (1024, 1024))

    cv2.imwrite(os.path.join(args.tmp_folder, "prediction.png"), pred)

    lesions = []
    for i, (box, type, scores, features) in tqdm(enumerate(zip(instances.pred_boxes, instances.pred_classes, instances.probs, instances.features)), total=len(instances)):
        lesion = dict()

        lesion_img = draw_box(file, box.cpu(), type, model)
        cv2.imwrite(os.path.join(args.tmp_folder, "explain_%d.png" % (i, )), lesion_img)

        healthy_prob = scores[-1].item()
        scores = scores[:-1]
        features = features.tolist()

        plot_pca_point(point=features, features_database=args.features_database, pca_model=args.pca_model, output=os.path.join(args.tmp_folder, "scatter_%d.png" % (i, )), fig_h=800, fig_w=600, fig_dpi=100)

        plot_histogram_dist(point=features, features_database=args.features_database, output=os.path.join(args.tmp_folder, "hist_%d.png" % (i, )), fig_h=800, fig_w=600, fig_dpi=100)
        

        lesion["healthy_prob"] = healthy_prob

        lesion["lesion"] = os.path.join(args.tmp_folder, "explain_%d.png" % (i, ))
        lesion["scatter"] = os.path.join(args.tmp_folder, "scatter_%d.png" % (i, ))
        lesion["hist"] = os.path.join(args.tmp_folder, "hist_%d.png" % (i, ))
        

        lesion["classes"] = dict()
        for c_id, (c_name, c_score) in enumerate(zip(["neoplastic", "aphthous", "traumatic"], scores)):
            lesion["classes"][c_name] = dict()
            lesion["classes"][c_name]["score"] = c_score.item()

            similarities = compute_similarities(features, database)
            similarities = OrderedDict(filter(lambda i: i[1]["type"] == c_id, similarities.items()))

            lesion["classes"][c_name]["cases"] = list()
            for j, result in zip(range(0, args.top_k), similarities):
                elem = similarities[result]
                img = draw_box(os.path.join(args.dataset_folder, "images", elem["file_name"]), elem["box"], elem["type"], model)
                cv2.imwrite(os.path.join(args.tmp_folder, "explain_%d_%d_%d.png" % (i, c_id, j)), img)
                lesion["classes"][c_name]["cases"].append(
                    dict(
                        image=os.path.join(args.tmp_folder, "explain_%d_%d_%d.png" % (i, c_id, j)),
                        dist=elem["dist"]
                    )
                )

        lesions.append(lesion)
    
    return lesions


def build_files(file, model):
    img = cv2.imread(file)
    img = cv2.resize(img, (800, 800))
    cv2.imwrite(os.path.join(args.tmp_folder, "input.png"), img)

    explain_files = explain(file, model)

    return dict(
        input=os.path.join(args.tmp_folder, "input.png"),
        prediction=os.path.join(args.tmp_folder, "prediction.png"),
        explain=explain_files
    )


def build_diagnosis(files, tmp_folder):
    tex = r"""
    \documentclass[12pt]{article}

    \usepackage{graphicx}
    \usepackage{caption}
    \usepackage{booktabs}
    \usepackage{subfig}
    \usepackage[export]{adjustbox}
    \usepackage[margin=0.5in]{geometry}
    \usepackage{placeins}
    \usepackage{float}

    \begin{document}
    \section{Overview}

    \begin{figure}[H]
    \centering
    \subfloat[Input]{
    \includegraphics[width=0.4\textwidth,valign=c]{%s}
    }
    \subfloat[Prediction]{
    \includegraphics[width=0.4\textwidth,valign=c]{%s}        
    }
    \caption{Detection and classification}
    \end{figure}
    
    \section{Lesions}
    In the next pages you will find a detailed explanation for each leasion.
    """ % (files["input"], files["prediction"])

    for i, explain in enumerate(files["explain"]):
        classes = explain["classes"].keys()
        probs = [explain["classes"][c]["score"] for c in classes]
        
        tex += "\\newpage \n \subsection{Lesion N.%d}\n" % (i, )

        tex += r"""
        \subsubsection{Classification}
        \begin{figure}[H]
        \centering
        \subfloat[Lesion N.%d]{
        \includegraphics[width=0.3\textwidth,valign=c]{%s}
        }
        \subfloat[Class Probabilities]{
        \adjustbox{valign=c}{
        \begin{tabular}{lr}
        \toprule
        \textbf{Metric} & \textbf{Score} \\ \midrule
        """ % (i, explain["lesion"])

        for c, p in zip(classes, probs):
            tex += "%s & %.3f \\%%\\\\\n" % (c, 100*p)
        tex += "%s & %.3f \\%%\\\\ \\bottomrule\n" % ("healthy tissue", 100*explain["healthy_prob"])

        tex += r"""
        \end{tabular}
        }
        }
        \caption{Lesion N.%d Classification}
        \end{figure}
        

        \subsubsection{Distances}

        \begin{figure}[H]
            \centering
            \subfloat[Lesion N.%d Scatter]{
                \includegraphics[width=0.5\textwidth,valign=c]{%s}
            }
            \subfloat[Lesion N.%d Histogram]{
                \includegraphics[width=0.5\textwidth,valign=c]{%s}
            }
        \end{figure}

        \newpage
        \subsubsection{Lesion N.%d Nearest Neighbors}
        """ % (i, i, explain["scatter"], i, explain["hist"], i)


        max_class = probs.index(max(probs))
        for ic, (c, p) in enumerate(zip(classes, probs)):
            #if ic != max_class:
            #    continue
            tex += r"""
            \begin{figure}[H]
            \centering
            """
            for case in explain["classes"][c]["cases"]:
                tex += r"""
                \subfloat[dist: %.3f]{
                \includegraphics[width=0.3\textwidth,valign=c]{%s}
                }""" % (case["dist"], case["image"])

            tex += r"""
            \caption{Lesion N.%d 3-NN for %s (prob: %.3f\%%)}
            \end{figure}
                """ % (i, c, p*100)
        tex += "\\newpage"

    tex += r"\end{document}"

    report = open(os.path.join(tmp_folder, "report.tex"), "w")
    report.write(tex)
    report.close()


model = load_model(args)
files = build_files(args.file, model)

build_diagnosis(files, args.tmp_folder)