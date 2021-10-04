import numpy as np
from detectron2.engine.hooks import CallbackHook, EvalHook
from detectron2.engine import DefaultPredictor
import copy
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from src.utils import get_catalogs, create_cfg
from src.evaluate import evaluate

def build_report(args, dataset, prefix):
    class_names = get_catalogs("train_dataset")["metadata"].thing_classes
    

    eval_args = copy.deepcopy(args)
    eval_args.dataset = dataset
    eval_args.model = os.path.join(args.output_folder, "model_final.pth")
    eval_args.debug = False
    eval_args.th = 0.5

    metrics = evaluate(eval_args)
    
    detection_matrix = metrics["detection_matrix"]
    classification_lists = metrics["classification_lists"]
    errors_matrix = metrics["errors_matrix"]
    false_positives = metrics["false_positives"]

    classification_matrix = confusion_matrix(classification_lists["true"], classification_lists["pred"])
    report = None

    try:
        report = classification_report(classification_lists["true"], classification_lists["pred"], target_names=class_names, output_dict=True)
    except:
        print("CANNOT CREATE REPORT, POOR TRAINING?")
        
    detection_accuracy = detection_matrix[0]/np.sum(detection_matrix)
    classification_accuracy = accuracy_score(classification_lists["true"], classification_lists["pred"])
    accuracy = detection_accuracy*classification_accuracy

    print("=====================")
    print("LOG FOR %s" % (prefix))
    print("CLASS NAMES: %s" % (class_names, ))
    print("Detections T/F")
    print(detection_matrix)
    print("Detected classification")
    print(classification_matrix)
    print("GT Detection Errors")
    print(errors_matrix)
    print("Class false positives")
    print(false_positives)
    print("Report")
    print(report)
    print("Accuracy:")
    print(accuracy)
    print("===================")
    
    return {
        prefix+"/results/detection_accuracy": detection_matrix[0]/np.sum(detection_matrix),
        prefix+"/results/accuracy": accuracy,
        prefix+"/results/classification_accuracy": classification_accuracy,
        prefix+"/report": report
    }, classification_lists

