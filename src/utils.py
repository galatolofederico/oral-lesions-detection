import cv2
import sys
import os
import torch
import torch.nn.functional as F

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T


dataset_registered = False
def register_dataset(args):
    global dataset_registered
    if not dataset_registered:        
        register_coco_instances("train_dataset", {}, os.path.join(args.dataset_folder, "train.json"), os.path.join(args.dataset_folder, "images"))
        register_coco_instances("validation_dataset", {}, os.path.join(args.dataset_folder, "validation.json"), os.path.join(args.dataset_folder, "images"))
        register_coco_instances("test_dataset", {}, os.path.join(args.dataset_folder, "test.json"), os.path.join(args.dataset_folder, "images"))
        
        dataset_registered = True


def create_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_model))
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("test_dataset", )
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.base_model)
    cfg.SOLVER.IMS_PER_BATCH = args.images_per_batch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.rois_per_image
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.epochs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(get_catalogs("train_dataset")["metadata"].thing_classes)
    cfg.OUTPUT_DIR = args.output_folder
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = args.rpn_loss_weight
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = args.roi_heads_loss_weight

    return cfg

def get_catalogs(dataset):
    lesioni_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = DatasetCatalog.get(dataset)

    return dict(
        metadata=lesioni_metadata,
        dicts=dataset_dicts
    ) 

def get_dataset_name(which):
    if which == "train":
        return "train_dataset"
    elif which == "test":
        return "test_dataset"
    else:
        raise Exception("%s must be train or test" % (which))


def get_predictor(args):
    from detectron2.engine import DefaultPredictor
    return DefaultPredictor


def extract_features(model, img, box):
    height, width = img.shape[1:3]
    inputs = [{"image": img, "height": height, "width": width}]
    with torch.no_grad():
        img = model.preprocess_image(inputs) 
        features = model.backbone(img.tensor)
        features_ = [features[f] for f in model.roi_heads.box_in_features]

        box_features = model.roi_heads.box_pooler(features_, [box])

        output_features = F.avg_pool2d(box_features, [7, 7])
        output_features = output_features.view(-1, 256)

        return output_features


def forward_model_full(model, cfg, img):
    cv_img = cv2.imread(img)

    height, width = cv_img.shape[:2]
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    image = transform_gen.get_transform(cv_img).apply_image(cv_img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor) 
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_head = model.roi_heads.box_head(box_features)
        predictions = model.roi_heads.box_predictor(box_head)
        
        output_features = F.avg_pool2d(box_features, [7, 7])
        output_features = output_features.view(-1, 256)

        probs = model.roi_heads.box_predictor.predict_probs(predictions, proposals)
        
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)
        
        instances = pred_instances[0]["instances"]

        instances.set("probs", probs[0][pred_inds])
        instances.set("features", output_features[pred_inds])
        
        return instances, cv_img