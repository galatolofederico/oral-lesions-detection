import json
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm

from src.utils import create_cfg, register_dataset
from src.arguments import get_parser

from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import ImageList
import detectron2.data.transforms as T

parser = get_parser()

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)


args = parser.parse_args()


def most_frequent(arr): 
    return max(set(arr), key = arr.count) 


dataset = json.load(open(args.dataset, "r"))

register_dataset(args)
cfg = create_cfg(args)

cfg.MODEL.WEIGHTS = args.model

model = build_model(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

model.eval()

transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

category_names = dict()
images_ids = dict()
images = dict()


for image in dataset["images"]:
    images_ids[image["id"]] = image

for category in dataset["categories"]:
    category_names[category["id"]-1] = category["name"]

for annotation in dataset["annotations"]:
    if annotation["image_id"] not in images:
        images[annotation["image_id"]] = dict(annotations=[], c=None, id=annotation["image_id"])
    images[annotation["image_id"]]["annotations"].append(annotation)

for id, image in images.items():
    categories = list(map(lambda a: a["category_id"], image["annotations"]))
    image["c"] = most_frequent(categories)

all_features = None
all_classes = None

for i, (id, image) in tqdm(enumerate(images.items()), total=len(images)):
    filename = images_ids[image["id"]]["file_name"]
    img = cv2.imread(args.dataset_folder+"/images/"+filename)
    
    if cfg.INPUT.FORMAT == "RGB":
        img = img[:, :, ::-1]
    height, width = img.shape[:2]
    image_tensor = transform_gen.get_transform(img).apply_image(img)
    image_tensor = torch.as_tensor(image_tensor.astype("float32").transpose(2, 0, 1))
    
    image_input = {"image": image_tensor, "height": height, "width": width}
    image_input = model.preprocess_image([image_input])
    
    features = model.backbone(image_input.tensor)
    proposals, _ = model.proposal_generator(image_input, features, None)
    
    instances = model.roi_heads._forward_box(features, proposals)
    mask_features = [features[f] for f in model.roi_heads.in_features]
    mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
    
    pred_classes = [x.detach().cpu().numpy().item() for x in instances[0].pred_classes]
    scores = [x.detach().cpu().numpy().item() for x in instances[0].scores]
    
    for f, c, s in zip(mask_features, pred_classes, scores):
        if s > 0.5:
            ff = f.view(1, f.numel()).detach().cpu().numpy()
            c = np.ones(1)*c
            if all_features is None:
                all_features = ff
                all_classes = c
            else:
                all_features = np.concatenate((all_features, ff))
                all_classes = np.concatenate((all_classes, c))    

pickle.dump(dict(
    features=all_features,
    classes=all_classes,
    categories=category_names,
), open(args.output, "wb"))

print("Ok!")