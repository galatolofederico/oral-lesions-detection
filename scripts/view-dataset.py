import cv2
import sys

from detectron2.utils.visualizer import Visualizer
from src.utils import register_dataset, get_catalogs
from src.arguments import get_parser

parser = get_parser()
parser.add_argument("--dataset", type=str, default="train_dataset")
args = parser.parse_args()

def view_dataset(dataset):
    catalogs = get_catalogs(dataset)
    for d in catalogs["dicts"]:
        print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=catalogs["metadata"], scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("dataset", vis.get_image()[:, :, ::-1])
        if cv2.waitKey(0) == ord("q"):
            sys.exit(0)
        print()

register_dataset(args)
view_dataset(args.dataset)