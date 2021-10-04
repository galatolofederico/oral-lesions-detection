import cv2
from detectron2.utils.visualizer import Visualizer

from src.dataloader import get_dataloader
from src.utils import create_cfg, register_dataset, get_catalogs
from src.arguments import get_parser
from detectron2.data import detection_utils as utils


parser = get_parser()
parser.add_argument("--dataset", type=str, default="train")
args = parser.parse_args()

def view_dataloader(dataloader, dataset, cfg):
    catalogs = get_catalogs(dataset)
    metadata = catalogs["metadata"]
    for batch in dataloader:
        print("New batch")
        for per_image in batch:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = Visualizer(img, metadata=metadata, scale=1)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == ord("q"):
                sys.exit(0)


register_dataset(args)
cfg = create_cfg(args)
dataloader = get_dataloader(cfg, args, args.dataset)

if args.dataset == "train":
    view_dataloader(dataloader, "train_dataset", cfg)
elif args.dataset == "test":
    view_dataloader(dataloader, "test_dataset", cfg)
else:
    raise Exception("Dataset one of train/test and not '%s'" % (args.dataset))


