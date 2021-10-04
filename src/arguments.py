import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-folder", type=str, required=True)
    
    parser.add_argument("--base-model", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--skip-coco-eval", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    
    parser.add_argument("--images-per-batch", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--data-augmentation", type=str, default="full", choices=["full", "crop-flip", "none"])

    parser.add_argument("--random-brightness", type=float, default=0.2)
    parser.add_argument("--random-contrast", type=float, default=0.2)
    parser.add_argument("--random-crop", type=float, default=0.2)

    parser.add_argument("--rpn-loss-weight", type=float, default=1.0)
    parser.add_argument("--roi-heads-loss-weight", type=float, default=1.0)

    parser.add_argument("--rois-per-image", type=int, default=512)
    
    parser.add_argument("--sampler", type=str, default="TrainingSampler", choices=["TrainingSampler", "RepeatFactorTrainingSampler"])
    parser.add_argument("--repeat-factor-th", type=float, default=0.3)
    
    parser.add_argument("--report-each", type=int, default=100)
    parser.add_argument("--output-folder", type=str, default="")

    parser.add_argument("--wandb-project", type=str, default="oral-ai")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-tag", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--tmp-dir", action="store_true")
    
    return parser