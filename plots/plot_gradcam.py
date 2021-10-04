import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt
from types import SimpleNamespace

from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from detectron2 import model_zoo

from plots.gradcam.detectron2_gradcam import Detectron2GradCAM


def plot_gradcam(**kwargs):
    kwargs = SimpleNamespace(**kwargs)
    
    config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg_list = [
    "MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(kwargs.th),
    "MODEL.ROI_HEADS.NUM_CLASSES", "3",
    "MODEL.WEIGHTS", kwargs.model
    ]

    metadata = Metadata()
    metadata.set(
        evaluator_type="coco",
        thing_classes=["neoplastic", "aphthous", "traumatic"],
        thing_dataset_id_to_contiguous_id={"1": 0, "2": 1, "3": 2}
    )


    cam_extractor = Detectron2GradCAM(config_file, cfg_list)
    image_dict, cam_orig = cam_extractor.get_cam(img=kwargs.file, target_instance=kwargs.instance, layer_name=kwargs.layer, grad_cam_type="GradCAM++")

    with torch.no_grad():
        plt.figure(figsize=(kwargs.fig_h/kwargs.fig_dpi, kwargs.fig_w/kwargs.fig_dpi), dpi=kwargs.fig_dpi)
        v = Visualizer(image_dict["image"], metadata, scale=1.0)
        img = image_dict["output"]["instances"][kwargs.instance]
        out = v.draw_instance_predictions(img.to("cpu"))

        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(out.get_image(), interpolation='none')
        plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)

        if kwargs.output == "":
            plt.show()
        else:
            plt.savefig(kwargs.output, dpi=kwargs.fig_dpi, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=str, default="backbone.bottom_up.res5.2.conv3")
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--fig-h", type=int, default=1080)
    parser.add_argument("--fig-w", type=int, default=720)
    parser.add_argument("--fig-dpi", type=int, default=100)

    args = parser.parse_args()

    plot_gradcam(**vars(args))