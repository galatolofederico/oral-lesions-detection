# Author: Alexander Riedel 
# License: Unlicensed
# Link: https://github.com/alexriedel1/detectron2-GradCAM

from plots.gradcam.gradcam import GradCAM, GradCamPlusPlus
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances

class Detectron2GradCAM():
  """
      Attributes
    ----------
    config_file : str
        detectron2 model config file path
    cfg_list : list
        List of additional model configurations
    root_dir : str [optional]
        directory of coco.josn and dataset images for custom dataset registration
    custom_dataset : str [optional]
        Name of the custom dataset to register
    """
  def __init__(self, config_file, cfg_list, root_dir=None, custom_dataset=None):
      # load config from file
      cfg = get_cfg()
      cfg.merge_from_file(config_file)

      if custom_dataset:
          register_coco_instances(custom_dataset, {}, root_dir + "coco.json", root_dir)
          cfg.DATASETS.TRAIN = (custom_dataset,)
          MetadataCatalog.get(custom_dataset)
          DatasetCatalog.get(custom_dataset)

      if torch.cuda.is_available():
          cfg.MODEL.DEVICE = "cuda"
      else:
          cfg.MODEL.DEVICE = "cpu"

      cfg.merge_from_list(cfg_list)
      cfg.freeze()

      self.cfg =  cfg

  def _get_input_dict(self, original_image):
      height, width = original_image.shape[:2]
      transform_gen = T.ResizeShortestEdge(
          [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
      )
      image = transform_gen.get_transform(original_image).apply_image(original_image)
      image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
      inputs = {"image": image, "height": height, "width": width}
      return inputs

  def get_cam(self, img, target_instance, layer_name, grad_cam_type="GradCAM"):
      """
      Calls the GradCAM++ instance

      Parameters
      ----------
      img : str
          Path to inference image
      target_instance : int
          The target instance index
      layer_name : str
          Convolutional layer to perform GradCAM on
      grad_cam_type : str
          GradCAM or GradCAM++ (for multiple instances of the same object, GradCAM++ can be favorable)

      Returns
      -------
      image_dict : dict
        {"image" : <image>, "cam" : <cam>, "output" : <output>, "label" : <label>}
        <image> original input image
        <cam> class activation map resized to original image shape
        <output> instances object generated by the model
        <label> label of the 
      cam_orig : numpy.ndarray
        unprocessed raw cam
      """
      model = build_model(self.cfg)
      checkpointer = DetectionCheckpointer(model)
      checkpointer.load(self.cfg.MODEL.WEIGHTS)

      image = read_image(img, format="BGR")
      input_image_dict = self._get_input_dict(image)

      if grad_cam_type == "GradCAM":
        grad_cam = GradCAM(model, layer_name)

      elif grad_cam_type == "GradCAM++":
        grad_cam = GradCamPlusPlus(model, layer_name)
      
      else:
        raise ValueError('Grad CAM type not specified')

      with grad_cam as cam:
        cam, cam_orig, output = cam(input_image_dict, target_category=target_instance)
      
      image_dict = {}
      image_dict["image"] = image
      image_dict["cam"] = cam
      image_dict["output"] = output[0]
      image_dict["label"] = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[output[0]["instances"].pred_classes[target_instance]]
      return image_dict, cam_orig