import os

from detectron2.engine import DefaultTrainer
from src.dataloader import get_dataloader
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

class Trainer(DefaultTrainer):
    def __init__(self, cfg, args):
        Trainer.args = args
        super(Trainer, self).__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        if cfg.OUTPUT_DIR != "":
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        else:
            output_folder = None
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return get_dataloader(cfg, cls.args, "train")
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return get_dataloader(cfg, cls.args, "coco-eval")