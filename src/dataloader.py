import torch
import copy

from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader, build_detection_test_loader
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.data.samplers import RepeatFactorTrainingSampler


def get_mapper(args, which):
    def train_mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        if args.data_augmentation == "full":
            image, transforms = T.apply_transform_gens([
                T.RandomFlip(),
                T.RandomBrightness(1-args.random_brightness, 1+args.random_brightness),
                T.RandomContrast(1-args.random_contrast, 1+args.random_contrast),
                T.RandomCrop("relative_range", [args.random_crop, 1]),
                T.Resize((800, 800)),
            ], image)
        elif args.data_augmentation == "crop-flip":
            image, transforms = T.apply_transform_gens([
                T.RandomFlip(),
                T.RandomCrop("relative_range", [args.random_crop, 1]),
                T.Resize((800, 800)),
            ], image)
        elif args.data_augmentation == "none":
            image, transforms = T.apply_transform_gens([
                T.Resize((800, 800)),
            ], image)
        else:
            raise Exception("Unknown data augmentation: %s " % args.data_augmentation)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def coco_eval_mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        image, transforms = T.apply_transform_gens([
            T.Resize((800, 800)),
        ], image)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
    

    def test_mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        image, transforms = T.apply_transform_gens([
            T.Resize((800, 800)),
        ], image)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    if which == "train":
        return train_mapper
    elif which == "test":
        return test_mapper
    elif which == "coco-eval":
        return coco_eval_mapper
    else:
        raise Exception("%s one of train/test/coco-eval" % (which, ))


def build_detection_loader(cfg, mapper, which, args):
    assert which in ["train", "coco-eval", "test"]

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN if which == "train" else cfg.DATASETS.TEST,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=None,
    )

    dataset = DatasetFromList(dataset_dicts, copy=False)
    #Compute weights before appling data augmentation
    if args.sampler == "TrainingSampler":
        print("[SAMPLER] Selected TrainingSampler")
        dataset = MapDataset(dataset, mapper)
        sampler = TrainingSampler(len(dataset))
    elif args.sampler == "RepeatFactorTrainingSampler":
        print("[SAMPLER] Selected RepeatFactorTrainingSampler")
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset_dicts, args.repeat_factor_th)
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        dataset = MapDataset(dataset, mapper)
    else:
        raise Exception("Unknown Sampler %s" % args.sampler)

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def get_dataloader(cfg, args, which):
    mapper = get_mapper(args, which)

    if which == "coco-eval":
        return build_detection_test_loader(cfg, "test_dataset", mapper)
    else:
        return build_detection_loader(cfg, mapper, which, args)

