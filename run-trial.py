import optuna
import copy
import argparse
import os

from train import train, cleanup
from src.utils import create_cfg, register_dataset

def sample_params(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    rpn_loss_weight = trial.suggest_uniform("rpn_loss_weight", 0, 1)
    roi_heads_loss_weight = trial.suggest_uniform("roi_heads_loss_weight", 0, 1)
    
    rois_per_image = trial.suggest_categorical("rois_per_image", [32, 64, 128, 256, 512])
    
    hp = dict(
        lr=lr,
        rpn_loss_weight=rpn_loss_weight,
        roi_heads_loss_weight=roi_heads_loss_weight,
        rois_per_image=rois_per_image
    )

    if args.data_augmentation == "full":
        random_brightness = trial.suggest_uniform("random_brightness", 0, 1)
        random_contrast = trial.suggest_uniform("random_contrast", 0, 1)
        hp.update(dict(
            random_brightness=random_brightness,
            random_contrast=random_contrast
        ))
    if args.data_augmentation == "full" or args.data_augmentation == "crop-flip":
        random_crop = trial.suggest_uniform("random_crop", 0.1, 1)
        hp.update(dict(
            random_crop=random_crop
        ))
    
    if args.sampler == "RepeatFactorTrainingSampler":
        repeat_factor_th = trial.suggest_uniform("repeat_factor_th", 0.1, 1)
        hp.update(dict(
            repeat_factor_th=repeat_factor_th
        ))

    return hp

def experiment(args):
    cfg = create_cfg(args)
    trainer, results, accuracy = train(args, cfg)
    return accuracy


def objective(trial):
    hyperparameters = sample_params(trial)
    trial_args = copy.deepcopy(args)

    vars(trial_args).update(hyperparameters)
    trial_args.name = str(trial.number)
    
    register_dataset(trial_args)

    results = []
    for i in range(0, args.optuna_replicas):
        results.append(experiment(trial_args))

    cleanup(trial_args)
    return sum(results)/len(results)


parser = argparse.ArgumentParser()

parser.add_argument('--study-name', type=str, required=True)

args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=os.environ["OPTUNA_STORAGE"])

vars(args).update(study.user_attrs["args"])
study.optimize(objective, n_trials=1)