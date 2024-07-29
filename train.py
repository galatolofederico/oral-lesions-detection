import os
import numpy as np
import tempfile
import shutil
import atexit

from detectron2.engine.hooks import CallbackHook, EvalHook

from src.utils import create_cfg, register_dataset, get_catalogs
from src.arguments import get_parser
from src.report import build_report
from src.trainer import Trainer

def get_trainer(args, cfg, callback=None):
    trainer = Trainer(cfg, args)
    trainer.resume_or_load(resume=False)

    if callback is not None:
        trainer.register_hooks([CallbackHook(after_step=callback)])

    for h in trainer._hooks:
        if args.skip_coco_eval and type(h) == EvalHook:
            trainer._hooks.remove(h)

    return trainer


def train(args, cfg, callback=None):
    if args.wandb:
        if args.wandb_entity == "":
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args), sync_tensorboard=True)
        else:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args), entity=args.wandb_entity, sync_tensorboard=True)
            
    if args.wandb_save_dir != "":
        import wandb
        assert args.wandb
        args.output_folder = os.path.join(args.wandb_save_dir, wandb.run.id)
        cfg.OUTPUT_DIR = args.output_folder
    
    if args.output_folder != "":
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    else:
        args.tmp_dir = True
        args.output_folder = tempfile.mkdtemp()
        cfg.OUTPUT_DIR = args.output_folder

    
    trainer = get_trainer(args, cfg, callback=callback)
    trainer.train()
    
    train_report, train_classification_lists = build_report(args, "train_dataset", "train")
    train_accuracy = train_report["train/results/accuracy"]

    validation_report, validation_classification_lists = build_report(args, "validation_dataset", "validation")
    validation_accuracy = validation_report["validation/results/accuracy"]

    test_report, test_classification_lists = build_report(args, "test_dataset", "test")
    test_accuracy = test_report["test/results/accuracy"]

    if args.wandb:
        import wandb
        wandb.log(train_report)
        wandb.log(validation_report)
        wandb.log(test_report)

    if not args.skip_coco_eval:
        print("Train Results:")
        print(trainer._last_eval_results)


    if args.wandb:
        import wandb
        class_names = get_catalogs("train_dataset")["metadata"].thing_classes
        wandb.sklearn.plot_confusion_matrix(train_classification_lists["true"], train_classification_lists["pred"], class_names)
        wandb.sklearn.plot_confusion_matrix(validation_classification_lists["true"], validation_classification_lists["pred"], class_names)
        wandb.sklearn.plot_confusion_matrix(test_classification_lists["true"], test_classification_lists["pred"], class_names)


    return trainer, None if args.skip_coco_eval else trainer._last_eval_results, validation_accuracy

def cleanup(args):
    # Wandb fork workaround
    if args.tmp_dir:
        assert args.output_folder[:5] == "/tmp/"
        print("Eliminating %s in 600 seconds (wandb workaround)" % (args.output_folder))
        os.system("bash -c 'sleep 600 && rm -rf %s' &" % (args.output_folder))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    register_dataset(args)
    cfg = create_cfg(args)

    trainer, results, accuracy = train(args, cfg)
    cleanup(args)