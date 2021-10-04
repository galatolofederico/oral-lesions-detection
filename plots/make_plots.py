import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotszoo


def get_hyperparameters(data_augmentation, sampler):
    hp = ["lr", "rpn_loss_weight", "roi_heads_loss_weight", "rois_per_image"]
    if data_augmentation == "full":
        hp.extend(["random_brightness", "random_contrast"])
    if data_augmentation == "full" or data_augmentation == "crop-flip":
        hp.extend(["random_crop"])
    if sampler == "RepeatFactorTrainingSampler":
        hp.extend(["repeat_factor_th"])
    
    return ["config/"+i for i in hp]


def plot_study():
    query = {"$or": [{"config.wandb_tag": {"$eq": tag}} for tag in args.tags_study_replicas]}
    data = plotszoo.data.WandbData(args.username, args.project, query, verbose=args.verbose)
    data.pull_scalars(force_update=args.update_scalars)


    group_keys = ["config/sampler", "config/data_augmentation"]

    fig, axes = plt.subplots(nrows=2, ncols=2)

    yticks_fn = lambda index: "Sampler: %s Data Augmentation: %s" % (index[0], index[1])

    test_detection_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/test/results/detection_accuracy")
    test_classification_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/test/results/classification_accuracy")

    test_detection_df = test_detection_plot.plot(axes[0][0], title="Test Detection Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)
    test_classification_df = test_classification_plot.plot(axes[0][1], title="Test Classification Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)

    train_detection_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/train/results/detection_accuracy")
    train_classification_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/train/results/classification_accuracy")

    train_detection_df = train_detection_plot.plot(axes[1][0], title="Train Detection Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)
    train_classification_df = train_classification_plot.plot(axes[1][1], title="Train Classification Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)

    test_detection_df.to_excel(os.path.join(args.output_directory, "study/test_detection.xlsx"))
    test_classification_df.to_excel(os.path.join(args.output_directory, "study/test_classification.xlsx"))
    train_detection_df.to_excel(os.path.join(args.output_directory, "study/train_detection.xlsx"))
    train_classification_df.to_excel(os.path.join(args.output_directory, "study/train_classification.xlsx"))
    

    for ax in axes.flatten():
        ax.set_xlim(xmin=0.5)

    fig.set_size_inches(30, 10)
    fig.tight_layout()

    plotszoo.utils.savefig(fig, os.path.join(args.output_directory, "study.png"))


def plot_optimization_history(ax, data, dataset):
        running_max = dict(accuracy=float("-inf"), detection_accuracy=float("-inf"), classification_accuracy=float("-inf"))
        plots=dict(best_accuracy=[], best_detection_accuracy=[], best_classification_accuracy=[], accuracy=[], detection_accuracy=[], classification_accuracy=[])
        plot_index = []
        for i, row in data.scalars.iterrows():
            if row["summary/"+dataset+"/results/accuracy"] > running_max["accuracy"]:
                running_max = dict(
                    accuracy=row["summary/"+dataset+"/results/accuracy"],
                    detection_accuracy=row["summary/"+dataset+"/results/detection_accuracy"],
                    classification_accuracy=row["summary/"+dataset+"/results/classification_accuracy"]
                )
            plots["accuracy"].append(row["summary/"+dataset+"/results/accuracy"])
            plots["detection_accuracy"].append(row["summary/"+dataset+"/results/detection_accuracy"])
            plots["classification_accuracy"].append(row["summary/"+dataset+"/results/classification_accuracy"])
           
            plots["best_accuracy"].append(running_max["accuracy"])
            plots["best_detection_accuracy"].append(running_max["detection_accuracy"])
            plots["best_classification_accuracy"].append(running_max["classification_accuracy"])

            plot_index.append(i)
        
        ax.plot(plot_index, plots["best_accuracy"], "k", label="Best "+dataset+" Accuracy")
        ax.plot(plot_index, plots["best_detection_accuracy"], "b--", label="Best "+dataset+" Detection Accuracy")
        ax.plot(plot_index, plots["best_classification_accuracy"], "g--", label="Best "+dataset+" Classification Accuracy")

        ax.scatter(plot_index, plots["accuracy"], c="k", alpha=0.5)
        ax.scatter(plot_index, plots["detection_accuracy"], c="b", alpha=0.5)
        ax.scatter(plot_index, plots["classification_accuracy"], c="g", alpha=0.5)

        ax.legend(loc="lower right")
        


def plot_optimization():
    for tag, params in args.tags_optimization.items():
        query = {"config.wandb_tag": {"$eq": tag}}

        parameters = get_hyperparameters(**params)
        parameters.extend(["summary/train/results/detection_accuracy", "summary/train/results/classification_accuracy"])

        data = plotszoo.data.WandbData(args.username, args.project, query, verbose=args.verbose)
        data.pull_scalars(force_update=args.update_scalars)
        assert len(data.scalars) > 0, "No data, check the tag name"
        data.pull_series(force_update=args.update_series)

        data.astype(["summary/train/results/accuracy", "summary/train/results/detection_accuracy", "summary/train/results/classification_accuracy"], float)
        data.dropna(["summary/train/results/accuracy"])        

        data.create_scalar_from_series("start_time", lambda s: s["_timestamp"].min())

        fig, axes = plt.subplots(1, len(parameters), sharey=False)

        parallel_plot = plotszoo.scalars.ScalarsParallelCoordinates(data, parameters, "summary/train/results/accuracy")

        parallel_plot.plot(axes)

        fig.set_size_inches(32, 10)
        plotszoo.utils.savefig(fig, os.path.join(args.output_directory, tag, "optim_parallel.png"))

        fig, ax = plt.subplots(2, 1)

        plot_optimization_history(ax[0], data, "train")
        plot_optimization_history(ax[1], data, "test")
        
        fig.set_size_inches(20, 10)
        plotszoo.utils.savefig(fig, os.path.join(args.output_directory, tag, "optim_history.png"))

        parameters.remove("summary/train/results/detection_accuracy")
        parameters.remove("summary/train/results/classification_accuracy")

        args_names = [p.split("/")[1].replace("_","-") for p in parameters]
        best_run = data.scalars["summary/train/results/accuracy"].idxmax()
        best_args = "".join(["--%s %s " % (n, data.scalars[k][best_run]) for n, k in zip(args_names, parameters)])
        best_args += "".join(["--%s %s " % (k.replace("_", "-"), v) for k, v in params.items()])
        print(best_run)
        print("Tag: %s" % tag)
        print(data.scalars.loc[best_run][["summary/train/results/detection_accuracy", "summary/train/results/classification_accuracy"]])
        print("HP: %s" % best_args)
        print()
        
        best_args_f = open(os.path.join(args.output_directory, tag, "best_args.txt"), "w")
        best_args_f.write(best_args)
        best_args_f.close()

def plot_replicas():
    query = {"$or": [{"config.wandb_tag": {"$eq": tag}} for tag in args.tags_best_replicas]}
    data = plotszoo.data.WandbData(args.username, args.project, query, verbose=args.verbose)
    data.pull_scalars(force_update=args.update_scalars)

    group_keys = ["config/sampler"]

    fig, axes = plt.subplots(nrows=2, ncols=1)

    yticks_fn = lambda index: "Sampler: %s" % (index, )

    detection_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/test/results/detection_accuracy")
    classification_plot = plotszoo.scalars.grouped.GroupedScalarsBarchart(data, group_keys, "summary/test/results/classification_accuracy")

    detection_df = detection_plot.plot(axes[0], title="Test Detection Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)
    classification_df = classification_plot.plot(axes[1], title="Test Classification Accuracy", nbins=20, grid=True, yticks_fn=yticks_fn)

    for ax in axes: ax.set_xlim(xmin=0.5)

    fig.set_size_inches(20, 10)
    fig.tight_layout()

    classification_df.to_excel(os.path.join(args.output_directory, "result/classification.xlsx"))
    detection_df.to_excel(os.path.join(args.output_directory, "result/detection.xlsx"))

    print(classification_df)
    print(detection_df)

    plotszoo.utils.savefig(fig, os.path.join(args.output_directory, "results.png"))


def plot_tables():
    query = {"$or": [{"config.wandb_tag": {"$eq": tag}} for tag in args.tags_best_replicas]}
    data = plotszoo.data.WandbData(args.username, args.project, query, verbose=args.verbose)
    data.pull_scalars(force_update=args.update_scalars)

    group_keys = ["config/sampler"]
    classes = ["neoplastic", "aphthous", "traumatic"]
    metrics = ["precision", "recall", "f1-score"]

    grouped_df = data.scalars.groupby(group_keys).agg(np.mean)
    for group in grouped_df.index:
        data_df = grouped_df.loc[group]
        table = np.zeros((len(classes), len(metrics)))
        for i, c in enumerate(classes):
            for j, m in enumerate(metrics):
                table[i, j] = data_df["summary/test/report/%s/%s" % (c, m)]*100
        
        table_df = pd.DataFrame(table, columns=metrics, index=classes)
        table_df.to_csv(os.path.join(args.output_directory, "%s_table.csv" % (group)))
        print("Sampler: %s" % (group))
        print(table_df)
        print()


parser = argparse.ArgumentParser()

parser.add_argument("--output-directory", type=str, default="./plots")
parser.add_argument("--username", type=str, default="mlpi")
parser.add_argument("--project", type=str, default="oral-ai")
parser.add_argument("--tags-study-replicas", type=str, default=["study-3"], nargs="+")
parser.add_argument("--tags-optimization", type=dict, default={
    "hp-optimization-none-trainingsampler-5": dict(
        data_augmentation="none",
        sampler="TrainingSampler"
    ),
    "hp-optimization-none-repeatfactortrainingsampler-5": dict(
        data_augmentation="none",
        sampler="RepeatFactorTrainingSampler"
    )
}, nargs="+")
parser.add_argument("--tags-best-replicas", type=str, default=["best-replicas-7"], nargs="+")
parser.add_argument("--update-scalars", action="store_true")
parser.add_argument("--update-series", action="store_true")
parser.add_argument("--verbose", action="store_true")


args = parser.parse_args()

plot_study()
plot_optimization()
plot_replicas()
#plot_tables()