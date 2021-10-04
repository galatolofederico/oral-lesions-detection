import argparse
import numpy as np
import json
import pickle
from scipy import spatial

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_histogram_dist(features_database, fig_h, fig_w, fig_dpi, output, point, distance="cosine"):
    features_database = json.load(open(features_database, "r"))
    dist_fn = getattr(spatial.distance, distance)
    class_names = ["neoplastic", "aphthous", "traumatic"]
    cmap = ListedColormap(["r","b","g"])

    dists = dict()
    for name, feature_list in features_database.items():
        for feature in feature_list:
            if feature["type"] not in dists:
                dists[feature["type"]] = []
            
            dists[feature["type"]].append(dist_fn(point, feature["features"]))

        
    fig, axes = plt.subplots(len(dists))

    for k, ax in zip(dists.keys(), axes):
        dist = dists[k]
        ax.set_title(class_names[k])
        ax.set_xlim(0, 1)
        n, bins, patches = ax.hist(dist, "auto", density=True, color=cmap(k))
    
    fig.tight_layout(pad=3.0)

    if output == "":
        plt.show()
    else:
        plt.savefig(output, dpi=fig_dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-database", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--fig-h", type=int, default=1080)
    parser.add_argument("--fig-w", type=int, default=720)
    parser.add_argument("--fig-dpi", type=int, default=100)
    parser.add_argument("--distance", type=str, default="cosine")


    parser.add_argument("--point", type=str, required=True)

    args = parser.parse_args()

    point = json.loads(args.point)

    dict_args = vars(args)
    del dict_args["point"]
    plot_histogram_dist(**dict_args, point=point)