import argparse
import numpy as np
import json
import pickle

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA


def plot_pca_point(features_database, pca_model, fig_h, fig_w, fig_dpi, output, point):
    features_database = json.load(open(features_database, "r"))
    pca = pickle.load(open(pca_model, "rb"))

    features = []
    classes = []
    for name, feature_list in features_database.items():
        for feature in feature_list:
            features.append(feature["features"])
            classes.append(feature["type"])

    features = np.array(features)
    classes = np.array(classes)

    features = pca.transform(features)
    point = pca.transform(np.atleast_2d(point))

    plt.figure(figsize=(fig_h/fig_dpi, fig_w/fig_dpi), dpi=fig_dpi)
    cmap = ListedColormap(["r","b","g"])
    scatter = plt.scatter(features[:, 0], features[:, 1], c=classes, cmap=cmap, s=10)

    plt.scatter(point[:, 0], point[:, 1], marker="x", s=200, c="k")

    plt.legend(handles=scatter.legend_elements()[0], labels=["neoplastic", "aphthous", "traumatic"])

    if output == "":
        plt.show()
    else:
        plt.savefig(output, dpi=fig_dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-database", type=str, required=True)
    parser.add_argument("--pca-model", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--fig-h", type=int, default=1080)
    parser.add_argument("--fig-w", type=int, default=720)
    parser.add_argument("--fig-dpi", type=int, default=100)

    parser.add_argument("--point", type=str, required=True)

    args = parser.parse_args()

    point = json.loads(args.point)

    dict_args = vars(args)
    del dict_args["point"]
    plot_pca_point(**dict_args, point=point)