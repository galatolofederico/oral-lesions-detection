import argparse
import numpy as np
import json

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import spatial

parser = argparse.ArgumentParser()

parser.add_argument("--features-database", type=str, required=True)
parser.add_argument("--decomposition", type=str, default="TSNE", choices=["TSNE", "PCA"])
parser.add_argument("--output", type=str, default="")
parser.add_argument("--fig-h", type=int, default=1080)
parser.add_argument("--fig-w", type=int, default=720)
parser.add_argument("--fig-dpi", type=int, default=100)
parser.add_argument("--distance", type=str, default="cosine")

parser.add_argument("--point", type=str, default="")

args = parser.parse_args()
point = None
if args.point != "":
    point = json.loads(args.point)


dist_fn = getattr(spatial.distance, args.distance)
features_database = json.load(open(args.features_database, "r"))

features = []
classes = []
for name, feature_list in features_database.items():
    for feature in feature_list:
        features.append(feature["features"])
        classes.append(feature["type"])

if point is not None:
    features.append(point)

features = np.array(features)
classes = np.array(classes)

if args.decomposition == "TSNE":
    decomposition = TSNE(n_components=2, metric=dist_fn)
elif args.decomposition == "PCA":
    decomposition = PCA(n_components=2)
transformed = decomposition.fit_transform(features)

if point is not None:
    transformed = transformed[:-1,:]
    transformed_point = transformed[-1,:]

plt.figure(figsize=(args.fig_h/args.fig_dpi, args.fig_w/args.fig_dpi), dpi=args.fig_dpi)
cmap = ListedColormap(["r","b","g"])
scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=classes, cmap=cmap, s=10)

if point is not None:
    plt.scatter(transformed_point[0], transformed_point[1], marker="x", s=200, c="k")

plt.legend(handles=scatter.legend_elements()[0], labels=["neoplastic", "aphthous", "traumatic"])

if args.output == "":
    plt.show()
else:
    plt.savefig(args.output, dpi=args.fig_dpi)
