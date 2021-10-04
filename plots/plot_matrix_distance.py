import json
import argparse
from scipy import spatial
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--rows", type=str, required=True)
parser.add_argument("--cols", type=str, required=True)
parser.add_argument("--distance", type=str, default="cosine")
parser.add_argument("--output", type=str, default="")

args = parser.parse_args()

rows_features = json.load(open(args.rows, "r"))
cols_features = json.load(open(args.cols, "r"))
dist_fn = getattr(spatial.distance, args.distance)


rows_features_rois = []
cols_features_rois = []

for row_feature in rows_features.values():
    for roi_feature in row_feature:
        rows_features_rois.append(roi_feature)

for col_feature in cols_features.values():
    for roi_feature in col_feature:
        cols_features_rois.append(roi_feature)


rows_features_rois = sorted(rows_features_rois, key=lambda e: e["type"])
cols_features_rois = sorted(cols_features_rois, key=lambda e: e["type"])


matrix = np.zeros((len(rows_features_rois), len(cols_features_rois)))
for i, row in tqdm(enumerate(rows_features_rois), total=len(rows_features_rois)):
    for j, col in enumerate(cols_features_rois):
        matrix[i, j] = dist_fn(row["features"], col["features"])

fig, ax = plt.subplots()

ax.set_xlabel(args.rows)
ax.set_ylabel(args.cols)

pos = ax.imshow(matrix)
fig.colorbar(pos, ax=ax)

if args.output == "":
    plt.show()
else:
    plt.savefig(args.output)


