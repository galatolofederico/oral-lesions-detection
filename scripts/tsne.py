import argparse
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()

parser.add_argument("--features", type=str, required=True)

args = parser.parse_args()

cache = pickle.load(open(args.features, "rb"))

features = cache["features"]
classes = cache["classes"]
categories = cache["categories"]


scaler = StandardScaler()
features = scaler.fit_transform(features)
print("[!] Fitting PCA")
pca = PCA(n_components=50)
features = pca.fit_transform(features)
print("[!] Fitting T-SNE")
tsne = TSNE(n_components=2)
features = tsne.fit_transform(features)


for c, name in categories.items():
    mask = classes == c
    xy = features[mask]
    plt.scatter(xy[:, 0], xy[:, 1], label=name)

plt.legend()
plt.show()
