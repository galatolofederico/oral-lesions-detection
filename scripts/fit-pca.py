import argparse
import numpy as np
import pickle
import json

from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()

parser.add_argument("--features-database", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

features_database = json.load(open(args.features_database, "r"))

features = []
classes = []
for name, feature_list in features_database.items():
    for feature in feature_list:
        features.append(feature["features"])
        classes.append(feature["type"])

features = np.array(features)
classes = np.array(classes)

decomposition = PCA(n_components=2)
decomposition.fit(features)

pickle.dump(decomposition, open(args.output,"wb"))
