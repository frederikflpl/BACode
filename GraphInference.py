from sklearn.semi_supervised import label_propagation
import numpy as np

def inferGraph(labeled_data, unlabeled_data):
    features = [l[0] for l in labeled_data+unlabeled_data]
    labels = [l[1] for l in labeled_data] + (-np.ones(len(unlabeled_data))).tolist()
    label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
    print(str(len(features)) + " " + str(len(labels)))
    label_spread.fit(features, labels)