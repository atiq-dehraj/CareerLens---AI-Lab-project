import numpy as np
import pandas as pd

# Tree ka Node structure
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Leaf node ki prediction

# Gini Impurity calculate karo
def calculate_gini(y):
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Best split dhoondho
def get_best_split(X, y):
    n_samples, n_features = X.shape

    if n_samples < 2:
        return None, None

    classes, encoded_y = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    parent_gini = calculate_gini(y)
    best_gini = parent_gini
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        order = np.argsort(X[:, feature_idx], kind="mergesort")
        values = X[order, feature_idx]
        labels = encoded_y[order]

        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = np.bincount(
            labels, minlength=n_classes
        ).astype(np.int64)

        for i in range(n_samples - 1):
            class_idx = labels[i]
            left_counts[class_idx] += 1
            right_counts[class_idx] -= 1

            if values[i] == values[i + 1]:
                continue

            n_left = i + 1
            n_right = n_samples - n_left

            gini_left = 1 - np.sum(
                (left_counts / n_left) ** 2
            )
            gini_right = 1 - np.sum(
                (right_counts / n_right) ** 2
            )

            weighted_gini = (
                n_left * gini_left +
                n_right * gini_right
            ) / n_samples

            if weighted_gini < best_gini - 1e-12:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = values[i]

    return best_feature, best_threshold

def majority_class(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

# Recursive Tree Build
def build_tree(X, y, depth=0, max_depth=3):
    # Stopping criteria: Agar sab same class hain ya depth limit cross ho gayi
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return TreeNode(value=majority_class(y))

    best_feature, best_threshold = get_best_split(X, y)
    
    if best_feature is None:
        return TreeNode(value=majority_class(y))

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] > best_threshold
    
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    
    return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

# Prediction function
def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)
