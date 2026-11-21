import numpy as np
import pandas as pd


def run_kmeans(df, column="salary", k=3, max_iters=100):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")

    if not isinstance(max_iters, (int, np.integer)) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer.")

    # Remove missing, infinite, zero, and negative salaries.
    values = pd.to_numeric(df[column], errors="coerce")
    valid_mask = np.isfinite(values) & (values > 0)

    result_df = df.loc[valid_mask].copy()
    result_df[column] = values.loc[valid_mask]

    if result_df.empty:
        raise ValueError("No valid salary values available for clustering.")

    data = result_df[column].to_numpy(dtype=float).reshape(-1, 1)

    # K cannot exceed the number of distinct values.
    unique_values = np.unique(data[:, 0])
    k = min(k, len(unique_values))

    # Reproducible initialization using distinct salary values.
    rng = np.random.default_rng(42)
    centroids = rng.choice(unique_values, size=k, replace=False)
    centroids = centroids.reshape(-1, 1).astype(float)

    labels = np.full(len(data), -1, dtype=int)

    for _ in range(max_iters):
        # Manual Euclidean distances for one-dimensional data.
        distances = np.abs(data - centroids.T)

        # Assign each salary to its nearest centroid.
        new_labels = np.argmin(distances, axis=1)

        # Recalculate centroids from assigned cluster members.
        new_centroids = centroids.copy()

        for j in range(k):
            cluster_points = data[new_labels == j]

            if len(cluster_points) > 0:
                new_centroids[j, 0] = np.mean(cluster_points)
            else:
                # Reinitialize empty clusters to a distant data point.
                nearest_distance = np.min(distances, axis=1)
                farthest_index = np.argmax(nearest_distance)
                new_centroids[j, 0] = data[farthest_index, 0]

        converged = (
            np.array_equal(labels, new_labels)
            and np.allclose(centroids, new_centroids, rtol=0, atol=1e-6)
        )

        labels = new_labels
        centroids = new_centroids

        if converged:
            break

    # Ensure returned assignments match the final centroids.
    final_distances = np.abs(data - centroids.T)
    labels = np.argmin(final_distances, axis=1)

    result_df["cluster"] = labels.astype(int)

    return result_df, centroids
