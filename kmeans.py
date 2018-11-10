from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

NUM_OF_POINTS = 8
OFFSET = 1
RANGE = 12
NUM_OF_CLUSTERS = 3
PLOT = True


def generate_random_points(num_of_points, offset, range_):
    points = (offset + np.random.rand(num_of_points, 2) * range_).astype(np.int)
    return points


def dist(a, b=None):
    if b is None:
        a, b = a
    return np.sqrt(np.sum((b-a) ** 2))


def get_closest_center(point, centers):
    return min(zip([point] * len(centers), centers), key=dist)[1]


def get_kmeans_cluster_centers(points, num_of_clusters, offset, range_):
    init_centers = generate_random_points(num_of_clusters, offset, range_)
    clusters_centers = init_centers

    while True:
        clusters = defaultdict(list)
        for point in points:
            clusters[tuple(get_closest_center(point, clusters_centers))].append(point)
        new_clusters_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters.values()])
        if len(new_clusters_centers) < num_of_clusters:
            new_clusters_centers = np.array(new_clusters_centers.tolist() +
                                            generate_random_points(num_of_clusters - len(new_clusters_centers),
                                                                   offset, range_).tolist())
        if np.all(clusters_centers == new_clusters_centers):
            break
        clusters_centers = new_clusters_centers

    return clusters_centers, init_centers


def main():
    points = generate_random_points(NUM_OF_POINTS, OFFSET, RANGE)
    if PLOT:
        plt.scatter(points[:, 0], points[:, 1], label="points")

    my_cluster_centers, init_centers = get_kmeans_cluster_centers(points, NUM_OF_CLUSTERS, OFFSET, RANGE)
    if PLOT:
        plt.scatter(my_cluster_centers[:, 0], my_cluster_centers[:, 1], label="my kmeans")

    cluster_centers = get_sk_kmean(points, NUM_OF_CLUSTERS, init_centers)
    if PLOT:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], label="sk kmeans")

    if PLOT:
        plt.legend()
        plt.grid()
        plt.show()

    assert all(c in cluster_centers for c in my_cluster_centers)


def get_sk_kmean(points, num_of_clusters, init_centers=None):
    kmeans = KMeans(num_of_clusters) if init_centers is None else KMeans(num_of_clusters, init_centers, n_init=1)
    kmeans.fit(points)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    main()
