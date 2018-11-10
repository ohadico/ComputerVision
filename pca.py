import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt

PLOT_BIASED = True
PLOT_UNBIASED = False
ERROR_TOLERANCE = 1e-15


def centralize(points, plot=False):
    mean = np.array(points).mean(axis=0)
    centralized_points = points - mean
    if plot:
        plt.scatter(centralized_points[:, 0], centralized_points[:, 1])
    return centralized_points, mean


def get_covariance(points,  is_centralize=False):
    if not is_centralize:
        points, _ = centralize(points)
    covariance = np.matmul(points.T, points)  # matrix multiplication
    return covariance


def get_eigen(covariance, plot=False):
    eig_val, eig_mat = np.linalg.eig(covariance)
    if plot:
        for i in range(len(eig_val)):
            plt.quiver(0, 0, eig_mat[i, 0], eig_mat[i, 1], scale=eig_val[i])
    return eig_mat, eig_val


def rotate(centralized_points, eig_mat, plot=False):
    rotated_points = np.matmul(centralized_points, eig_mat)
    if plot:
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1])
    return rotated_points


def my_pca(points, plot=False):
    centralized_points, mean = centralize(points, plot)

    covariance = get_covariance(centralized_points)

    eig_mat, eig_val = get_eigen(covariance, plot)

    rotated_points = rotate(centralized_points, eig_mat, plot)

    return rotated_points, mean, eig_mat, eig_val


def one_dim_reduction(points, eig_val, plot=False):
    reduced_points = points
    reduced_points[:, np.argmin(eig_val)] = 0.0
    if plot:
        plt.scatter(reduced_points[:, 0], reduced_points[:, 1])
    return reduced_points


def inverse(points, eig_mat, plot=False):
    restore_points = np.matmul(points, eig_mat)
    if plot:
        plt.scatter(restore_points[:, 0], restore_points[:, 1])
    return restore_points


def main():
    points = np.array([(2.5, 2.9), (0.5, 1.2), (2.2, 3.4), (1.9, 2.7), (3.1, 3.5), (2.3, 3.2),
                       (2, 2.1), (1, 1.6), (1.5, 2.1), (1.1, 1.4)])

    if PLOT_BIASED:
        plt.scatter(points[:, 0], points[:, 1])

    rotated_points, mean, eig_mat, eig_val = my_pca(points, plot=PLOT_UNBIASED)

    reduced_points = one_dim_reduction(rotated_points, eig_val, plot=PLOT_UNBIASED)

    restore_points = inverse(reduced_points, eig_mat.T, plot=PLOT_UNBIASED)

    restore_points += mean

    if PLOT_BIASED:
        plt.scatter(restore_points[:, 0], restore_points[:, 1])

    if PLOT_BIASED or PLOT_UNBIASED:
        plt.grid()
        plt.show()

    new_points = sk_pca_dim_reduction(points)

    assert np.all(np.abs(restore_points - new_points) < ERROR_TOLERANCE)


def sk_pca_dim_reduction(points):
    pca = sklearn.decomposition.PCA()
    pca.fit(points)
    transformed_points = pca.transform(points)
    transformed_points[:, 1] = 0
    new_points = pca.inverse_transform(transformed_points)
    return new_points


if __name__ == '__main__':
    main()
