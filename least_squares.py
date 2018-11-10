import numpy as np
import matplotlib.pyplot as plt

PLOT = True
ERROR_TOLERANCE = 1e-10


def get_X_Y(points):
    x = points[:, 0]
    x2 = x ** 2
    ones = np.ones(x.shape)
    X = np.column_stack((x2, x, ones))
    Y = points[:, 1]
    return X, Y


def solve_for_B(X, Y):
    return np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))


def least_square_parabola(points, plot=False):
    X, Y = get_X_Y(points)
    B = solve_for_B(X, Y)

    if plot:
        x_plot = np.linspace(0, 10, 100)
        f = np.poly1d(B)
        plt.plot(x_plot, f(x_plot), label="my least squares")

    return X, B, Y


def main():
    points = np.array([(1, 3.96), (4, 27.96), (3, 15.15), (5, 45.8), (2, 7.07), (6, 69.4)])
    if PLOT:
        plt.scatter(points[:, 0], points[:, 1], label="points")

    X, B, Y = least_square_parabola(points, plot=PLOT)

    coefficients = get_numpy_polyfit_coef(points, 2)
    print get_poly_str(coefficients, precision=2)

    if PLOT:
        x_plot = np.linspace(0, 10, 100)
        f = np.poly1d(coefficients)
        plt.plot(x_plot, f(x_plot), label="np.polyfit")

    if PLOT:
        plt.grid()
        plt.legend()
        plt.show()

    assert np.all(np.abs(B - coefficients) < ERROR_TOLERANCE)


def get_poly_str(coefficients, precision):
    poly_str = ''
    term = ":.{precision}f".format(precision=precision)
    term = "{}{" + term + "}{}"
    for p, c in enumerate(coefficients):
        prefix = '+' if c >= 0 and p > 0 else ''
        suffix = ''
        p = len(coefficients) - p - 1
        if p > 0:
            suffix = 'x'
            if p > 1:
                suffix += '**{}'.format(p)
        poly_str += term.format(prefix, c, suffix)
    return poly_str


def get_numpy_polyfit_coef(points, degree):
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    return coefficients


if __name__ == '__main__':
    main()
