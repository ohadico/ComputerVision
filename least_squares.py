import numpy as np
import matplotlib.pyplot as plt


PLOT = False


def main():
    points = np.array([(1, 3.96), (4, 27.96), (3, 15.15), (5, 45.8), (2, 7.07), (6, 69.4)])
    if PLOT:
        plt.scatter(points[:, 0], points[:, 1], label="points")

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
