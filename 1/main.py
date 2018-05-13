import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import count, takewhile

A = 6

LEFT = 0
RIGHT = 1


def func1(x):
    return math.sqrt(1 + A * math.log10(x + 1))


FUNC2_ACTUAL = math.pi / 4
def func2(x):
    return 1 / (1 + x * x)


def func3(x):
    return math.log(math.sin(x)) if math.sin(x) > 0 else 0


def _get_n(h):
    return int(round((RIGHT - LEFT) / h))


def _get_step(n):
    return (RIGHT - LEFT) / n


def _get_linspace(h):
    return np.linspace(LEFT, RIGHT, _get_n(h) + 1)


def middle_rect_formula(func, h):
    linspace = _get_linspace(h)
    return h * sum(func(x + h / 2) for x in linspace[:-1])


def trapeze_formula(func, h):
    linspace = _get_linspace(h)
    return h * sum([
        func(LEFT) / 2,
        func(RIGHT) / 2,
        sum(map(func, linspace[1:-1]))
    ])


def simpson_formula(func, h):
    linspace = _get_linspace(h)
    return h / 3 * sum([
        func(LEFT),
        func(RIGHT),
        2 * sum(map(func, linspace[2:-2:2])),
        4 * sum(map(func, linspace[1:-1:2]))
    ])


def compound_formulas(func, h):
    formulas = [
        (middle_rect_formula, 2),
        (trapeze_formula, 2),
        (simpson_formula, 4)
    ]
    for formula, m in formulas:
        sum_h1 = formula(func, h)
        sum_h2 = formula(func, h / 2)
        r = abs(sum_h2 - sum_h1) / (2 ** m - 1)

        print(
            "func: {}, sum_h1: {:.10f}, sum_h2: {:.10f}, r: {:.10f}"
            .format(formula.__name__, sum_h1, sum_h2, r)
        )


def gaussian_quadrature(func):
    xs_ = [0.033765, 0.169395, 0.380690, 0.619310, 0.830605, 0.966235]
    as_ = [0.085662, 0.180381, 0.233957, 0.233957, 0.180381, 0.085662]
    res = sum([a * func(x) for a, x in zip(as_, xs_)])
    print("Gaussian quadrature: {:.10f}".format(res))


def first_part():
    compound_formulas(func1, .1)
    gaussian_quadrature(func1)


def _get_func2_deltas(formula, epsilon=.00001, step=1):
    vals = (formula(func2, _get_step(n)) for n in count(2, step))
    diffs = (abs(val - FUNC2_ACTUAL) for val in vals)
    some = takewhile(lambda diff: diff > epsilon, diffs)
    enumerated = [(2 + step * i, diff) for i, diff in enumerate(some)]
    return np.array(enumerated)


def second_part():
    middle_rect_vals = _get_func2_deltas(middle_rect_formula)
    trapeze_vals = _get_func2_deltas(trapeze_formula)
    simpson_vals = _get_func2_deltas(simpson_formula, step=2, epsilon=.000001)

    plt.plot(*middle_rect_vals.T, color='red', marker='.', label='middle_rect_formula')
    plt.plot(*trapeze_vals.T, color='green', marker='.', label='trapeze_formula')
    plt.plot(*simpson_vals.T, color='blue', marker='.', label='simpson_formula')

    plt.legend()
    plt.show()


def third_part():
    print(math.sqrt(math.pi / (4 * 0.01)))

    compound_formulas(func3, .0001)
    gaussian_quadrature(func3)


if __name__ == '__main__':
    first_part()
    second_part()
    third_part()
