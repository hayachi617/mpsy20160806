import numpy as np


def ma(history, n):
    return np.array([0, ] * (n - 1) + [np.average(history[i - n: i]) for i in range(n, len(history) + 1)])
