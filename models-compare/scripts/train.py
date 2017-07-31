"""Train an rbfln model

Usage:
  train.py -m <n> -n <n>
  train.py -h | --help

Options:
  -m <n>        Number of neurons in the hidden layer.
  -n <n>        Number of iterations.
  -h --help     Show this screen.
"""
from docopt import docopt
from RBFLN.rbfln import RBFLN
import numpy as np
import matplotlib.pyplot as plt
from scipy import random


def plotit(f, xs, ts):
    ys = [f(x) for x in xs]
    ys_classes = [round(y) for y in ys]
    for i in range(len(ys_classes)):
        if ys_classes[i] < 0:
            ys_classes[i] = 0

    correlation = np.corrcoef(ts, ys_classes)
    print("Correlation: ", correlation)
    weeks = np.array([i for i in range(len(xs))])
    plt.plot(weeks, ys_classes, 'b')


def to_classes(xs, start, stop, nclasses=10):
    """Return a Classes array"""
    classified = xs
    intervals = np.linspace(start, stop, nclasses)
    for i in range(len(xs)):
        j = 0
        while not (xs[i] >= intervals[j] and xs[i] <= intervals[j + 1]):
            j += 1
            if j == len(intervals) - 1:
                break
        classified[i] = j
    assert len(xs) == len(classified)
    return classified


def smoothing(xs):
    n = len(xs)
    for i in range(1, n - 1):
        xs[i] = sum([xs[j] for j in [i - 1, i, i + 1]]) / 3
    xs[0] = xs[1]
    xs[n - 1] = xs[n - 2]

    return xs


if __name__ == '__main__':
    opts = docopt(__doc__)

    random.seed(1)
    # Load the data
    qlen = 229
    # Select the columns correspondly to the data that we use to predict
    data_columns = (2, 4, 6, 10)
    xs = np.loadtxt('to_predict.csv', delimiter=',',
                    usecols=data_columns, dtype=float)

    # Extracting abundancy data
    ts = np.loadtxt('to_predict.csv', delimiter=',',
                    usecols=(1), dtype=float)

    # Normalization
    for i in range(0, qlen):
        xs[i][2] /= 300

    for i in range(1, qlen):
        xs[i][3] /= 50

    ts = ts / np.amax(ts)
    ts = smoothing(ts)

    ts = to_classes(ts, 0, 0.3, 100)
    # Number of neurons in the input layer
    N = len(data_columns)

    # Read the number of neurons in the hidden layer
    M = int(opts['-m'])
    # Read the number of iterations
    niter = int(opts['-n'])

    xs_to_train = xs[101:qlen]
    ts_to_train = ts[101:qlen]
    xs_to_validate = xs[50:100]
    ts_to_validate = ts[50:100]
    model = RBFLN(xs_to_train, ts_to_train,
                  xs_to_validate, ts_to_validate,
                  M, N, niter, variance=0.005)
    plotit(model.predict, xs, ts)
    weeks = np.array([i for i in range(229)])
    plt.plot(weeks, ts, 'r')
    plt.show()
