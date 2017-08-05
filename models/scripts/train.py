"""Train a model

Usage:
  train.py [-k <model>] -m <order> -n <order> [-c <order>] -o <file>
  train.py -h | --help

Options:
  -k <model>    Model to use [default: rbfln]:
                    rbfln: Radial Basis Function Linear Network
                    MLPC: Multi-layer Perceptron
                    rndm: Random Forest Classifier
                    svmc: Support Vector Machine Classifier
                    svmr: Support Vector Machine Reggression
                    lineal: Linear Reggression
  -m <n>        Number of neurons in the hidden layer.
  -n <n>        Number of iterations.
  -c <n>        Number of classes to train [default: 6]
  -o <file>     Output filename
  -h --help     Show this screen.
"""
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pickle
from models.rbfln import RBFLN
from models.mlpc import MLPC
from models.forest import FOREST
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from scipy import random


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


models = {
    'rbfln': RBFLN,
    'mlpc': MLPC,
    'rndm': FOREST,
    'svmc': SVC,
    'svmr': SVR,
    'lineal': LinearRegression,
}

if __name__ == '__main__':
    opts = docopt(__doc__)
    random.seed(1)

    # Load the data
    qlen = 305
    # Select the columns correspondly to the data that we use to predict
    # data_columns = (2, 3, 4, 5, 6, 7, 8)
    data_columns = (2, 3, 4, 5)
    xs = np.loadtxt('to_predict.csv', delimiter=',',
                    usecols=data_columns, dtype=float)
    # Extracting abundancy data
    ts = np.loadtxt('to_predict.csv', delimiter=',',
                    usecols=(1), dtype=float)

    # Normalization
    # LST day
    for i in range(0, qlen):
        xs[i][2] /= 300
    # LST night
    for i in range(0, qlen):
        xs[i][3] /= 300

    # Lluvia
    for i in range(1, qlen):
        xs[i][4] /= 100

    ts = ts / np.amax(ts)
    ts = smoothing(ts)

    # Read the number of classes
    nclasses = int(opts['-c'])

    ts = to_classes(ts, 0, 0.3, nclasses)
    # Number of neurons in the input layer
    N = len(data_columns)

    # Read the number of neurons in the hidden layer
    M = int(opts['-m'])
    # Read the number of iterations
    niter = int(opts['-n'])
    # Read the model selected
    model = models[opts['-k']]

    if opts['-k'] == 'mlpc':
        xs_to_train = xs[qlen - 100:qlen]
        ts_to_train = ts[qlen - 100:qlen]

        model = MLPC(solver='lbfgs', alpha=1e-5, max_iter=niter,
                     hidden_layer_sizes=(50, 50),
                     validation_fraction=0.1,
                     random_state=2)
        model.fit(xs_to_train, ts_to_train)
        model.plot(xs, ts)

    elif opts['-k'] == 'rndm':
        pass
    elif opts['-k'] == 'svm':
        pass
    elif opts['-k'] == 'lineal':
        pass
    # Default rbfln
    else:
        xs_to_train = xs[76:qlen]
        ts_to_train = ts[76:qlen]
        xs_to_validate = xs[50:75]  # 10% to validate
        ts_to_validate = ts[50:75]  # 10% to validate
        model = RBFLN(xs_to_train, ts_to_train,
                      xs_to_validate, ts_to_validate,
                      M, N, niter, variance=0.0005)

        model.plot(xs, ts, (0, 49))

    print('Saving...')
#    filename = opts['-o']
#    filename = 'Models/' + filename
#    f = open(filename, 'wb')
#    pickle.dump(model, f)
#    f.close()


    weeks = np.array([i for i in range(qlen)])
    plt.plot(weeks, ts, 'r')
    plt.show()
