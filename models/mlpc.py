import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


class MLPC(MLPClassifier):
    def __init__(self):

        super(MLPC, self).__init__(self)

    def plotit(self, xs, ts):
        ys = [self.predict(x.reshape(1, -1)) for x in xs]
        ys = np.array(ys).reshape(-1)
        correlation = np.corrcoef(ts, ys)
        print("Correlation: ", correlation)
        weeks = np.array([i for i in range(len(xs))])
        plt.plot(weeks, ys, 'b')
