from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


class FOREST(RandomForestClassifier):
    def __init__(self):
        RandomForestClassifier.__init__(self)

    def plotit(self, f, xs, ts):
        ys = [f(x.reshape(1, -1)) for x in xs]
        ys = np.array(ys).reshape(-1)
        correlation = np.corrcoef(ts, ys)
        print("Correlation: ", correlation)
        weeks = np.array([i for i in range(len(xs))])
        plt.plot(weeks, ys, 'b')
