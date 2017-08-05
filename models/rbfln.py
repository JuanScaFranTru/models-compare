import numpy as np
from numpy.linalg import norm
from math import exp
import matplotlib.pyplot as plt


class RBFLN(object):

    def __init__(self, xs, ts, xs_val, ts_val, M, N, niter=100,
                 eta_linear_weights=None,
                 eta_non_linear_weights=None,
                 variance=None):
        """Create a Radial Basis Functional Link Network.

        Create a RBFLN with N neurons in the input layer, M neurons in the
        hidden layer and 1 neuron in the output layer.
        The xs and ts parameters should have the same length.
        The lengths of all the elements of xs should be equal to n.
        The lengths of all the elements of ts should be equal to 1.

        :param xs: input feature vectors used for training.
        :param ts: associated output target vectors used for training.
        :param xs_val: input feature vectors used for validation.
        :param ts_val: associated output target vectors used for
                              validation.
        :param M: Number of neurons in the hidden layer.
        :param N: Number of neurons in the input layer.
        :param niter: Number of iterations.
        :param eta_linear_weights: Learning rate of linear weights.
        :param eta_non_linear_weights: Learning rate of non linear weights.
        :param variance: The initial variance of the RBF.

        :type xs: list of vector of float
        :type ts: list of float
        :type xs_val: list of vector of float
        :type ts_val: list of float
        :type M: int
        :type N: int
        :type niter: int
        :type eta_linear_weights: float
        :type eta_non_linear_weights: float
        :type variance: float

        """
        self.xs = np.array([np.array(x) for x in xs])
        self.ts = np.array(ts)
        self.xs_val = np.array([np.array(x) for x in xs_val])
        self.ts_val = np.array(ts_val)
        self.M = M
        self.N = N
        self.niter = niter
        self.eta_linear_weights = eta_linear_weights
        self.eta_non_linear_weights = eta_non_linear_weights
        self.variance = variance
        x2q = {tuple(x): q for q, x in enumerate(list(xs))}

        msg = 'The xs and ts parameters should have the same length'
        assert len(xs) == len(ts), msg

        # Initialize variables
        self._init_center_vectors()
        self._init_variances()
        self._init_weights()
        self._init_learning_rates()

        # Train the model using the training data
        for i in range(niter):
            # Calculate ys and zs
            ys = np.apply_along_axis(self._ys, 1, xs)

            def mapper(x):
                x = tuple(x)
                q = x2q[x]
                y = ys[q]
                return self._z(x, y)
            zs = np.apply_along_axis(mapper, 1, xs)

            # Update weights
            self.us = self._update_non_linear_weights(ys, zs)
            self.ws = self._update_linear_weights(ys, zs)

            error = self.total_sq_error(xs, ts)
            validation_error = self.total_sq_error(xs_val, ts_val)
            print("    {:2.4f}   {:2.4f} {} ".format(error,
                                                     validation_error, i))
        print("Fin del entrenamiento")
        print()

    def _sum_sq_error(self, x, t):
        """Partial sum squared errors of the given training input feature
        vectors and associated output target vectors.

        :param x: input feature vector.
        :param t: associated output target vector.
        :type x: vector of float
        :type t: float
        """
        return (t - self.predict(x)) ** 2

    def total_sq_error(self, xs, ts):
        """Sum of the partial sum squared errors.

        :param xs: input feature vectors.
        :param ts: associated output target vectors.

        :type xs: list of vector of float
        :type ts: list of float
        """
        msg = 'Input and output vectors should have the same length'
        assert len(xs) == len(ts), msg

        return sum([self._sum_sq_error(x, t) for x, t in zip(xs, ts)])

    def _ys(self, x):
        """Calculate the RBF output of every hidden neuron.

        :param x: input feature vector.
        :type x: vector of float

        :return: Output of the hidden layer.
        :rtype: vector of float
        """
        vs = self.vs
        variances = self.variances
        squared_norms = np.array([norm(x - v) ** 2 for v in vs])
        return np.array([exp(-sn / (2 * var))
                         for var, sn in zip(variances, squared_norms)])

    def _z(self, x, ys=None):
        """Calculate the output of the RBFLN model.

        :param x: input feature vector.
        :param ys: precalculated output of the hidden layer.
        :type x: vector of float
        :type ys: vector of float

        :return: output of the model.
        :rtype: float
        """
        M = self.M
        N = self.N
        us = self.us
        ws = self.ws

        if ys is None:
            ys = self._ys(x)

        linear_component = np.dot(x, ws)
        nonlinear_component = np.dot(ys, us)
        return (1 / (M + N)) * (linear_component + nonlinear_component)

    def predict(self, x):
        """Predict the output using the model in the given input vector.

        :param x: input feature vector.
        :type x: vector of float

        :return: output of the model.
        :rtype: float
        """
        return self._z(x)

    def _update_non_linear_weights(self, ys, zs):
        """Update weights via gradient descent."""
        eta1 = self.eta_non_linear_weights
        us = self.us
        ts = self.ts
        xs = self.xs
        M = self.M
        N = self.N
        Q = len(xs)

        assert len(ys) == len(zs) == len(ts) == Q

        new_us = us + eta1/(M + N) * \
            np.sum([(t - z) * y for t, z, y in zip(ts, zs, ys)], axis=0)

        return new_us

    def _update_linear_weights(self, ys, zs):
        eta2 = self.eta_linear_weights
        ws = self.ws
        ts = self.ts
        xs = self.xs
        M = self.M
        N = self.N
        Q = len(xs)

        assert len(ys) == len(zs) == len(ts) == Q

        new_ws = ws + eta2/(M + N) * \
            np.sum([(t - z) * x for t, z, x in zip(ts, zs, xs)], axis=0)

        return new_ws

    def _init_weights(self):
        """Init linear and non-linear weights.

        Select all weights randomly between -0.5 and 0.5."""
        MIN, MAX = -0.5, 0.5
        self.us = np.random.uniform(MIN, MAX, (self.M))
        self.ws = np.random.uniform(MIN, MAX, (self.N))

    def _init_variances(self):
        """Compute initial values for variances."""
        variance = self.variance
        N = self.N
        M = self.M
        if variance is None:
            variance = (1 / (2 * M)) ** (1/N)
        self.variances = np.array([variance] * M)

    def _init_center_vectors(self):
        """Init center vectors.

        Let Q be the number of input feature vectors.
        Initialize RBF center vectors by putting v(m) = x(m) if M <= Q, else
        put v(q) = x(q) , q = 1,...,Q, and draw the remaining M - Q centers at
        random in the feature space.
        """
        M = self.M
        N = self.N
        xs = self.xs
        Q = len(xs)

        vs = xs[:M]
        if M > Q:
            vs = np.concatenate((vs, np.random.uniform(0, 1, (M - Q, N))))

        self.vs = vs
        assert len(self.vs) == M

    def _init_learning_rates(self):
        """Init learning rates."""
        if self.eta_linear_weights is None:
            self.eta_linear_weights = 75

        if self.eta_non_linear_weights is None:
            self.eta_non_linear_weights = 90

    def plot(self, xs, ts, test_interval):
        ys = [self.predict(x) for x in xs]
        ys_classes = [round(y) for y in ys]

        for i in range(len(ys_classes)):
            if ys_classes[i] < 0:
                ys_classes[i] = 0
        correlation_all = np.corrcoef(ts, ys_classes)

        # start, end = test_interval
        # correlation_test = np.corrcoef(ts[start:end], ys_classes[start:end])

        print("Correlation with all data: ", correlation_all)
        # print("Correlation with test data: ", correlation_test)
        weeks = np.array([i for i in range(len(xs))])
        # test_weeks = weeks[start:end]
        plt.plot(weeks, ys_classes, 'b')
        #         test_weeks, ys_classes[start:end], 'g')
