import numpy as np
import cvxopt  # library for Convex Optimization

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


class SVM:
    # region Summary
    """
    Hard (C = None) and Soft (C > 0) Margin Support Vector Machine Binary Classifier
    """
    # endregion Summary

    # region Constructor

    def __init__(self, C=1):
        self.C = C
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.w = None
        self.t = None

    # endregion Constructor

    # region Functions

    def fit(self, X, y):
        # region Summary
        """

        :param X: NumPy array or Pandas DataFrame
        :param y: NumPy array or Pandas DataFrame with 1 and -1 encoding
        :return:
        """
        # endregion Summary

        # region Body

        nr_samples, nr_features = np.shape(X)

        # Define the Quadratic Optimization problem
        P = cvxopt.matrix(np.outer(y, y) * (X @ X.T), tc='d')
        q = cvxopt.matrix(np.ones(nr_samples) * -1)
        A = cvxopt.matrix(y, (1, nr_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:  # <=> C = None <=> Hard Margin SVM
            G = cvxopt.matrix(np.identity(nr_samples) * -1)
            h = cvxopt.matrix(np.zeros(nr_samples))
        else:  # <=> C > 0 <=> Soft Margin SVM
            G_max = np.identity(nr_samples) * -1
            G_min = np.identity(nr_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(nr_samples))
            h_min = cvxopt.matrix(np.ones(nr_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the Quadratic Optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange's multipliers (denoted by alphas in the lecture slides)
        alphas = np.ravel(minimization['x'])

        # First, get indexes of non-zero Lagrange's multipliers
        idx = alphas > 1e-7

        # Get the corresponding Lagrange's multipliers (non-zero alphas)
        self.alphas = alphas[idx]

        # Get the support vectors
        self.support_vectors = X[idx]

        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        self.w = self.alphas * self.support_vector_labels @ self.support_vectors

        # Calculate the intercept (denoted by t in lecture slides) with first support vector
        self.t = self.w @ self.support_vectors[0] - self.support_vector_labels[0]

        # endregion Body

    def predict(self, X):
        return np.sign(self.w @ X.T - self.t)

    # endregion Functions
