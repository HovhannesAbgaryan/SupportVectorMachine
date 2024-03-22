import numpy as np
import cvxopt  # library for Convex Optimization

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


class SupportVectorMachine:
    # region Summary
    """
    Hard (C = 0) and Soft (C > 0) Margin Support Vector Machine Classifier with Kernels
    """
    # endregion Summary

    # region Constructor

    def __init__(self, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
        # region Summary
        """
        Constructor of SupportVectorMachine class.
        :param C:
        :param kernel_name: implement for 'linear', 'poly' and 'rbf'
        :param power: degree of the polynomial kernel (d in the slides)
        :param gamma: Kernel coefficient for "rbf" and "poly"
        :param coef: coefficient of the polynomial kernel (r in the slides)
        """
        # endregion Summary

        self.C = C
        self.kernel_name = kernel_name
        self.power = power
        self.gamma = gamma
        self.coef = coef

        self.kernel = None
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.t = None

    # endregion Constructor

    # region Functions

    def get_kernel(self, kernel_name):
        # region Summary
        """
        Define 3 kernel functions under this method and then use a dictionary with keys being the names of the kernels
        and the values being the kernel functions with respective parameters.
        :param kernel_name: Name of kernel.
        :return: Kernel function.
        """
        # endregion Summary

        # region Nested Functions

        def linear(x1, x2):
            return np.dot(x1, x2)

        def polynomial(x1, x2):
            return (self.coef + self.gamma * np.dot(x1, x2)) ** self.power

        def rbf(x1, x2):
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        # endregion Nested Functions

        # region Body

        kernel_functions = dict(linear = linear, poly = polynomial, rbf = rbf)
        return kernel_functions[kernel_name]

        # endregion Body

    def fit(self, X, y):
        # Transform the data into array to speed up the kernel matrix construction
        X = np.array(X)
        y = np.array(y)

        nr_samples, nr_features = np.shape(X)

        # Setting a default value for gamma
        if not self.gamma:
            self.gamma = 1 / nr_features

        # Set the kernel function
        self.kernel = self.get_kernel(self.kernel_name)

        # Construct the kernel matrix
        kernel_matrix = np.zeros((nr_samples, nr_samples))
        for i in range(nr_samples):
            for j in range(nr_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(nr_samples) * -1)
        A = cvxopt.matrix(y, (1, nr_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(nr_samples) * -1)
            h = cvxopt.matrix(np.zeros(nr_samples))
        else:
            G_max = np.identity(nr_samples) * -1
            G_min = np.identity(nr_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(nr_samples))
            h_min = cvxopt.matrix(np.ones(nr_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
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

        # Calculate intercept (t) with 1st support vector
        self.t = -self.support_vector_labels[0]
        for i in range(len(self.alphas)):
            self.t += self.alphas[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        for instance in np.array(X):
            prediction = 0
            # Determine the label of the given instance by the support vectors
            for i in range(len(self.alphas)):
                prediction += self.alphas[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], instance)
            prediction -= self.t
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    # endregion Functions
