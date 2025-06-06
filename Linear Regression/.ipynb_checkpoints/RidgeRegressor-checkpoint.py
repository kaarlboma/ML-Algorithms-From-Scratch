import numpy as np

class RidgeRegressor:
    def __init__(self):
        """
        Initializes the model
        """
        pass

    def beta_update(self, j):
        """
        Computes the gradient for B_j and updates its 
        coefficient estimate
        """
        if j == 0:
            self.beta[0] = np.mean(self.y)
        else:
            residual = self.y - np.dot(self.beta, self.x)
            dj_dBj = (-2 / self.n) * np.dot(residual, self.x[:, j]) + (2 * self.lam * self.beta[j])
            self.beta[j] = self.beta[j] - self.alpha * dj_dBj


    def fit(self, x, y, lam):
        """
        Stores the attributes in the class

        x: training data features
        y: training data labels
        n: number of observations
        lam: tuning parameter
        beta: array of predictor coefficients
        """
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.p = x.shape[1]
        self.lam = lam
        self.beta = np.zeros(self.n)
        self.alpha = 0.01

        for j in range(len(self.beta)):
            self.beta_update(j)
    