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
            residual = self.y - np.dot(self.x, self.beta)
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
        intercept = np.ones((x.shape[0], 1))
        self.x = np.concatenate((intercept, x), axis = 1)
        self.y = y
        self.n = self.x.shape[0]
        self.p = self.x.shape[1]
        self.lam = lam
        self.beta = np.zeros(self.p)
        self.alpha = 0.01
        self.iter = 10000

        for _ in range(self.iter):
            for j in range(len(self.beta)):
                self.beta_update(j)
    
    def predict(self, x):
        """
        Computes the dot product of the betas with the 
        testing data
        """

        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis = 1)
        return np.dot(x, self.beta)