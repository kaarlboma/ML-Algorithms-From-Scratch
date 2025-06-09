import numpy as np

class LassoRegressor:
    def __init__(self):
        """
        Initializes the model
        """
        pass

    def soft_threshold(self, z):
        """
        Computes soft threshold
        """
        return np.sign(z) * max(abs(z) - self.lam, 0)

    def beta_update(self, j):
        """
        Computes the gradient for B_j and updates its 
        coefficient estimate
        """
        n = self.n
        
        if j == 0:
            self.beta[0] = np.mean(self.y - self.x[:, 1:] @ self.beta[1:])
        else:
            partial_residual = self.y - np.dot(self.x, self.beta) + np.dot(self.x[:,j], self.beta[j])
            z_j = (1/n) * np.dot(self.x[:,j].T, partial_residual)
            self.beta[j] = self.soft_threshold(z_j) / (np.sum(self.x[:,j] ** 2) / n)


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