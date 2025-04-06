import numpy as np

class LogisticRegressor:
    # Initiate the model
    def __init__(self):
        """
        Hyperparameters of learning rate
        and the number of iterations
        """
        self.lr = 0.05
        self.iter = 1000
        pass

    # Helper Functions
    def sigmoid(self, x_i):
        """
        Sigmoid function of logistic regression
        """
        z = np.clip(x_i @ self.params, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def update_params(self):
        """
        Using SGD to iterate thorugh dataset 1000 times
        and using each data point to update the parameters
        in each iteration
        """
        for _ in range(self.iter):
            for i in range(len(self.x_train)):
                x_i = self.x_train[i]
                y_i = self.y_train[i]
                self.params = self.params - (self.lr * (self.sigmoid(x_i) - y_i) * x_i)

    def predict(self, x_test):
        """
        Using sigmoid function to model probability, if 
        p >= 0.5 then it is 1, else it is 0
        """
        intercept = np.ones((len(x_test), 1))
        x_test = np.hstack((intercept, x_test))
        print(x_test.shape)
        p_x = self.sigmoid(x_test)
        return (p_x >= 0.5).astype(int)

    # Main Functions
    def fit(self, x_train, y_train):
        """
        x_train: m x n numpy array of data
        y_train: n x 1 numpy array of the target variables
        """
        # Store the training data 
        intercept = np.ones((x_train.shape[0], 1))
        self.x_train = np.hstack((intercept, x_train))
        self.y_train = y_train
        self.params = np.zeros(len(x_train[0]) + 1) # +1 here for intercept

        # Update the weights and biases
        self.update_params()
