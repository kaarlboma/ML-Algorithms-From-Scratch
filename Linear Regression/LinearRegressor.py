import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def get_prediction(self, i):
        return np.dot(self.weights, self.X_train[i]) + self.bias
    
    def get_MSE(self):
        sum = 0
        n = len(self.X_train)

        for i in range(len(self.X_train)):
            # Specify the terms for gradient descent
            y_hat = self.get_prediction(i)
            y_i = self.y_train[i]

            sum += (y_i - y_hat) ** 2

        return sum / n

    def get_weight_gradient(self, feature):
        sum = 0
        for i in range(len(self.X_train)):
            sum += (self.y_train[i] - self.get_prediction(i)) * self.X_train[i][feature]
        return (-2 / len(self.X_train)) * sum
    
    def get_bias_gradient(self):
        sum = 0
        for i in range(len(self.X_train)):
            sum += (self.y_train[i] - self.get_prediction(i))
        return (-2 / len(self.X_train)) * sum
    
    def update_weights(self):
        # Update weights
        MSE = float('inf')
        tolerance = 1e-6
        for iteration in range(10000):
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * self.get_weight_gradient(i)
            
            self.bias -= self.learning_rate * self.get_bias_gradient()

            new_MSE = self.get_MSE()

            if np.abs(MSE - new_MSE) < tolerance:
                print(f"Converged after {iteration} iterations")
                break
            MSE = new_MSE

    def fit(self, X_train, y_train):
        # Store data, weights and bias attributes
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.zeros(X_train[0].size)
        self.bias = 0
        self.learning_rate = 0.01

        self.update_weights()

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (np.dot(X, self.weights)) + self.bias

def main():
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate 100 x points
    x = np.linspace(0, 10, 100)

    # Define a true underlying linear relationship
    # Let's use y = 2x + 1 as our true line, but add some noise
    true_slope = 2
    true_intercept = 1

    # Generate y values with some random noise
    # The noise will make the points not perfectly linear
    noise = np.random.normal(0, 1.5, 100)
    y = true_slope * x + true_intercept + noise

    model = LinearRegression()
    model.fit(x, y)
    model.predict(np.array([4]))
    print(model.bias)
    print(model.weights)

if __name__ == "__main__":
    main()



