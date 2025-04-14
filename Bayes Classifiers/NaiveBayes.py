import numpy as np
from scipy.stats import norm

class NaiveBayesClassifier:
    def __init__(self):
        """
        Initialize a dictionary representing
        the arrays of class-specific means for each class.
        The keys are the classes and the values are the
        class-specific-mean arrays.
        """
        self.mu_arr = {}
        self.pi = {}
        self.sigma = {}
        pass

    def get_class_specific_means(self):
        """
        Iterates through each class, and gets the 
        class-specific-mean array for each class and add
        it to the corresponding class in the dictionary
        """
        for k in self.classes:
            class_specific_arr = np.zeros(len(self.x[0]))
            num_k = 0
            for i in range(len(self.x)):
                if self.y[i] == k:
                    num_k += 1
                    class_specific_arr += self.x[i]
            class_specific_arr /= num_k
            self.mu_arr[k] = class_specific_arr
            
    def get_sigma(self):
        for k in self.classes:
                x_k = self.x[self.y == k]
                self.sigma[k] = np.std(x_k, axis = 0)

    def get_pi(self):
        """
        Iterate through each class and get
        the proportion of observations belonging
        to that class and add it to the dictionary
        """
        for k in self.classes:
            self.pi[k] = np.sum(self.y == k) / len(self.y)

    def fit(self, x_train, y_train):
        """
        Takes in the training data and stores it
        in the attributes
        """
        self.x = x_train
        self.y = y_train
        self.classes = np.unique(y_train)

        # Get the class-specific-means and the 
        # shared covariance
        self.get_class_specific_means()
        self.get_pi()
        self.get_sigma()

    def predict(self, x):
        """
        Takes in the x_test and performs computes the
        linear discriminant for each class, given this
        training data"""

        prediction = np.empty(len(x))

        for i in range(len(x)):
            posterior_dict = {}
            for k in self.classes:
                likelihood_k = np.prod(norm.pdf(x[i], loc = self.mu_arr[k], scale = self.sigma[k]))
                posterior_dict[k] = self.pi[k] * likelihood_k
            prediction[i] = max(posterior_dict, key=posterior_dict.get)

        return prediction
    
    def score(self, y_hat, y_test):
        """
        Calculates the proportion of elements in the 
        y_test vector that are correctly classified.
        """
        matching_elements = np.sum(y_hat == y_test)
        proportion = matching_elements / len(y_hat)
        return proportion