import numpy as np

class LDAClassifier:
    def __init__(self):
        """
        Initialize a dictionary representing
        the arrays of class-specific means for each class.
        The keys are the classes and the values are the
        class-specific-mean arrays.
        """
        self.mu_arr = {}
        self.pi = {}
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
            
    def sigma(self):
        return np.cov(self.x.T)

    def get_pi(self):
        """
        Iterate through each class and get
        the proportion of observations belonging
        to that class and add it to the dictionary
        """
        for k in self.classes:
            self.classes[k] = np.sum(self.y[self.y == k]) / len(self.y)

    def fit(self, x_train, y_train):
        """
        Takes in the training data and stores it
        in the attributes
        """
        self.x = x_train
        self.y = y_train
        self.classes = y_train.unique()

        # Get the class-specific-means and the 
        # shared covariance
        self.get_class_specific_means()
        self.get_pi()
        self.sigma = self.sigma()

    def predict(self, x):
        """
        Takes in the x_test and performs computes the
        linear discriminant for each class, given this
        training data"""

        discriminant_dict = {}
        for k in self.classes:
            discriminant_dict[k] = (x.T @ np.linalg.inv(self.sigma) @ self.mu_arr[k]) - (0.5 * self.mu_arr[k].T @ np.linalg.inv(self.sigma) @ self.mu_arr[k]) + np.log(self.pi[k])

        prediction = max(discriminant_dict, key = discriminant_dict.get)
