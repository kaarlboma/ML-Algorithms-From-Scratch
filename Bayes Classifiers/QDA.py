import numpy as np

class QDAClassifier:
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
            self.sigma[k] = np.cov(x_k.T, bias = True) 

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

        self.prediction = np.empty(len(x))

        for i in range(len(x)):
            discriminant_dict = {}
            for k in self.classes:
                mu_k = self.mu_arr[k]
                sigma_k = self.sigma[k]
                pi_k = self.pi[k]
                term1 = (-0.5 * (x[i] - mu_k).T) @ (np.linalg.inv(sigma_k)) @ (x[i] - mu_k)
                term2 = -0.5 * np.linalg.slogdet(sigma_k)[1]
                term3 = np.log(pi_k)
                discriminant_dict[k] = (term1 + term2 + term3).item()
            self.prediction[i] = max(discriminant_dict, key = discriminant_dict.get)

        print(self.prediction)

        self.prediction
    
    def score(self, y_test):
        """
        Calculates the proportion of elements in the 
        y_test vector that are correctly classified.
        """
        matching_elements = np.sum(self.prediction == y_test)
        proportion = matching_elements / len(self.prediction)
        print(proportion)
        proportion