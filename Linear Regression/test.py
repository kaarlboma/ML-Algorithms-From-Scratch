import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 100 x points
x = np.linspace(0, 10, 100).reshape(-1, 1)

# Define a true underlying linear relationship
true_slope = 2
true_intercept = 1

# Generate y values with some random noise
noise = np.random.normal(0, 1.5, 100).reshape(-1, 1)  # Ensure correct shape
y = true_slope * x + true_intercept + noise

# Flatten y to ensure it's 1D
y = y.ravel()


# Split the data

# Fit model
model = LinearRegression()
model.fit(x, y)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
