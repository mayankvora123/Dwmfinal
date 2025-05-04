# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get user input for X and Y
# x_input = input("Enter values for X separated by spaces: ")
# y_input = input("Enter values for Y separated by spaces: ")
# # Convert input strings to numpy arrays of floats or integers
# X = np.array(list(map(float, x_input.strip().split())))
# Y = np.array(list(map(float, y_input.strip().split())))

data = pd.read_csv('/Users/aryaangala/Downloads/Linear Regression.csv')
X = data['X'].values
Y = data['Y'].values

print(X) # 3 8 9 13 3 6 11 21 1 16
print(Y) # 30 57 64 72 36 43 59 90 20 83

# Step 1: Calculate means
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Step 2: Calculate the slope (m) and intercept (b)
numerator = 0
denominator = 0

for i in range(len(X)):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2

beta = numerator / denominator
alpha = mean_y - beta * mean_x

print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")

# Step 3: Predict values
def predict(x):
    return alpha + beta * x

print(f"Y = {alpha:.4f} +  {beta:.4f} X")

X_input = int(input("Enter the X variable for which Y is to be predicted: "))
# X_input = 10

Y_output = predict(X_input)
print(f"Predicted Y value for X = {X_input} is {Y_output:.4f}")

Y_pred = predict(X)

# Step 4: Plotting the results
plt.scatter(X, Y, color='blue', label='Actual points')
plt.plot(X, Y_pred, color='red', label='Best Fit Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()

