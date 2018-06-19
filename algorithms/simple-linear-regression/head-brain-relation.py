import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading Data
data = pd.read_csv('../../datasets/headbrain.csv')

# Collecting X and Y
X = data.iloc[:, [2]].values
Y = data.iloc[:, [3]].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg.fit(x_train, y_train)
# Y Prediction
Y_pred = reg.predict(x_train)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_train, Y_pred))
r2 = reg.score(X, Y)

print("RMSE")
print(rmse)
print("R2 Score")
print(r2)

# Plotting Values and Regression Line
plt.figure()

plt.title("Head Vs Brain size")
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.gcf().text(0, 0, "RMSE: " + str(rmse), fontsize=10)

plt.plot(x_train,y_train, 'g.', label='Train data')
plt.plot(x_test, y_test, 'b.', label='Test data')
plt.plot(x_train, reg.predict(x_train), color='red', label='Regression Line')

plt.legend()
plt.grid(True)
plt.show()