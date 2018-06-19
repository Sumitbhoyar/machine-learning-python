import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('../../datasets/student-reading-writing-marks.csv')
print(data.shape)
data.head()

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='#ef1234')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# X and Y Values
X = np.array([write, read]).T
Y = np.array(math)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print(rmse)
print(r2)

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X,Y)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
plt.scatter(X[:,1],Y)
plt.xlabel("Read & Write")
plt.ylabel("Maths")
# plt.title("Relationship between Open and Close Stock")
# plt.show()

#predicting stock price
print(lm.predict(X)[0:5])
plt.scatter(X[:,1],lm.predict(X))
plt.plot(X[:,1],lm.predict(X), 'k.') #color code is k or m or etc.,
plt.xlabel("Read & Write")
plt.ylabel("Predicted Maths")
plt.title("Read & Write vs Maths")
plt.show()
#Mean Sqaured Error
MSE=np.mean((Y-lm.predict(X))**2)
print("Mean Sqaured Error %r" %(MSE))
#Sum of Sqaured Error
SSE=np.sum((Y-lm.predict(X))**2)
print("SUM of Sqaured Error %r" %(SSE))
