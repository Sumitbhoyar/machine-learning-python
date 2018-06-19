import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/country-lifeexp-gdp.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print('X', X.shape)
print('y', y.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)