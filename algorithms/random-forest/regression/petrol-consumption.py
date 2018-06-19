import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
''' Reference: http://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/ '''

# Headers
FEATURE_HEADERS = ['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence(%)']

TARGET_HEADERS = ['Petrol_Consumption']

# Reading Data
dataset = pd.read_csv('../../../datasets/petrol_consumption.csv')

features = dataset[FEATURE_HEADERS]
print('Features\n', features.head(5))

target = dataset[TARGET_HEADERS]
print('Targets\n', target.head(5))

# Feature Scaling
from sklearn.preprocessing import StandardScaler

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 1/3, random_state = 0)
sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)

#
print("feature_train Shape :: ", feature_train.shape)
print("feature_test Shape :: ", feature_test.shape)
print("target_train Shape :: ", target_train.shape)
print("target_test Shape :: ", target_test.shape)

# Create random forest classifier instance
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(feature_train, target_train.values.ravel())
print("Trained model :: ", regressor)
predictions_of_target = regressor.predict(feature_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(target_test, predictions_of_target))
print('Mean Squared Error:', metrics.mean_squared_error(target_test, predictions_of_target))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_test, predictions_of_target)))
