import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Scale All columns
dataset = pd.read_csv('../datasets/boston-housing.csv', delimiter=' * \s*', engine='python' )
features = dataset.iloc[:, :].values
print(features.shape)

print(features)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(features[:, :])
print(X_train)

#Scale specific columns
dataset = pd.read_csv('../datasets/boston-housing.csv', delimiter=' * \s*', engine='python' )
FEATURE_HEADERS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
features = dataset[FEATURE_HEADERS]
print(features[:][1:5])
le = preprocessing.LabelEncoder()
features['LSTAT_cat']  = le.fit_transform(features.LSTAT)
features['PTRATIO_cat']  = le.fit_transform(features.PTRATIO)
#drop the old categorical columns from dataframe
dummy_fields = ['LSTAT', 'PTRATIO']
features = features.drop(dummy_fields, axis = 1)
print(features[:][1:5])
