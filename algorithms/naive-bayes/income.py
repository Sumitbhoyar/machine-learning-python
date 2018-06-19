''' Reference: http://dataaspirant.com/2017/02/20/gaussian-naive-bayes-classifier-implementation-python/'''
# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix

# Headers
FEATURE_HEADERS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country']
SCALERS = ['age', 'fnlwgt', 'education_num','capital_gain', 'capital_loss', 'hours_per_week']
TARGET_HEADERS = ['income']
ALL_HEADERS = FEATURE_HEADERS + TARGET_HEADERS
HEADERS_WITH_QUESTION_MARK = ['workclass', 'occupation', 'native_country']
# Reading Data
dataset = pd.read_csv('../../datasets/income-range.data',
                      header=None, delimiter=' *, *', engine='python')

dataset.columns = ALL_HEADERS
#whether there is any null value in our dataset or not
print('IsNull' , dataset.isnull().sum())

for value in HEADERS_WITH_QUESTION_MARK:
    dataset[value].replace(['?'], [dataset.describe(include='all')[value][2]],
                                inplace=True)

features = dataset[FEATURE_HEADERS]
print('Features\n', features.head(5))

target = dataset[TARGET_HEADERS]
print('Targets\n', target.head(5))

le = preprocessing.LabelEncoder()
features['workclass_cat']  = le.fit_transform(features.workclass)
features['education_cat']  = le.fit_transform(features.education)
features['marital_status_cat']  = le.fit_transform(features.marital_status)
features['occupation_cat']  = le.fit_transform(features.occupation)
features['relationship_cat']  = le.fit_transform(features.relationship)
features['race_cat']  = le.fit_transform(features.race)
features['sex_cat']  = le.fit_transform(features.sex)
features['native_country_cat'] = le.fit_transform(features.native_country)
#drop the old categorical columns from dataframe
dummy_fields = ['workclass', 'education', 'marital_status',
                  'occupation', 'relationship', 'race',
                  'sex', 'native_country']
features = features.drop(dummy_fields, axis = 1)

feature_train, feature_test, target_train, target_test = train_test_split(features, target.values.ravel(), test_size = 1/3, random_state = 0)

#
print("feature_train Shape :: ", feature_train.shape)
print("feature_test Shape :: ", feature_test.shape)
print("target_train Shape :: ", target_train.shape)
print("target_test Shape :: ", target_test.shape)

# Create GaussianNB classifier instance
trained_model = GaussianNB()
trained_model.fit(feature_train, target_train)
predictions = trained_model.predict(feature_test)

print("Train Accuracy :: ", accuracy_score(target_train, trained_model.predict(feature_train)))
print("Test Accuracy  :: ", accuracy_score(target_test, predictions))
print(" Confusion matrix ", confusion_matrix(target_test, predictions))
