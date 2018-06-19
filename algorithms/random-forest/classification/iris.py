import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
''' Reference: http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/'''

# Headers
FEATURE_HEADERS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

TARGET_HEADERS = ['Species']

# Reading Data
dataset = pd.read_csv('iris.csv')

features = dataset[FEATURE_HEADERS]
print('Features\n', features.head(5))

target = dataset[TARGET_HEADERS]
print('Targets\n', target.head(5))

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 1/3, random_state = 0)
#
print("feature_train Shape :: ", feature_train.shape)
print("feature_test Shape :: ", feature_test.shape)
print("target_train Shape :: ", target_train.shape)
print("target_test Shape :: ", target_test.shape)

# Create random forest classifier instance
trained_model = RandomForestClassifier()
print(feature_train.dtypes)
trained_model.fit(feature_train, target_train.values.ravel())
print("Trained model :: ", trained_model)
predictions = trained_model.predict(feature_test)

print("Train Accuracy :: ", accuracy_score(target_train, trained_model.predict(feature_train)))
print("Test Accuracy  :: ", accuracy_score(target_test, predictions))
print(" Confusion matrix ", confusion_matrix(target_test, predictions))
