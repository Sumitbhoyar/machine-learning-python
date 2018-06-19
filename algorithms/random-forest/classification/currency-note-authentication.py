import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

''' Reference: http://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/'''

# Headers
FEATURE_HEADERS = ['Variance', 'Skewness', 'Curtosis', 'Entropy']

TARGET_HEADERS = ['Class']

# Reading Data
dataset = pd.read_csv('../../../datasets/bill_authentication.csv')

features = dataset[FEATURE_HEADERS]
print('Features\n', features.head(5))

target = dataset[TARGET_HEADERS]
print('Targets\n', target.head(5))

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
trained_model = RandomForestClassifier(n_estimators=20)
trained_model.fit(feature_train, target_train.values.ravel())
print("Trained model :: ", trained_model)
predictions = trained_model.predict(feature_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(" Confusion matrix ",confusion_matrix(target_test, predictions))
print(" Classification Report ",classification_report(target_test, predictions))
print(" Accuracy score ",accuracy_score(target_test, predictions))
