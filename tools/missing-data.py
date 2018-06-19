import numpy as np

from pandas import read_csv
from sklearn.preprocessing import Imputer

dataset = read_csv('../datasets/country-lifeexp-gdp.csv')

# Find zeros
# print('Zeros:\n', (dataset[list(range(0,dataset.shape[1]))] == 0).sum())

# mark mean values as missing or NaN
print('NaN:\n', dataset.isnull().sum())
'''
strategy:
You can replace the missing data with  the following values
1.) Mean
2.) Median
3.) Most_frequent/Mode

axis:
1.) axis =0  This means that the computer will take the mean per column
2.) axis =1  This means that the computer will take the mean per row
'''
imp=Imputer(missing_values=np.NaN, strategy="mean" )
dataset["year"]=imp.fit_transform(dataset[["year"]]).ravel()
dataset["lifeExp"]=imp.fit_transform(dataset[["lifeExp"]]).ravel()

print('NaN:\n', dataset.isnull().sum())
print('Shape:\n', dataset.shape)

# mark zero values as missing or NaN
dataset = read_csv('../datasets/country-lifeexp-gdp.csv')
print((dataset == 0).sum())
dataset = dataset.replace(0, np.NaN)
print((dataset == 0).sum())

# delete record when NaN
dataset = read_csv('../datasets/country-lifeexp-gdp.csv')
print('Shape:\n', dataset.shape)
print('NaN:\n', dataset.isnull().sum())
dataset.dropna(subset = ['year', 'lifeExp'], inplace = True)
print('Shape:\n', dataset.shape)
print('NaN:\n', dataset.isnull().sum())


