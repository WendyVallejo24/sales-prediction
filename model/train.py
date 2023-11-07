from os import PathLike
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
import pathlib
import numpy as np

df = pd.read_csv(pathlib.Path('data/train.csv'))
y = df.pop('Item_Outlet_Sales')
columns_to_drop = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
X = df.drop(columns_to_drop, axis=1)
print (X.head())
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imp.fit_transform(X)

imp_y = SimpleImputer(missing_values=np.nan, strategy='mean')
y = imp_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=44, shuffle =True)

print ('Training model.. ')
clf = RandomForestRegressor(n_estimators=100,
                            max_depth=7, 
                            random_state=33)
clf.fit(X_train, y_train)
print ('Saving model..')

print('Random Forest Regressor Train Score is : ' , clf.score(X_train, y_train))

dump(clf, pathlib.Path('model/train-v1.joblib'))