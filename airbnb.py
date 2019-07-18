# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 06:23:53 2019

@author: somesh
"""

import numpy as np
import pandas as pd

#from math import sin, cos, sqrt, atan2

from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

from sklearn.metrics import r2_score,mean_squared_error

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

file_listing = pd.read_csv('listings_summary.csv')
listing = pd.DataFrame(file_listing)
file2 = pd.read_csv('listings.csv')
list2 = pd.DataFrame(file2)
#print(listing)

#print(listing['room_type'])


print(listing.shape)

print(listing.isnull().sum())


print('*********************************')
print('Missing data in each column')
print('neighbourhood:' + str(listing['neighbourhood'].isnull().sum()))
print('property_type:'+ str(listing['property_type'].isnull().sum()))
print('room_type:'+ str(listing['room_type'].isnull().sum()))
print('accommodates:' + str(listing['accommodates'].isnull().sum()))
print('bathrooms:' + str(listing['bathrooms'].isnull().sum()))
print('bedrooms:' + str(listing['bedrooms'].isnull().sum()))
print('no. of reviews:' + str(listing['number_of_reviews'].isnull().sum()))
print('security_deposit:' + str(listing['security_deposit'].isnull().sum()))
print('security_deposit:' + str(listing['security_deposit'].isnull().sum()))
print('cleaning_fee:' + str(listing['cleaning_fee'].isnull().sum()))

print('*********************************')
print('After ffill')

listing['neighbourhood'] = listing['neighbourhood'].ffill(inplace=False)
listing['accommodates'] = listing['accommodates'].ffill(inplace=False)
listing['bathrooms'] = listing['bathrooms'].ffill(inplace=False)
listing['bedrooms'] = listing['bedrooms'].ffill(inplace=False)
listing['security_deposit'] = listing['security_deposit'].ffill(inplace=False)
listing['cleaning_fee'] = listing['cleaning_fee'].ffill(inplace=False)

print(listing['neighbourhood'].isnull().sum())
print(listing['bathrooms'].isnull().sum())
print(listing['bedrooms'].isnull().sum())
print('security_deposit:' + str(listing['security_deposit'].isnull().sum()))
print('cleaning_fee:' + str(listing['cleaning_fee'].isnull().sum()))


print(listing['number_of_reviews'].isnull().sum())

#print(listing['neighbourhood'])

print('Label Encoding')
le = LabelEncoder()
listing['neighbourhood'] = le.fit_transform(listing['neighbourhood'])
listing['property_type']= le.fit_transform(listing['property_type'])
listing['room_type'] = le.fit_transform(listing['room_type'])
listing['bed_type'] = le.fit_transform(listing['bed_type'])
listing['has_availability']=le.fit_transform(listing['has_availability'])
listing['instant_bookable']= le.fit_transform(listing['instant_bookable'])

#print(listing['bed_type'])


#listing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7), 
#       c="price", cmap="gist_heat_r", colorbar=True)



#listing['cleaning_fee'] = listing['cleaning_fee'].apply(lambda x:x.strip('$'))
#listing[''] = listing['security_deposit'].apply(lambda x:x.strip(','))
#print('removed $ from cleaning fee')
#listing['security_deposit'] = listing['security_deposit'].apply(lambda x:x.strip('$'))
#listing['security_deposit'] = listing['security_deposit'].apply(lambda x:x.strip(','))



Y= list2['price']#.apply(lambda x:x.strip('$')) 

from geopy.distance import great_circle
def distance_from_berlin(lat, lon):
    berlin_centre = (52.5027778, 13.404166666666667)
    record = (lat, lon)
    return great_circle(berlin_centre, record).km

#add distanse dataset
    
list2.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(12,8), 
          c="price", cmap="CMRmap", colorbar=True, sharex=False);
plt.close()

listing['distance'] = listing.apply(lambda x: distance_from_berlin(x.latitude, x.longitude), axis=1)
#print(listing['distance'])

X = listing[['neighbourhood','accommodates','distance','property_type','room_type','bathrooms',
             'bedrooms','bed_type','number_of_reviews','has_availability','instant_bookable']]
#print(X)
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=21,test_size=0.35)

#print(X_train) # = X_train.reshape(-1,1)
#print(Y_train)

print('Using Linear Model:-')
lr = linear_model.LinearRegression()
lr.fit(X_train,Y_train)
lr_pred = lr.predict(X_test)
r2_lr = r2_score(Y_test,lr_pred)
print('r2 = ' + str(r2_lr))

plt.scatter(X_test['distance'],Y_test,marker='^')
plt.scatter(X_test['distance'], lr_pred,marker='o')
plt.ylim(0,600)
plt.xlabel('distance(km)',fontsize=18)
plt.ylabel('Price($)',fontsize=18)
plt.savefig('Linear.jpg')
plt.show()
plt.close()

print('Using Random Forest:-')
rf = RandomForestRegressor(random_state=1,n_estimators=10)
rf.fit(X_train,Y_train)
rf_pred = rf.predict(X_test)
r2_rf = r2_score(Y_test,rf_pred)
#rmse_rf = np.sqrt(mean_squared_error(Y_test,rf_pred))
print('r2 = ' + str(r2_rf))
#print(rmse_rf)

plt.scatter(X_test['distance'],Y_test,marker='^')
plt.scatter(X_test['distance'],rf_pred,marker='o')
plt.ylim(0,600)
plt.xlabel('distance(km)',fontsize=18)
plt.ylabel('Price($)',fontsize=18)
plt.savefig('RF.jpg')
plt.show()
plt.close()

#kernelridge = KernelRidge(alpha=0.2)
#kernelridge.fit(X_train,Y_train)
#kernel_ridge_pred = kernelridge.predict(X_test)
#r2_kernel_ridge = r2_score(Y_test,kernel_ridge_pred)
#rmse = np.sqrt(mean_squared_error(Y_test,decision_tree_pred))
#print(r2_kernel_ridge)
#print(rmse)

#plt.scatter(X_test['distance'],Y_test,marker='^')
#plt.scatter(X_test['distance'], kernel_ridge_pred,marker='o')
#plt.xlabel('distance',fontsize=18)
#plt.ylabel('Price',fontsize=18)
#plt.show()
#plt.close()

#booster = xgb.XGBRegressor()
#from sklearn.model_selection import GridSearchCV

# create Grid
#param_grid = {'n_estimators': [100, 150, 200],
#              'learning_rate': [0.01, 0.05, 0.1], 
#              'max_depth': [3, 4, 5, 6, 7],
#              'colsample_bytree': [0.6, 0.7, 1],
#              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
#booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
#booster_grid_search.fit(X_train,Y_train)

# print best estimator parameters found during the grid search
#print(booster_grid_search.best_params_)

#booster = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, 
#                           max_depth=6, n_estimators=200, random_state=4)

# train
#booster.fit(X_train, Y_train)

# predict
#Y_pred_train = booster.predict(X_train)
#Y_pred_test = booster.predict(X_test)
