

######
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import timedelta
import datetime
#from bayes_opt import BayesianOptimization



cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Sales')) #Remove sales from list
df = df[cols + ['Sales']] #Create new dataframe with sales right at the end 
X, y = df.iloc[:, :-1],df.iloc[:, -1]
df=df[['av_SalesPerCustomer','av_SalesPerCustomer_dayofweek','av_SalesPerCustomer_dayofmonth','Customers','Promo','Promo2','CompetitionDistance','Sales','dayofyear','dayofweek','Decay','comp_open_since']]

date_range_days=(df.index.max() - df.index.min()).days
split_date=df.index.min() + timedelta(date_range_days*0.8) #train set 80% of full population
#randomly creating train and test subsets. may need to refine this 
df_early,df_later = df.loc[df.index <= split_date], df.loc[df.index > split_date]
#create feature matrix of everything up to sales, create labels from sales 
X_train, X_test, y_train, y_test = df_early.iloc[:,:-1], df_later.iloc[:,:-1], df_early.iloc[:,-1], df_later.iloc[:,-1] 



# creating XGB optimised data structure. we will need this for our cross validation model later
df_DM = xgb.DMatrix(data=X, label=y)

#here we decide the parameters that we are going to use in the model
params = {"objective":"reg:squarederror", #type of regressor, shouldnt change
          'colsample_bytree': 0.9, #percentage of features used per tree. High value can lead to overfitting.
          'learning_rate': 0.1, #step size shrinkage used to prevent overfitting. Range is [0,1]
          'max_depth': 3, #determines how deeply each tree is allowed to grow during any boosting round. keep this low! this will blow up our variance if high
          'lambda': 4, #L1 regularization on leaf weights. A large value leads to more regularization. Could consider l2 euclidiean regularisation
          'n_estimators': 500, #number of trees you want to build.
          'n_jobs': 4,#should optimise core usage on pc
         'subsample':0.9} 

#now we must instantiate the XGB regressor by calling XGB regressor CLASS from the XGBoost library, we must give it the hypter parameters as arguments
xg_reg = xgb.XGBRegressor(**params)

#Fit the regressor to the training set and make predictions for the test set using .fit() and .predict() methods
xg_reg.fit(X_train, y_train)
test_preds = xg_reg.predict(X_test)
train_preds = xg_reg.predict(X_train)
print("RMSE train: %f" % np.sqrt(mean_squared_error(y_train, train_preds)))
print("RMSE test: %f" % np.sqrt(mean_squared_error(y_test, test_preds)))

