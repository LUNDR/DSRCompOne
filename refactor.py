import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import seaborn as sns


train = pd.read_csv('~/data/train.csv', low_memory=False)
store = pd.read_csv('~/data/store.csv', low_memory=False)


#merge two data sets on store number

train_all=pd.merge(train,store,on="Store",how='left')
    

#%% 1.define function to Split data by date to create a train and CV set
#for this function dates must be formatted as Dates and called "Date"

def date_split(split_percent,df):
    date_range_days=(df['Date'].max() - df['Date'].min()).days
    split_date=df['Date'].min() + dt.timedelta(date_range_days*split_percent) 
    df_early,df_later = df.loc[df['Date'] <= split_date], df.loc[df['Date'] > split_date]
    return df_early, df_later

## create binary variable in dataset for whether store faces competition or not today
def has_competition(df,col1,col2):
    
    # Finding stores which have competition openend
    open_comp = df[[col1,col2]].any(axis=1)

    # Stores with competition
    open_ = df.loc[open_comp].index.unique()
    df['has_competition']=0
    df.loc[open_,'has_competition']=1
    
## create function for variable of competition open since

def how_long_competition(df,col1,col2):
  
    # Finding stores which have competition openend
    open_comp = df[[col1,col2]].any(axis=1)
    open_ = df.loc[open_comp].index.unique()
    
    df['how_long_comp']=0
    for i in open_:
        date=df.loc[i,'CompetitionOpenSinceYear']+df.loc[i,'CompetitionOpenSinceMonth']
        df.loc[i,'comp_since'] =df[i,'Date']-pd.to_datetime(date,'%Y%m') 
        
        
    





train['Date']=pd.to_datetime(train['Date'])

##create train and CV sets
df_train, df_CV = date_split_time(0.8,train)