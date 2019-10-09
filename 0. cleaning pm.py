#%% 0. Housekeeping 

# =============================================================================
# 0.1 Import packages
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# 0.2 Import data
# =============================================================================

train = pd.read_csv('data/train.csv', low_memory=False)
store = pd.read_csv('data/store.csv', low_memory=False)

#%% 1. Merging store to train data

# =============================================================================
# 1.1 Since data has to has the same size as 'train', a container is created
# =============================================================================

# Creating dataframe
expanded_store = train.loc[:,['Date','Store']]

# =============================================================================
# 1.2 Merging variables which do not have to be changed
# =============================================================================

# Variables which can be merged right away
easy = store.loc[:,['Store','StoreType','Assortment','CompetitionDistance']]

# Variables which can be simply merged
expanded_store = pd.merge(expanded_store, easy, on=['Store'])

# =============================================================================
# 1.3 Creating a dummy variable since when competition is open for every store
# =============================================================================

# Dataset of the variables which have to be transformed
competition = store.loc[:,['Store',
                           'CompetitionOpenSinceMonth','CompetitionOpenSinceYear']]

# For easier looping 
competition.set_index('Store', inplace=True)

# Finding stores which have competition openend
open_comp = competition.any(axis=1)

# Stores with competition
open_stores = competition.loc[open_comp].index.unique()

# Generating Date when opened
for store_id in open_stores: 
    year = competition.loc[store_id,'CompetitionOpenSinceYear'].astype(int)
    month = competition.loc[store_id,'CompetitionOpenSinceMonth'].astype(int)
    competition.loc[store_id,'CompetitionDate'] = datetime(year,month,1)

# Creating a dummy variable for whether competition openend for each store
expanded_store.loc[:,'CompetitionOpened'] = 0

for store_id in open_stores: 
    
    # Getting the date when competition openend
    date = competition.loc[store_id,'CompetitionDate']
    date_str= date.strftime("%Y-%m-%d")
    
    store_number = expanded_store.loc[:,'Store'] == store_id
    
    maximum_date = expanded_store.loc[store_number,'Date'].max()
    
    # Whether it falls in time frame
    competition_existing = (expanded_store.loc[:,'Date'].between(date_str,maximum_date)) &  (expanded_store.loc[:,'Store'] == store_id)
    
    # Indicating whether competition is around
    expanded_store.loc[competition_existing, 'CompetitionOpened'] = 1


#%% Join
    

