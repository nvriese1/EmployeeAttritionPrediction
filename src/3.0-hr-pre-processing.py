#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


# load in the pre-cleaned HR dataset
df = pd.read_csv('HR_data_cleaned_EDA.csv')

# examine the data
df.head()


# In[3]:


# detect any remaining null values
print(df.isnull().sum())


# In[4]:


# clean null values, replacing with median (categorical)
nan_columns = ['EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance','SatisfactionRatio','FlightRatio']

for col in nan_columns:
    df[col] = df[col].fillna(df[col].median())
    
# drop rows with fewer than 20 null values
df.dropna(how='any', axis='rows', inplace=True)

# ensure null values have been filled
print(df.isnull().sum())


# In[5]:


# drop redundant columns
df.drop(columns=['Unnamed: 0'], inplace=True)

# select ordinal categorical features to be encoded rather than scaled
ordinal = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'PerformanceRating']

# change dtype to categorical for ordinal features
df[ordinal] = df[ordinal].astype('object')


# In[6]:


# get numerical columns and assign to a dataframe
numerical = df.select_dtypes(include=['float64', 'int64'])

# get categorical columns and assign to a DataFrame for encoding
categorical  = df.select_dtypes(include=['object'])

# Drop the target variable 'Attrition' from the DataFrame to be encoded
categorical.drop(columns='Attrition', inplace=True)

# manually map 'Attrition' to a binary encoding
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# get categorical dummies and assign to a DataFrame for scaling
df_dummies = pd.get_dummies(categorical)
df_dummies = pd.concat([numerical, df_dummies], axis=1)

# ensure all dummy datatypes are numerical
print(df_dummies.dtypes)


# In[7]:


# assign feature array and label array
X = df_dummies.values
y = df['Attrition'].values

indices = np.arange(len(y))

# split X and y into testing and training datasets, stratify the split 
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.3, random_state=42, stratify=y)

# make a StandardScaler object 
scaler = StandardScaler() 

# fit numerical test and train data to the scaler object independently
X_train_scaled = scaler.fit_transform(X_train[:,0:24])
X_test_scaled = scaler.transform(X_test[:,0:24]) 

# concatenate the scaled numerical data to the encoded categorical data for both test and train sets
X_train = np.concatenate([X_train_scaled, X_train[:,24:]], axis=1)
X_test = np.concatenate([X_test_scaled, X_test[:,24:]], axis=1)

# examine the results
print(pd.DataFrame(X_train))


# In[8]:


# export train & test arrays for later use
pd.DataFrame(X_train).to_csv("X_train.csv")
pd.DataFrame(X_test ).to_csv("X_test.csv" )
pd.DataFrame(y_train).to_csv("y_train.csv")
pd.DataFrame(y_test ).to_csv("y_test.csv" )
pd.DataFrame(idx_train).to_csv("idx_train.csv")
pd.DataFrame(idx_test).to_csv( "idx_test.csv" )

# export feature labels
pd.DataFrame(df_dummies.columns).to_csv("features.csv" )

