#!/usr/bin/env python
# coding: utf-8

# We can start by importing the packages we'll be using in this exploration.

# In[1]:


# Import relevant packages
import os
import pandas as pd


# Next, lets load in our dataset from Kaggle. It consists of 6 files, which we will import as DataFrames from their respective filetypes.

# In[2]:


# Load HR data sourced from Kaggle 
# Source: https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study

data_dictionary = pd.read_excel('data_dictionary.xlsx')
emp_survey = pd.read_csv('employee_survey_data.csv')
man_survey = pd.read_csv('manager_survey_data.csv')
in_time = pd.read_csv('in_time.csv')
out_time = pd.read_csv('out_time.csv')
gen_data = pd.read_csv('general_data.csv')


# Let's check out the largest of these DataFrames, 'gen_data'.

# In[3]:


# Print info
print(gen_data.info())


# Let's also take a look at the metadata file to make sure we understand what the 'gen_data' columns are referencing.

# In[4]:


# Print metadata
print(data_dictionary)


# Now we can begin with some housekeeping: lets save a copy of the raw general dataframe for later reference, and also set a common index between DataFrames where possible.

# In[5]:


# Save a copy of the original dataframe for later reference
gen_data_orig = gen_data.copy()

# Set indexes to Employee ID
gen_data = gen_data.set_index('EmployeeID')
emp_survey = emp_survey.set_index('EmployeeID')
man_survey = man_survey.set_index('EmployeeID')


# Focusing on the general HR DataFrame, lets do some initial cleaning tasks, like calculating the number of missing values for each column and removing records with missing values.

# In[6]:


# Missing value counts for general data
print(gen_data.isna().sum())

# Remove records with missing values
gen_data.dropna(subset=['NumCompaniesWorked','TotalWorkingYears'], inplace=True)

# Reprint missing value counts to ensure removal of missing records
print(gen_data.isna().sum())


# Next, lets convert non-US units to those used in the United States.

# In[7]:


# Convert commute distance to miles (mi) from kilometers (km)
gen_data['DistanceFromHome'] = gen_data['DistanceFromHome']*0.621371

# Convert monthly income to United States Dollar (USD) from Indian Rupee (INR)
gen_data['MonthlyIncome'] = gen_data['MonthlyIncome']*0.012900993 


# Focusing now on the two timesheet DataFrames, lets again start with initial evaluation and cleaning tasks, like calculating the number of missing values for each column and removing holidays and weekends (dates in which all employees were not present at work).

# In[8]:


# --- WORKING WITH TIMESHEET DATA --- #

# Missing value counts for timesheet data
print(in_time.isna().sum())
print(out_time.isna().sum())

# Find working days and remove holidays
in_time_workdays = in_time.T.dropna(how="all")
out_time_workdays  = out_time.T.dropna(how="all")

# Transpose the DataFrames for later manipulations
in_time_workdays  = in_time_workdays.T
out_time_workdays  = out_time_workdays.T

# Drop irrelevant columns
in_time_workdays = in_time_workdays.drop(labels=['Unnamed: 0'], axis=1)
out_time_workdays = out_time_workdays.drop(labels=['Unnamed: 0'], axis=1)


# Now lets convert our in/out timesheet DataFrames to datetime objects for time calculation.

# In[9]:


# Set columns to index
workdays_cols  = in_time_workdays.columns

# Convert datatype to datetime on 'in' timesheet DataFrame
in_time_workdays[workdays_cols] = in_time_workdays[workdays_cols].apply(pd.to_datetime, errors='coerce')

# Convert datatype to datetime on 'out' timesheet DataFrame
out_time_workdays[workdays_cols] = out_time_workdays[workdays_cols].apply(pd.to_datetime, errors='coerce')


# We can take advantage of the identical configuration of the 'in_time'/'out_time' DataFrames to evaluate the period worked each day by each employee via the .subtract() method. We can then clean the result by replacing missing days (days in which the employee did not clock in) with a timedelta value of 0 seconds.

# In[10]:


# Calculate daily worked hours via subtraction
hours_timedelta = out_time_workdays.subtract(in_time_workdays)

# Replace NaT values with 0 for hours worked
hours_timedelta = hours_timedelta.fillna(pd.Timedelta(seconds=0))


# Next, we can calculate summary statistics over each employee's worked hours and rename our columns to reflect the calculations. 

# In[11]:


# Calculate summary statistics on hours worked
hours_stats = hours_timedelta.T.agg(['mean','max','sum','std'])

# Rename columns
hours_stats = hours_stats.T
hours_stats.columns = ['MeanHrsWorked', 'MaxHrsWorked', 'SumHrsWorked', 'StdHrsWorked']

# Set the index to 'EmployeeID' column for later joining
hours_stats = hours_stats.set_index(gen_data_orig['EmployeeID'])


# For easier calculations, we will now convert the timedelta values in our summary statistics DataFrame to floating point numbers in units of hours.

# In[12]:


# Convert timedelta to time in hours
for column in hours_stats:
    hours_stats[column] = hours_stats[column].dt.total_seconds()/3600


# Back-tracking to the 'in_time' DataFrame, we can quickly calculate the number of sick days/vacation days taken by each employee. This is accomplished by summing the number of days in which the employee did not check in (value is NaT).

# In[13]:


# Initialize sick_days DataFrame
sick_days = pd.DataFrame([])

# Sum the number of days for each employee where no check-in time was recorded (NaT)
sick_days['SickDays'] = in_time_workdays.T.isna().sum() 

# Set the index to the common 'EmployeeID' column for joining
sick_days = sick_days.set_index(gen_data_orig['EmployeeID'])


# Finally, lets inner-join the four cleaned DataFrames to form our final cleaned DataFrame for export.

# In[14]:


# Join the four cleaned DataFrames using an inner-join
df = gen_data.merge(hours_stats, on='EmployeeID')    .merge(sick_days, on='EmployeeID')        .merge(emp_survey, on='EmployeeID')            .merge(man_survey, on='EmployeeID')


# Last, lets check out and export our aggregated and cleaned HR dataset.

# In[15]:


print(df.info())
print(df)


# In[16]:


# export cleaned DataFrame
pd.DataFrame(df).to_csv("HR_data_cleaned.csv" )

# export features
pd.DataFrame(df.columns).to_csv("features.csv" )

