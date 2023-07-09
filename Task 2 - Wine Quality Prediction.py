#!/usr/bin/env python
# coding: utf-8

# ## Importing the Essential Libraries, Metrics

# In[164]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[165]:


# Load the dataset
data = pd.read_csv('D:\Task 2\WineQT.csv')


# In[166]:


#Let's check how the data is distributed
data.head()


# In[167]:


#Information about the data columns
data.info()


# ## Let's do some plotting to know how the data columns are distributed in the dataset

# In[168]:


#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)


# In[169]:


#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = data)


# In[170]:


#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = data)


# In[171]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = data)


# In[172]:


#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = data)


# In[173]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = data)


# In[174]:


#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = data)


# In[175]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = data)


# ## Preprocessing Data for performing Machine learning algorithms

# In[176]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


# In[177]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[178]:


#Bad becomes 0 and good becomes 1 
data['quality'] = label_quality.fit_transform(data['quality'])


# In[179]:


data['quality'].value_counts()


# In[180]:


sns.countplot(data['quality'])


# In[181]:


#Now seperate the dataset as response variable and feature variabes
X = data.drop('quality', axis = 1)
y = data['quality']


# In[182]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[183]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[184]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ## Our training and testing data is ready now to perform machine learning algorithm

# # Linear Regression

# In[185]:


model = LinearRegression()


# In[186]:


# Train the model using the training data
model.fit(X_train, y_train)


# In[187]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[188]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[189]:


# Print the evaluation metrics
print('Mean Squared Error:', mse)
print('R-squared:', r2)


# In[190]:


# Visualize the regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Linear Regression: Actual vs. Predicted Wine Quality')
plt.show()

