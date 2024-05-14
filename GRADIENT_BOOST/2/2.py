#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import warnings


# In[2]:


pima = pd.read_csv('diabetes.csv')


# In[4]:


pima.head()


# In[5]:


pima.describe()


# In[ ]:





# In[6]:


pima.info()


# In[ ]:





# In[7]:


X_train, X_test, y_train, y_test = train_test_split(pima.drop('Outcome', axis=1),
                                                    pima['Outcome'], test_size=0.2)


# In[8]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


# In[ ]:





# In[9]:


# Define Gradient Boosting Classifier with hyperparameters
gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.05,random_state=100,max_features=5 )
# Fit train data to GBC
gbc.fit(X_train_transformed, y_train)


# In[ ]:





# In[10]:


# Confusion matrix will give number of correct and incorrect classifications
print(confusion_matrix(y_test, gbc.predict(X_test_transformed)))


# In[ ]:





# In[11]:


# Accuracy of model
print("GBC accuracy is %2.2f" % accuracy_score(
    y_test, gbc.predict(X_test_transformed)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




