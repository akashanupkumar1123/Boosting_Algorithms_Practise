#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))


# In[ ]:





# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)


# In[ ]:





# In[3]:


import xgboost as xgb

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)


# In[ ]:





# In[4]:


param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 3} 
epochs = 10 


# In[ ]:





# In[5]:


model = xgb.train(param, train, epochs)


# In[ ]:





# In[6]:


predictions = model.predict(test)


# In[7]:


print(predictions)


# In[8]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)


# In[ ]:


"""Holy crow! It's perfect, and that's just with us guessing as to the 
best hyperparameters!

Normally I'd have you experiment to find better hyperparameters as 
an activity, but you can't improve on those results. Instead, 
see what it takes to make the results worse! How few epochs (iterations) 
can I get away with? How low can I set the max_depth? Basically try to 
the simplicity and performance of the model, now that you already have 
perfect accuracy.
"""

