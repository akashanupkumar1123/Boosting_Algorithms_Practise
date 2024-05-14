#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("mushrooms.csv")


# In[3]:


df.head()


# In[ ]:





# In[4]:


sns.countplot(data=df,x='class')


# In[ ]:





# In[5]:


df.describe()


# In[ ]:





# In[6]:


df.describe().transpose()


# In[ ]:





# In[7]:


plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=df.describe().transpose().reset_index().sort_values('unique'),x='index',y='unique')
plt.xticks(rotation=90);


# In[8]:


X = df.drop('class',axis=1)


# In[9]:


X = pd.get_dummies(X,drop_first=True)


# In[10]:


y = df['class']


# In[11]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[ ]:





# In[13]:


from sklearn.ensemble import AdaBoostClassifier


# In[14]:


model = AdaBoostClassifier(n_estimators=1)


# In[15]:


model.fit(X_train,y_train)


# In[ ]:





# In[16]:


from sklearn.metrics import classification_report,plot_confusion_matrix,accuracy_score


# In[17]:


predictions = model.predict(X_test)


# In[18]:


predictions


# In[ ]:





# In[19]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[20]:


model.feature_importances_


# In[ ]:





# In[21]:


model.feature_importances_.argmax()


# In[22]:


X.columns[22]


# In[ ]:





# In[23]:


sns.countplot(data=df,x='odor',hue='class')


# In[ ]:





# In[ ]:


# Analyzi ng performance as more weak learners are added


# In[24]:


len(X.columns)


# In[25]:


error_rates = []

for n in range(1,96):
    
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    err = 1 - accuracy_score(y_test,preds)
    
    error_rates.append(err)


# In[ ]:





# In[26]:


plt.plot(range(1,96),error_rates)


# In[27]:


model


# In[ ]:





# In[29]:


model.feature_importances_


# In[ ]:





# In[30]:


feats = pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Importance'])


# In[31]:


feats


# In[ ]:





# In[32]:


imp_feats = feats[feats['Importance']>0]


# In[34]:


imp_feats


# In[35]:


imp_feats = imp_feats.sort_values("Importance")


# In[36]:


plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')

plt.xticks(rotation=90);


# In[37]:


sns.countplot(data=df,x='habitat',hue='class')


# In[ ]:


#Interesting to see how the importance of the features shift as 
#more are allowed to be added in! But remember these are all weak learner
#stumps, and feature importance is available for all the tree methods!

