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


# In[4]:


X = df.drop('class', axis =1)


# In[5]:


y = df['class']


# In[6]:


X = pd.get_dummies(X,drop_first=True)


# In[7]:


X.head()


# In[ ]:





# In[8]:


y.head()


# In[9]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[12]:


from sklearn.ensemble import GradientBoostingClassifier


# In[13]:


help(GradientBoostingClassifier)


# In[ ]:





# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


param_grid = {"n_estimators":[1,5,10,20,40,100],'max_depth':[3,4,5,6]}


# In[ ]:





# In[17]:


gb_model = GradientBoostingClassifier()


# In[18]:


grid = GridSearchCV(gb_model,param_grid)


# In[19]:


grid.fit(X_train,y_train)


# In[ ]:





# In[20]:


from sklearn.metrics import classification_report,plot_confusion_matrix,accuracy_score


# In[21]:


predictions = grid.predict(X_test)


# In[ ]:





# In[22]:


predictions


# In[23]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[24]:


grid.best_estimator_.feature_importances_


# In[ ]:





# In[25]:


feat_import = grid.best_estimator_.feature_importances_


# In[ ]:





# In[27]:


imp_feats = pd.DataFrame(index=X.columns,data=feat_import,columns=['Importance'])


# In[ ]:





# In[28]:


imp_feats


# In[29]:


imp_feats.sort_values("Importance",ascending=False)


# In[30]:


imp_feats.describe().transpose()


# In[32]:


imp_feats = imp_feats[imp_feats['Importance'] > 0.000527]


# In[33]:


imp_feats.sort_values('Importance')


# In[34]:


plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')
plt.xticks(rotation=90);


# In[ ]:




