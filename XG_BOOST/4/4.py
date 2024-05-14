#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[ ]:





# In[35]:


df = sns.load_dataset('titanic')


# In[ ]:





# In[36]:


df.dropna(inplace=True)


# In[ ]:





# In[37]:


X = df[['pclass', 'sex', 'age']].copy()


# In[ ]:





# In[38]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()


# In[ ]:





# In[39]:


X['sex'] = lb.fit_transform(X['sex'])


# In[ ]:





# In[40]:


y = df['survived']


# In[ ]:





# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:





# In[43]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:





# In[44]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        


# In[ ]:





# In[45]:


import xgboost as xgb


# In[46]:


xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3,
                            n_jobs=-1)


# In[ ]:





# In[47]:


xgb_clf.fit(X_train, y_train)


# In[ ]:





# In[48]:


print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)


# In[ ]:





# In[49]:


print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# # | Classifier | Decision Tree | Bagging | Random Forest | Optimised RF | Extra-Trees | AdaBoost (CART) | AdaBoost (RF) | Gradient Boosting |
# |:-|:-|:- |:- |:- |:- |:-|:-| :- |
# | Train accuracy score | 0.9528 | 0.9528 | 0.9325 | 0.9264 | 0.9448 | 0.8661 | 0.9528 | 0.9449 |
# | Average accuracy score | 0.7724 | 0.7879 | 0.7801 | 0.7059 | 0.7548 | 0.7793 | 0.7353 | 0.7906 |
# | SD | 0.1018 | 0.1008 | 0.1474 | 0.1308 | 0.1406 | 0.1172 | 0.0881 | 0.0912 |
# | Test accuracy score | 0.7636 | 0.7455 | 0.7895 | 0.6316 | 0.7895 | 0.6545 | 0.7818 | 0.7818 |
# 
# 
