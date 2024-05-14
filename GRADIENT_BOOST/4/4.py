#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[ ]:





# In[2]:


df = sns.load_dataset('titanic')


# In[3]:


df.dropna(inplace=True)


# In[4]:


df['pclass'].unique()


# In[5]:


df['pclass'].value_counts()


# In[17]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[21]:


from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
def print_score(clf, X_train, X_test, y_train, y_test, train=True):
    '''
    v0.1 Follow the scikit learn library format in terms of input
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    if train:
        '''
        training performance
        '''
        res = clf.predict(X_train)
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, 
                                                                res)))
        print("Classification Report: \n {}\n".format(classification_report(y_train, 
                                                                            res)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, 
                                                                  res)))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), 
                                                      lb.transform(res))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        res_test = clf.predict(X_test)
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, 
                                                                res_test)))
        print("Classification Report: \n {}\n".format(classification_report(y_test, 
                                                                            res_test)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, 
                                                                  res_test)))   
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), 
                                                      lb.transform(res_test))))
        


# In[ ]:





# In[6]:


df['sex'].unique()


# In[7]:


df['sex'].value_counts()


# In[ ]:





# In[8]:


df['age'].hist(bins=50);


# In[ ]:





# In[ ]:





# In[9]:


subset = df[['pclass', 'sex', 'age', 'survived']].copy()
subset.dropna(inplace=True)
X = df[['pclass', 'sex', 'age']].copy()


# In[ ]:





# In[10]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()


# In[ ]:





# In[11]:


X['sex'] = lb.fit_transform(X['sex'])


# In[12]:


X.head()


# In[ ]:





# In[13]:


y = subset['survived']


# In[14]:


y.value_counts()


# In[ ]:





# In[15]:


from sklearn.ensemble import GradientBoostingClassifier


# In[19]:


gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)


# In[ ]:





# In[22]:


print_score(gbc_clf, X_train, X_test, y_train, y_test, train=True)
print("\n*****************************\n")
print_score(gbc_clf, X_train, X_test, y_train, y_test, train=False)


# In[ ]:




