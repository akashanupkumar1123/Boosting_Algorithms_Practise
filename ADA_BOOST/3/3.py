#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# BAGIN MACHINE LEARNING ALGORITHM


# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


df = sns.load_dataset('titanic')


# In[3]:


df.shape


# In[4]:


df.head()


# In[ ]:





# In[24]:


df.dropna(inplace=True)


# In[25]:


df['pclass'].unique()


# In[26]:


df['pclass'].value_counts()


# In[27]:


df['sex'].value_counts()


# In[ ]:





# In[28]:


df['age'].hist(bins=50)


# In[ ]:





# In[29]:


subset = df[['pclass', 'sex', 'age', 'survived']].copy()
subset.dropna(inplace=True)
X = df[['pclass', 'sex', 'age']].copy()


# In[30]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()


# In[31]:


X['sex'] = lb.fit_transform(X['sex'])


# In[32]:


X.head()


# In[ ]:





# In[33]:


X.shape


# In[ ]:





# In[34]:


X.describe()


# In[ ]:





# In[35]:


X.info()


# In[ ]:





# In[36]:


y = subset['survived']


# In[37]:


y.value_counts()


# In[ ]:





# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3)


# In[ ]:





# In[40]:


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





# In[ ]:


# ADAPTIVE BOOSTING


# In[41]:


from sklearn.ensemble import AdaBoostClassifier


# In[42]:


ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)


# In[43]:


ada_clf.fit(X_train, y_train)


# In[ ]:





# In[44]:


print_score(ada_clf, X_train, X_test, y_train, y_test, train=True)

print("\n**************************************\n")
print_score(ada_clf, X_train, X_test, y_train, y_test, train= False)


# In[ ]:





# In[ ]:


#ADABOOST with RANDOM FOREST


# In[45]:


from sklearn.ensemble import RandomForestClassifier


# In[46]:


ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100), n_estimators=100)


# In[47]:


ada_clf.fit(X_train, y_train)


# In[ ]:


print_score(ada_clf, X_train, X_test, y_train, y_test, train=True)
print("\n***************************\n")
print_score(ada_clf, X_train, X_test, y_train, y_test,train = False)


# In[ ]:




