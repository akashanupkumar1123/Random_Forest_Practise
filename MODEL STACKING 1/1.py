#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
num_col = list(df.describe().columns)
col_categorical = list(set(df.columns).difference(num_col))
remove_list = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
col_numerical = [e for e in num_col if e not in remove_list]
attrition_to_num = {'Yes': 0,
                    'No': 1}
df['Attrition_num'] = df['Attrition'].map(attrition_to_num)
col_categorical.remove('Attrition')
df_cat = pd.get_dummies(df[col_categorical])
X = pd.concat([df[col_numerical], df_cat], axis=1)
y = df['Attrition_num']


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[5]:


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
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), 
                                                        lb.transform(clf.predict(X_train)))))

        #cv_res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        #print("Average Accuracy: \t {0:.4f}".format(np.mean(cv_res)))
        #print("Accuracy SD: \t\t {0:.4f}".format(np.std(cv_res)))
        
    elif train==False:
        '''
        test performance
        '''
        res_test = clf.predict(X_test)
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))      
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), lb.transform(res_test))))
        


# In[ ]:





# In[ ]:


# Model 1 ) DT


# In[6]:


from sklearn.tree import DecisionTreeClassifier


# In[7]:


tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)


# In[8]:


print_score(tree_clf, X_train, X_test, y_train, y_test, train=True)
print_score(tree_clf, X_train, X_test, y_train, y_test, train=False)


# In[ ]:





# In[ ]:


# Model 2 ) RF


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train.ravel())


# In[11]:


print_score(rf_clf, X_train, X_test, y_train, y_test, train=True)
print_score(rf_clf, X_train, X_test, y_train, y_test, train=False)


# In[12]:


en_en = pd.DataFrame()


# In[13]:


tree_clf.predict_proba(X_train)


# In[ ]:





# In[14]:


en_en['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_train))[1]
en_en['rf_clf'] =  pd.DataFrame(rf_clf.predict_proba(X_train))[1]
col_name = en_en.columns
en_en = pd.concat([en_en, pd.DataFrame(y_train).reset_index(drop=True)], axis=1)


# In[ ]:





# In[15]:


en_en.head()


# In[16]:


tmp = list(col_name)
tmp.append('ind')
en_en.columns = tmp


# In[17]:


en_en.head()


# In[ ]:





# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


m_clf = LogisticRegression(fit_intercept=False, solver='lbfgs')


# In[20]:


m_clf.fit(en_en[['tree_clf', 'rf_clf']], en_en['ind'])


# In[21]:


en_test = pd.DataFrame()


# In[22]:


en_test['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_test))[1]
en_test['rf_clf'] =  pd.DataFrame(rf_clf.predict_proba(X_test))[1]
col_name = en_en.columns
en_test['combined'] = m_clf.predict(en_test[['tree_clf', 'rf_clf']])


# In[24]:


col_name = en_test.columns
tmp = list(col_name)
tmp.append('ind')


# In[ ]:





# In[25]:


tmp


# In[26]:


en_test = pd.concat([en_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)


# In[27]:


en_test.columns = tmp


# In[ ]:





# In[28]:


print(pd.crosstab(en_test['ind'], en_test['combined']))


# In[ ]:





# In[29]:


print(round(accuracy_score(en_test['ind'], en_test['combined']), 4))


# In[ ]:





# In[30]:


print(classification_report(en_test['ind'], en_test['combined']))


# In[ ]:




