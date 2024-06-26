#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loans = pd.read_csv('loan_data.csv')


# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


loans.head()


# In[ ]:





# In[6]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:





# In[7]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:





# In[8]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# In[ ]:





# In[9]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# In[ ]:





# In[10]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[ ]:





# In[11]:


loans.info()


# In[ ]:





# In[12]:


cat_feats = ['purpose']


# In[ ]:





# In[13]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[ ]:





# In[14]:


final_data.info()


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[16]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:





# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# In[18]:


dtree = DecisionTreeClassifier()


# In[ ]:





# In[19]:


dtree.fit(X_train,y_train)


# In[ ]:





# In[20]:


predictions = dtree.predict(X_test)


# In[ ]:





# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:





# In[22]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[23]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rfc = RandomForestClassifier(n_estimators=600)


# In[ ]:





# In[26]:


rfc.fit(X_train,y_train)


# In[ ]:





# In[27]:


predictions = rfc.predict(X_test)


# In[ ]:





# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:





# In[29]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[30]:


print(confusion_matrix(y_test,predictions))


# In[ ]:





# In[31]:


# Depends what metric you are trying to optimize for. 
# Notice the recall for each class for the models.
# Neither did very well, more feature engineering is needed.

