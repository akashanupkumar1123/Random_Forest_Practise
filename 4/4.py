#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("penguins_size.csv")


# In[3]:


df = df.dropna()
df.head()


# In[ ]:





# In[4]:


X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:





# In[7]:


from sklearn.ensemble import RandomForestClassifier


# In[8]:


help(RandomForestClassifier)


# In[ ]:





# In[9]:


# Use 10 random trees
model = RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)


# In[11]:


model.fit(X_train,y_train)


# In[ ]:





# In[12]:


preds = model.predict(X_test)


# In[ ]:





# In[13]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score


# In[ ]:





# In[14]:


confusion_matrix(y_test,preds)


# In[15]:


plot_confusion_matrix(model,X_test,y_test)


# In[ ]:





# In[16]:


model.feature_importances_


# In[ ]:





# In[17]:


test_error = []

for n in range(1,40):
    # Use n random trees
    model = RandomForestClassifier(n_estimators=n,max_features='auto')
    model.fit(X_train,y_train)
    test_preds = model.predict(X_test)
    test_error.append(1-accuracy_score(test_preds,y_test))


# In[ ]:





# In[18]:


plt.plot(range(1,40),test_error,label='Test Error')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


df = pd.read_csv("data_banknote_authentication.csv")


# In[20]:


df.head()


# In[21]:


sns.pairplot(df,hue='Class')


# In[22]:


X = df.drop("Class",axis=1)


# In[25]:


y = df["Class"]


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)


# In[29]:


from sklearn.model_selection import GridSearchCV


# In[30]:


n_estimators=[64,100,128,200]
max_features= [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]


# In[31]:


param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}  # Note, oob_score only makes sense when bootstrap=True!


# In[ ]:





# In[42]:


rfc = RandomForestClassifier()
grid = GridSearchCV(rfc,param_grid)


# In[ ]:





# In[43]:


grid.fit(X_train,y_train)


# In[ ]:





# In[44]:


grid.best_params_


# In[45]:


predictions = grid.predict(X_test)


# In[46]:


print(classification_report(y_test,predictions))


# In[ ]:





# In[47]:


plot_confusion_matrix(grid,X_test,y_test)


# In[ ]:





# In[48]:


# No underscore, reports back original oob_score parameter
grid.best_estimator_.oob_score


# In[ ]:





# In[49]:


# With underscore, reports back fitted attribute of oob_score
grid.best_estimator_.oob_score_


# In[ ]:





# In[50]:


from sklearn.metrics import accuracy_score


# In[ ]:





# In[51]:


errors = []
misclassifications = []

for n in range(1,64):
    rfc = RandomForestClassifier( n_estimators=n,bootstrap=True,max_features= 2)
    rfc.fit(X_train,y_train)
    preds = rfc.predict(X_test)
    err = 1 - accuracy_score(preds,y_test)
    n_missed = np.sum(preds != y_test) 
    errors.append(err)
    misclassifications.append(n_missed)


# In[ ]:





# In[52]:


plt.plot(range(1,64),errors)


# In[ ]:





# In[53]:


plt.plot(range(1,64),misclassifications)


# In[ ]:




