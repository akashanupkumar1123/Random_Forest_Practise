#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("rock_density_xray.csv")


# In[3]:


df.head()


# In[ ]:





# In[4]:


df.columns=['Signal',"Density"]


# In[5]:


plt.figure(figsize=(12,8),dpi=200)
sns.scatterplot(x='Signal',y='Density',data=df)


# In[ ]:





# In[6]:


X = df['Signal'].values.reshape(-1,1)  
y = df['Density']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[ ]:





# In[ ]:


#1) linear regression


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


lr_model = LinearRegression()


# In[11]:


lr_model.fit(X_train,y_train)


# In[12]:


lr_preds = lr_model.predict(X_test)


# In[13]:


from sklearn.metrics import mean_squared_error


# In[14]:


np.sqrt(mean_squared_error(y_test,lr_preds))


# In[15]:


signal_range = np.arange(0,100)


# In[16]:


lr_output = lr_model.predict(signal_range.reshape(-1,1))


# In[17]:


plt.figure(figsize=(12,8),dpi=200)
sns.scatterplot(x='Signal',y='Density',data=df,color='black')
plt.plot(signal_range,lr_output)


# In[ ]:





# In[ ]:


#2) polynomial regression


# In[18]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[19]:


def run_model(model,X_train,y_train,X_test,y_test):
    
    # Fit Model
    model.fit(X_train,y_train)
    
    # Get Metrics
    
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    print(f'RMSE : {rmse}')
    
    # Plot results
    signal_range = np.arange(0,100)
    output = model.predict(signal_range.reshape(-1,1))
    
    
    plt.figure(figsize=(12,6),dpi=150)
    sns.scatterplot(x='Signal',y='Density',data=df,color='black')
    plt.plot(signal_range,output)


# In[20]:


run_model(model,X_train,y_train,X_test,y_test)


# In[ ]:





# In[21]:


from sklearn.pipeline import make_pipeline


# In[22]:


from sklearn.preprocessing import PolynomialFeatures


# In[23]:


pipe = make_pipeline(PolynomialFeatures(2),LinearRegression())


# In[24]:


run_model(pipe,X_train,y_train,X_test,y_test)


# In[25]:


pipe = make_pipeline(PolynomialFeatures(10),LinearRegression())
run_model(pipe,X_train,y_train,X_test,y_test)


# In[ ]:





# In[ ]:


#3) knn regression


# In[26]:


from sklearn.neighbors import KNeighborsRegressor


# In[27]:


preds = {}
k_values = [1,5,10]
for n in k_values:
    
    
    model = KNeighborsRegressor(n_neighbors=n)
    run_model(model,X_train,y_train,X_test,y_test)


# In[ ]:





# In[ ]:


#4) Decision tree


# In[28]:


from sklearn.tree import DecisionTreeRegressor


# In[29]:


model = DecisionTreeRegressor()

run_model(model,X_train,y_train,X_test,y_test)


# In[30]:


model.get_n_leaves()


# In[ ]:





# In[ ]:


#5) support vector regression


# In[34]:


from sklearn.svm import SVR


# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


param_grid = {'C':[0.01,0.1,1,5,10,100,1000],'gamma':['auto','scale']}
svr = SVR()


# In[37]:


grid = GridSearchCV(svr,param_grid)


# In[38]:


run_model(grid,X_train,y_train,X_test,y_test)


# In[39]:


grid.best_estimator_


# In[ ]:





# In[ ]:


#6) random forest regression


# In[40]:


from sklearn.ensemble import RandomForestRegressor


# In[41]:


trees = [10,50,100]
for n in trees:
    
    model = RandomForestRegressor(n_estimators=n)
    
    run_model(model,X_train,y_train,X_test,y_test)


# In[ ]:





# In[ ]:


#7) gradient boosting


# In[42]:


from sklearn.ensemble import GradientBoostingRegressor


# In[43]:


help(GradientBoostingRegressor)


# In[ ]:





# In[44]:



model = GradientBoostingRegressor()

run_model(model,X_train,y_train,X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:


#8) Adaboost


# In[45]:


from sklearn.ensemble import AdaBoostRegressor


# In[46]:


model = GradientBoostingRegressor()

run_model(model,X_train,y_train,X_test,y_test)


# In[ ]:




