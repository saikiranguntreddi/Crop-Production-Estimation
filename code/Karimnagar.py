#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df=pd.read_csv('District\\Karimnagar.csv')

le = preprocessing.LabelEncoder()

df[df.Year != '2016-17']

le.fit(df['Year'])
k=le.transform(df['Year'])
df['Year']=k

del df['Year']
del df['May']

X = np.array(df.drop(['Production (Tonnes)'], 1))
y = np.array(df['Production (Tonnes)'])

seed=32
X,y=shuffle(X,y,random_state=seed)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[24]:



lis=[]
lis1=[]
lis2=[]
k=10000000000

for i in range(1,300, 5):
    for j in range(1, 32):
        #print(i,j, ":")
        rfr = RandomForestRegressor(max_depth=j, n_estimators=i,random_state=False, 
                                    verbose=False)

        rfr.fit(x_train, y_train)
        y_pred = rfr.predict(x_test)

        #print('Coefficients: \n', rfr.coef_)

        #print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
        #print('Variance score: %.2f' % r2_score(y_test, y_pred))
        m=mean_squared_error(y_test, y_pred)
        o=mean_absolute_error(y_test, y_pred)

        #print(df.std())

        ytrain_pred = rfr.predict(x_train)

        #print("Mean squared error: %.2f"% mean_squared_error(y_train, ytrain_pred))
        #print('Variance score: %.2f' % r2_score(y_train, ytrain_pred))
        n=mean_squared_error(y_train, ytrain_pred)
        p=mean_absolute_error(y_train, ytrain_pred)
        if(m>n):
            lis.append(m-n)
            if o<k:
                k=o
                nestim=i
                dep=j
        else:
            lis.append(n-m)
            if o<k:
                k=o
                nestim=i
                dep=j
            
        lis1.append(o)
        lis2.append(p)

min(lis)


# In[25]:


print(nestim, dep)

rfr = RandomForestRegressor(max_depth=nestim, n_estimators=dep,random_state=False, 
                            verbose=False)

rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)

#print('Coefficients: \n', rfr.coef_)

#print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
#print('Variance score: %.2f' % r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
mean_absolute_error(y_test, y_pred)


# In[20]:


scores = cross_val_score(rfr, x_train, y_train, cv=3, scoring='neg_mean_absolute_error')

predictions = cross_val_predict(rfr, X, y, cv=10)

scores


# In[21]:


predictions


# In[22]:


scoring = {'abs_error': 'neg_mean_absolute_error','squared_error': 'neg_mean_squared_error'}

scores = cross_validate(rfr, X, y, cv=10, scoring=scoring, return_train_score=True)
print("MAE :", abs(scores['test_abs_error'].mean()), "| RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())))


# In[23]:


data={'pred': y_pred , 'actual': y_test}
df1=pd.DataFrame(data)

#df1.to_csv('C:\\Users\\vinay\\OneDrive\\Desktop\\adisample.csv', index= False)


df1.plot.bar()
plt.bar(df1['pred'], df1['actual']) 
plt.show()


# In[ ]:




