#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')


# 

# In[3]:


sample_submission_df.head()


# In[4]:


train_df.head()


# In[5]:


sns.heatmap(train_df.corr())


# In[6]:


train_df.describe()


# In[7]:


train_df.replace(False, int(0), inplace=True)


# In[8]:


train_df.replace(True, int(1), inplace=True)


# In[9]:


train_df.head()


# In[10]:


train_df.info()


# In[11]:


train_df['Destination'].unique()


# In[12]:


train_df.isnull().sum()


# In[13]:


train_df.fillna(int(train_df['VRDeck'].mean()), inplace=True)


# In[14]:


train_df.fillna(int(train_df['Spa'].mean()), inplace=True)


# In[15]:


train_df.fillna(int(train_df['ShoppingMall'].mean()), inplace=True)


# In[16]:


train_df.fillna(int(train_df['FoodCourt'].mean()), inplace=True)


# In[17]:


train_df.fillna(int(train_df['RoomService'].mean()), inplace=True)


# In[18]:


train_df.fillna(int(train_df['VIP'].mean()), inplace=True)


# In[19]:


train_df.fillna(int(train_df['Age'].mean()), inplace=True)


# In[20]:


train_df.fillna(int(train_df['CryoSleep'].mean()), inplace=True)


# In[21]:


train_df['Destination'].unique()


# In[22]:


train_df.isnull().sum()


# In[23]:


train_df.drop(['Name'], axis=1, inplace=True)


# In[24]:


train_df.drop(['Cabin'], axis=1, inplace=True)


# In[25]:


train_df.info()


# In[26]:


train_df.head()


# In[27]:


train_df.HomePlanet.unique()


# In[28]:


train_df.Destination.unique()


# In[29]:


train_df = pd.get_dummies(train_df, columns = ['HomePlanet', 'Destination'], drop_first=True)


# In[30]:


train_df.head()


# In[31]:


train_df = train_df.rename(columns = {'HomePlanet_Earth': 'HP_Earth', 'HomePlanet_Europa': 'HP_Europa', 'HomePlanet_Mars': 'HP_Mars', 'Destination_55 Cancri e': 'Dest_Cancrie', 'Destination_PSO J318.5-22': 'Dest_PSOJ318.5-22', 'Destination_TRAPPIST-1e': 'Dest_TRAPPIST-1e'})


# In[32]:


train_df.head()


# In[33]:


sns.heatmap(train_df.corr())


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


plt.hist(train_df['Age'], bins=20, facecolor='b')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()


# In[36]:


plt.hist(train_df['RoomService'], bins=40, facecolor='b')
plt.xlabel('RoomService')
plt.ylabel('Count')
plt.title('Distribution of RoomService')
plt.show()


# In[37]:


sns.pairplot(train_df)


# In[38]:


def clean(df):
    df.replace(False, int(0), inplace=True)
    df.replace(True, int(1), inplace=True)
    df.fillna(int(df['VRDeck'].mean()), inplace=True)
    df.fillna(int(df['Spa'].mean()), inplace=True)
    df.fillna(int(df['ShoppingMall'].mean()), inplace=True)
    df.fillna(int(df['FoodCourt'].mean()), inplace=True)
    df.fillna(int(df['RoomService'].mean()), inplace=True)
    df.fillna(int(df['VIP'].mean()), inplace=True)
    df.fillna(int(df['Age'].mean()), inplace=True)
    df.fillna(int(df['CryoSleep'].mean()), inplace=True)
    df.drop(['Cabin'], axis=1, inplace=True)
    df.drop(['Name'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns = ['HomePlanet', 'Destination'], drop_first=True)
    df = df.rename(columns = {'HomePlanet_Earth': 'HP_Earth', 'HomePlanet_Europa': 'HP_Europa', 'HomePlanet_Mars': 'HP_Mars', 'Destination_55 Cancri e': 'Dest_Cancrie', 'Destination_PSO J318.5-22': 'Dest_PSOJ318.5-22', 'Destination_TRAPPIST-1e': 'Dest_TRAPPIST-1e'})
    return(df)


# In[39]:


test_X = clean(test_df)


# In[40]:


test_X.head()


# In[41]:


test_X.shape


# In[42]:


sns.pairplot(test_X)


# In[43]:


sns.pairplot(test_X.corr())


# In[44]:


sns.heatmap(test_X.corr())


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


columns = ['PassengerId', 'CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HP_Earth', 'HP_Europa', 'HP_Mars', 'Dest_Cancrie', 'Dest_PSOJ318.5-22', 'Dest_TRAPPIST-1e']


# In[47]:


X = train_df[columns]
y = train_df['Transported']


# In[48]:


X.head()


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[50]:


X_test.shape, y_test.shape


# In[51]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[53]:


y_pred.shape


# In[54]:


test_X.PassengerId.shape


# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


accuracy_rf = accuracy_score(y_test, y_pred)
print(accuracy_rf)


# In[57]:


test_df.shape


# In[58]:


test_X = test_X.reindex(columns=X_train.columns, fill_value=0)


# In[59]:


test_X.info()


# In[60]:


t_pred = rf.predict(test_X)
t_pred


# In[61]:


t_pred.shape


# In[62]:


submission = pd.DataFrame({'PassengerId':test_X['PassengerId'],'Transported':t_pred})


# In[68]:


submission['Transported'] = list(map(bool, submission['Transported']))


# In[69]:


print(submission)


# In[70]:


submission.to_csv('sample_submission.csv', index=False)


# In[ ]:




