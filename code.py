import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


# In[334]:


#Importing the dataset
path='churn.csv'
df=pd.read_csv(path,index_col=0)
df.head()


# In[335]:


df.dtypes


# In[336]:


df.describe()


# ### Removing extra columns

# First we remove column just by observation. We can see that we do not need the columns:
# 
# - security_no (irrelevant)
# - referral_id (irrelevant)
# - joining_date (not something we are interested in)
# - last_visit_time (since we have days_since_last_login)
# - avg_frequency_login_days (since we have days_since_last_login)
# 

# In[337]:


df=df.drop(columns=['security_no', 'referral_id','joining_date','last_visit_time','avg_frequency_login_days'])


# In[338]:


df.shape


# ### Remove NaN values

# In[339]:


#Checing NaN values in each column
check_nan = df.isnull().sum()
print("The number of NaN values in each column are: \n",check_nan)

#Removing rows with NaN values as the remaining number of rows are sufficuent
print("Dropping rows with NaN values")
df=df.dropna()

# #Checking the rows again for NaN values
# check_nan_post_processing=df.isnull().sum()
# print("The number of NaN values in each column are: \n",check_nan_post_processing)


# In[340]:


print("Shape of the dataset after removing NaN and error values: \n", df.shape)


# ### Categorical Features

# #### Removing Junk Variables

# In[341]:


# #Getting the names of categorical columns
category = ['object']
categorical_column=df.select_dtypes(include=category)
categorical_column.columns


# In[342]:


categorical_column.columns


# In[343]:


df['medium_of_operation'].value_counts()


# In[344]:


df['medium_of_operation'].unique()


# In[345]:


fig, axs = plt.subplots(len(categorical_column.columns))
fig.set_figheight(40)
fig.set_figwidth(5)
for i,j in enumerate(categorical_column.columns):
    sns.histplot(ax=axs[i],data=df,x=j,hue=j,multiple="stack",shrink=0.3)
    


# Observing these, we can see that there are junk values in:
# - gender
# - joined_through_referral
# - medium_of_operation

# In[346]:


#gender
df = df.loc[df.loc[:,'gender'].isin(['M','F']), :].copy(deep=True)
df.shape


# In[347]:


df = df.loc[df.loc[:,'joined_through_referral'].isin(['No','Yes']), :].copy(deep=True)
df.shape


# In[348]:


df = df.loc[~df.loc[:,'medium_of_operation'].isin(['?']), :].copy(deep=True)
df.reset_index(inplace=True,drop=True)
df.shape


# In[312]:


df.shape


# #### Encoding

# In[349]:


from sklearn.preprocessing import OneHotEncoder


# In[350]:


enc = OneHotEncoder(handle_unknown='ignore')
for i in categorical_column.columns:
    x=np.array(df[i]).reshape(-1,1)
    y=enc.fit_transform(x).toarray()
    converted_list = [[str(int(i)) for i in sublist] for sublist in y]
    converted_list = [[''.join(sublist)] for sublist in converted_list]
    df[i]=np.array(converted_list)

df


# ### Numerical 

# #### Removing Outliers

# In[351]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#Getting numeric column
df_numeric = df.select_dtypes(include=numerics)
numeric_col=df_numeric.columns

#Deciding the shape of the subplot
shape_subplot=int((len(numeric_col)/2))

# #Plotting boxplots
figure, axis = plt.subplots(2,shape_subplot)
figure.set_figheight(10)
figure.set_figwidth(20)
for i,j in enumerate (numeric_col):
    if i<3:
        axis[0,i].boxplot(df[str(j)])
        axis[0, i].set_title(j)
    else:
        axis[1,i-3].boxplot(df[str(j)])
        axis[1, i-3].set_title(j)


# Through this we can clearly see there outliers are present in the column:
# - days_since_last_login -> There is a negative value that needs to be removed
# - avg_time_spent -> There are negative values that need to be removed
# - point_in_wallet -> There are negative values that need to be removed

# In[352]:


df=df[df['days_since_last_login']>0]
print(df.shape)
df=df[df['avg_time_spent']>0]
print(df.shape)
df=df[df['points_in_wallet']>0]
print(df.shape)


# In[353]:


# #Plotting boxplots after removing outliers
figure, axis = plt.subplots(2,shape_subplot)
figure.set_figheight(10)
figure.set_figwidth(20)
plt.title("Before")
for i,j in enumerate (numeric_col):
    if i<3:
        axis[0,i].boxplot(df[str(j)])
        axis[0, i].set_title(j)
    else:
        axis[1,i-3].boxplot(df[str(j)])
        axis[1, i-3].set_title(j)


# In[354]:


numerical_column.columns


# #### Scaling

# In[356]:


#scale the target variable
# apply normalization techniques
for column in numerical_column.columns:
    df[column] = df[column]  / df[column].abs().max()
df


# ### Checking Target Variable

# In[357]:


#We check to see if the target variable is balanced or not
#define Seaborn color palette to use
colors = sns.color_palette('pastel')

#create pie chart
pie=df['churn_risk_score'].value_counts()
pie.plot.pie(autopct="%.1f%%")


# ### Correlation

# In[358]:


x=cor.corr()
sns.heatmap(x)


# We can clearly observe that the correlation between the variables is not high.
# 
# Since the variables are negatively correlated, this indicated they are independant from each other we can use decison trees, random trees and random forests as our classifiers

# ### Machine Learning

# #### Split Training and Testing Data
# 

# In[359]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=30)

print("Shape of x_train:  ", x_train.shape)
print("Shape of x_test:  ", x_test.shape)
print("Shape of y_train: ", y_train.shape)
print('Shape of y_test: ',y_test.shape)

x_train.to_csv('x_train.csv')
x_test.to_csv('x_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')


# #### Models

# ##### Decision Tree

# In[329]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
preds=clf.predict(x_test)

from sklearn.metrics import confusion_matrix
y_test=y_test
preds=preds
cm=confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(cm)
#Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True,cmap="crest",fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ##### Logistic Regression

# In[360]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

preds=clf.predict(x_test)

from sklearn.metrics import confusion_matrix
y_test=y_test
preds=preds
cm=confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(cm)
#Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True,cmap="crest",fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ##### Random Forest

# In[361]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
preds=clf.predict(x_test)

from sklearn.metrics import confusion_matrix
y_test=y_test
preds=preds
cm=confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(cm)
#Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True,cmap="crest",fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[ ]:




