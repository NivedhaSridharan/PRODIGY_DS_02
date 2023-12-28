#!/usr/bin/env python
# coding: utf-8

# ![Screenshot%202023-07-11%20200705.png](attachment:Screenshot%202023-07-11%20200705.png)

# ![Screenshot%202023-07-11%20200035.png](attachment:Screenshot%202023-07-11%20200035.png)

# # | Domain Knowledge
# * survival: Target column has two values (0 = No, 1 = Yes).
# * pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
# * sex: male or female.
# * Age: Age of passengers in years.
# * sibsp: number of siblings / spouses aboard the Titanic.
# * parch: number of parents / children aboard the Titanic.
# * ticket: Ticket number.
# * fare: Passenger fare.
# * cabin: Cabin number.
# * embarked: Port of Embarkation has three values (C = Cherbourg, Q = Queenstown, S = Southampton)

#  # --------------------------------------------- Table of Content  ---------------------------------------
# | No | Content |
# |-----|--------|
# | 1 | Introduction |
# | 2 | Data Overview |
# | 3 | Data Cleaning |
# | 4 | Exploratory Data Analysis |

# # 1 | Introduction

# ## I | Import libraries

# In[1]:


import numpy as np
import pandas as pd
import re

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

import warnings
warnings.filterwarnings('ignore')


# ## II | Import data

# In[2]:


data = pd.read_csv("titanicdata.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# # 2 | Data Overview

# In[5]:


print(f'''The shape of data: {data.shape}''')


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.describe(include=['O'])


# In[9]:


round(data['Survived'].mean()*100,2)


# In[10]:


data.isnull().sum()


# In[11]:


# columns which have nulls and the percentage of nulls in each column

data_na = (data.isnull().sum() / len(data)) *100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Percentage of nulls' :data_na , 'Number of nulls' : data[data_na.index].isna().sum()})
missing_data


# In[12]:


data[['Ticket']].duplicated().sum()/len(data)*100


# ## | Observations 
# * The data-set has 891 rows and 11 features + survived column (target feature).
# 
# * Categorical columns: Survived, Sex, and Embarked. Ordinal columns: Pclass.
# 
# * Continous columns: Age, Fare. Discrete columns: SibSp, Parch.
# 
# * Alphanumeric columns: Ticket and Cabin.
# 
# * Around 38.38% of the data-set survived the Titanic.
# 
# * The passenger ages range from 0.4 to 80.
# 
# * Sex column has two values with 65% male (freq=577/count=891).
# 
# * Embarked column has three values. Port S used by 72.4% of passengers.
# 
# * Ticket column contains high ratio of duplicates (23.5%). we might want to drop it.
# 
# * There are three columns in our data have missing values:
# 
#    * Cabin column have almost 77% null values of its data. we might want to drop it.
#    * 177 value in Age column are missed, Around 19% of its data.
#    * Just two values in Embarked are missing, which can easily be filled.
#    
# * SibSp and Parch these features have zero correlation for certain values. We might derive a feature or a set of features from   these individual features.

# # 3 | Data Cleaning

# ## 1- Drop unuseful columns
# * Drop PassengerId column from the data set, because it won't benefit in analysis. 
# 
# * Drop Cabin column, becouse 77% of its data are missing. And a general rule is that, if more than half of the data in a column is missing, it's better to drop it.
# 
# * Drop Ticket column, becouse there may not be a correlation between Ticket and survival and its high ratio of duplicates.
# 
# 

# In[13]:


#Drop PassengerId column 
data.drop(columns='PassengerId', inplace=True)

#Drop Cabin column.
data.drop(columns='Cabin', inplace=True)

#Drop Ticket column
data.drop(columns='Ticket', inplace=True)


# ## 2- Dealing with missing values
# * We will guess Age missing values using random numbers between mean and standard deviation.
# 
# * we will fill missing values of Embarked columns with mode value. As a reminder, we have to deal with just two missing values.

# In[14]:


#Imputing null values of Age column
    
mean = data["Age"].mean()
std = data["Age"].std()
nulls = data["Age"].isnull().sum()
    
# compute random numbers between the mean, std and is_null
random_age = np.random.randint(mean - std, mean + std, size = nulls)
    
# fill NaN values in Age column with random values generated
data["Age"][data["Age"].isna()] = random_age
data["Age"] = data["Age"].astype(int)
    
#Imputing null values of Embarked column
    
data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)    
    


# In[15]:


data.isnull().sum()


# ## 3- Create new columns
# 

# ## I | Create Title column 

# In[16]:


title_list = data['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1]).unique()
title_list


# In[17]:


# replacing all titles with mr, mrs, miss, master, and boy 
def replace_titles(x):
    title=x['Title'].strip()
    
    if (x['Age']<13): return 'Boy'
    
    if title in ['Don', 'Rev', 'Col','Capt','Sir','Major','Jonkheer']: return 'Mr'
    
    elif title in ['Countess', 'Mme']: return 'Mrs'
    
    elif title in ['Mlle', 'Ms','Lady','Dona']: return 'Miss'
    
    elif title =='Dr':
        
        if x['Sex']=='male': return 'Mr'
        else: return 'Mrs'
        
    else: return title

#create a new columns containing the title for each name
data['Title'] = data['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1])
    
#apply replacing title function to all titles
data['Title'] = data.apply(replace_titles, axis=1)


# In[18]:


print(f'Data has : {data["Title"].unique()}')
print()
print(data["Title"].value_counts())


# In[19]:


#drop Name column
data.drop(columns='Name', inplace=True)


# ## II | Create FamilyCount column
# * We can create a FamilyCount feature which combines Parch (number of parents and children) and SibSp (number of siblings and spouses) columns. This will enable us to drop Parch and SibSp from our datasets.

# In[20]:


#create FamilyCount column.
data['FamilyCount'] = data['SibSp'] + data['Parch']+1


# In[21]:


data['FamilyCount'].value_counts()


# ## III | Create IsAlone column 
# * Create a IsAlone feature which contain two values (0 or 1). 0 when family count is 1 means there is one alone person and 1 when family count is more than 1.

# In[22]:


#create IsAlone column.
data.loc[data['FamilyCount'] > 1, 'IsAlone'] = 0
data.loc[data['FamilyCount'] == 1, 'IsAlone'] = 1   
data['IsAlone'] = data['IsAlone'].astype(int)


# In[23]:


data['IsAlone'].value_counts()


# In[24]:


data.groupby(['IsAlone', 'Survived'])['Survived'].count()


# In[25]:


#drop SibSp and Parch column
data.drop(columns='SibSp', inplace=True)
data.drop(columns='Parch', inplace=True)


# In[26]:


data.head()


# # 4 | Exploratory Data Analysis(EDA)

# ## 1- Univariate Analysis

# In[27]:


# Add labels to the end of each bar in a bar chart.

def add_value_labels(ax, spacing=5):

    # For each bar: Place a label    
    for rect in ax.patches:
        
        # Get X and Y placement of label from rect.
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()-3

        # Determine vertical alignment for positive and negative values
        va = 'bottom' if y >= 0 else 'top'

        # Format the label to one decimal place
        label = "{}".format(y)

        # Determine the vertical shift of the label
        # based on the sign of the y value and the spacing parameter
        y_shift = spacing * (1 if y >= 0 else -1)

        # Create the annotation
        ax.annotate(label, (x, y), xytext=(0, y_shift),textcoords="offset points", ha='center', va=va)


# ## I | Analysis of  categorical columns separately

# In[28]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
plt.title('Survived countplot', size=15)
plot= sns.countplot(data=data, x='Survived' ,palette="Set2")
add_value_labels(plot)

plt.subplot(1,3,2)
plt.title('Pclass countplot', size=13)
plot= sns.countplot(data=data, x='Pclass', palette="Set2")
add_value_labels(plot)

plt.subplot(1,3,3)
plt.title('Sex countplot', size=13)
plot= sns.countplot(data=data, x='Sex', palette='Set2')
add_value_labels(plot)

plt.tight_layout()


# In[29]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
plt.title('Pclass-Survived plot', size=15)
plot= sns.countplot(data=data, x='Pclass',hue='Survived' ,palette="Blues")
add_value_labels(plot)

plt.subplot(1,3,2)
plt.title('Sex-Survived plot', size=15)
plot= sns.countplot(data=data, x='Sex', hue='Survived' ,palette="Greens")
add_value_labels(plot)

plt.subplot(1,3,3)
plt.title('Embarked-Survived plot', size=15)
plot= sns.countplot(data=data, x='Embarked',hue='Survived' ,palette="Reds")
add_value_labels(plot)


# In[30]:


data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[31]:


data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# ## 2 | Analysis of  Age column

# In[32]:


#survived passengers
survived_passengers= data[data['Survived']==1]

#non-survived passengers
unsurvived_passengers= data[data['Survived']==0]


# In[33]:


plt.figure(figsize=(13,5))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
sns.histplot(data=survived_passengers, x='Age', kde=True, bins=20,  alpha=0.3 );

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
sns.histplot(data=unsurvived_passengers, x='Age', kde=True, bins=20, alpha=0.3 );


# In[34]:


Infant_passengers = data[data['Age']<=5]
Infant_passengers['Survived'].value_counts(normalize=True)


# In[35]:


Old_passengers = data[data['Age']==80]
Old_passengers['Survived'].value_counts()


# ## | Observations 
# * Pclass=3 had most passengers(488 passengers), however the most of them didn't survive (116 passengers survived and not-survived about 75.8%). 
# 
# * Most passengers in Pclass=1 survived about 62.9%.
# 
# * Infant passengers (Age <=5) had high survival rate, about 70.4% of infant passengers survived.
# 
# * There is only one passengers with 80 years old and he survived.
# 
# * Large number of 15-25 year olds did not survive.
# 
# * Female passengers had much better survival rate than males( 74.2% of female passengers survived but just 18.8% of males survived.)
# 
# * Port S had most passengers(640 passengers) but the most of them didn't survive (424 passengers survived and not-survived about 67%).
# 
# * The majority of port C passengers survived (90 passengers survived out of 162 about 55.5%)

# ## 2- Bivariative Analysis
# 

# ## I | Sex and Age analysis

# In[36]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.title('survived passenger ages')
sns.histplot(data=survived_passengers, x='Age', hue='Sex', kde=True, bins=20,  alpha=0.3 );

plt.subplot(2,2,2)
plt.title('unsurvived passenger ages')
sns.histplot(data=unsurvived_passengers, x='Age',hue='Sex', kde=True, bins=20, alpha=0.3 );

plt.subplot(2,2,3)
plt.title('survived passenger ages')
sns.boxplot(x=survived_passengers['Sex'], y=data["Age"],palette="Set2");

plt.subplot(2,2,4)
plt.title('unsurvived passenger ages')
sns.boxplot(x=unsurvived_passengers['Sex'], y=data["Age"],palette="Set2");


# In[37]:


grid = sns.FacetGrid(data, col='Sex', row='Survived', aspect=1.2)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[38]:


print('surviving male passengers \n')

print(survived_passengers[survived_passengers['Sex']=='male'][['Age']].describe().T)
print('--------------------------------')
print('surviving female passengers \n')

print(survived_passengers[survived_passengers['Sex']=='female'][['Age']].describe().T)


# In[39]:


print('non-surviving male passengers \n')

print(unsurvived_passengers[unsurvived_passengers['Sex']=='male'][['Age']].describe().T)
print('--------------------------------')
print('non-surviving female passengers \n')

print(unsurvived_passengers[unsurvived_passengers['Sex']=='female'][['Age']].describe().T)


# ## II | Pclass and Age analysis 

# In[40]:


plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
sns.boxplot(x=survived_passengers['Pclass'], y=data["Age"],palette="Set2");

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
sns.boxplot(x=unsurvived_passengers['Pclass'], y=data["Age"],palette="Set2");


# In[41]:


grid = sns.FacetGrid(data, col='Pclass', row='Survived', aspect=1.2)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ## III | Sex and Pclass analysis 

# In[42]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_passengers, x='Pclass', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_passengers, x='Pclass', hue='Sex',palette="Set2");
add_value_labels(plot)


# In[43]:


grid = sns.FacetGrid(data, col='Sex', aspect=1.2)
grid.map(sns.pointplot,'Pclass', 'Survived')
grid.add_legend();


# ## IV | Sex and IsAlone analysis

# In[44]:


#survived passengers
survived_passengers= data[data['Survived']==1]

#non-survived passengers
unsurvived_passengers= data[data['Survived']==0]


# In[45]:


grid = sns.FacetGrid(data, col='Sex', aspect=1.2)
grid.map(sns.pointplot,'IsAlone', 'Survived')
grid.add_legend();


# In[46]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_passengers, x='IsAlone', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_passengers, x='IsAlone', hue='Sex',palette="Set2");
add_value_labels(plot)


# ## | Observations 
# * Average age for non-surviving male passengers is 31, And on the other hand non-surviving female passengers is 26 .
# 
# * Most male passengers aged 20-35 did not survive.
# 
# * Infant passengers in Pclass=2 and Pclass=3 mostly survived.
# * Half of the female passengers inside Pclass=3 survive (50% of passengers counted 69).
# * All female passengers inside Pclass=1 survived and about 95 of female passengers inside Pclass=2 survived.
# * About 87% of male passengers inside Pclass=3 and Pclass=2 non-survived but about 36.3% of them survived in Pclass=1.
# * Most of alone men non-survived (About 85%), on other hand about 72% of alone women survived.
# * Almost half of the passengers who are not alone survived(About 50.5%).

# ## 3- Multivariative Analysis

# ## I | Sex, Pclass and Embarked analysis

# In[47]:


grid = sns.FacetGrid(data, col='Embarked', aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
grid.add_legend();


# ## II | Sex, Fare and Embarked analysis 

# In[48]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_passengers, x='Embarked', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_passengers, x='Embarked', hue='Sex',palette="Set2");
add_value_labels(plot)


# In[49]:


grid = sns.FacetGrid(data, col='Embarked', row='Survived', aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend();


# ## | Observations 
# * Women on port Q and S have a higher chance of survival. But it's inverse at port C.
# 
# * Men have a high survival probability on port C, but a low probability on port Q or S.
# 
# * Most female passengers inside Pclass=3 on port C and S non-survived but most of them survived on port Q.
# 
# * Higher fare paying passengers had better survival.
# 
# * Passengers on port Q paid less fare.
# 
# * Nearly no male survived on port Q.
# 
# * Femals on port Q (about 37% of all port Q passengers) Survived, however they paid small fare.
