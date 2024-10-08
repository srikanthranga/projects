#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install seaborn')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("audit_risk.csv")
df.head()


# In[3]:


#Audit_Risk Dataset has 776 rows and 27 columns. 
df.shape


# In[4]:


#below table demonstrate statiscal summary across all the features and dependant variables 
df.describe().T


# In[5]:


#Below table provides features types for each column. The dataset has 23 column of double-precision Float64 real number, 3 columns of 64-bit integer data and 1 object column
df.dtypes


# In[6]:


#As per following table there is only one null value in Money_Value column
df.isnull().sum()


# In[7]:


#below Histogram charts demonstrate distribution type across different features. it is observed that most of the features has non-normal distribution with some features distribution scattered across the range.

plt.figure(1,figsize=(20, 2))
n=0
for x in ['Sector_score', 'Risk_A', 'Risk_B', 'Risk_C', 'Money_Value', 'Risk_D', 'District_Loss']:
    n+=1
    plt.subplot(1,7,n)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    sns.histplot(df[x], bins=10, kde=True, color='blue', stat='density')
    plt.title ('Histplot of {}'.format(x))
plt.show


# In[6]:


plt.figure(1,figsize=(20, 2))
n=0
for x in ['RiSk_E', 'Risk_F', 'Inherent_Risk', 'CONTROL_RISK','Detection_Risk', 'Audit_Risk']:
    n+=1
    plt.subplot(1,6,n)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    sns.histplot(df[x], bins=10, kde=True, color='blue', stat='density')
    plt.title ('Histplot of {}'.format(x))
plt.show


# In[8]:


#In order to get more insight from some of the features, we plotted cross table between some features and target column (“Risk”). 
#The first feature is Sector_Score which sector score 2.37, 2.72, 3.41 and 3.89 sectors have demonstrated the highest amount of being fraudulent. 

Risk_crosstab = pd.crosstab (df["Risk"], df["Sector_score"], margins=False)
sns.countplot(x="Sector_score", hue="Risk", data=df)
Risk_norm = Risk_crosstab.div(Risk_crosstab.sum(axis=1) ,axis=0)
Risk_norm.plot(kind = 'bar', stacked =True)


# In[9]:


#For District_Loss, District #2 has the highest amount of being fraudulent firm.

Risk_crosstab2 = pd.crosstab(df["Risk"], df["District_Loss"], margins=False)
sns.countplot(x="District_Loss", hue="Risk", data=df)
Risk_norm2 = Risk_crosstab2.div(Risk_crosstab2.sum(axis=1) ,axis=0)
Risk_norm2.plot(kind = 'bar', stacked =True)


# In[35]:


#For Location_ID, Locations #8, #19 and #2 have this highest amount of fradulant firms. 

Risk_crosstab3 = pd.crosstab(df["Risk"], df["LOCATION_ID"], margins=False)
sns.countplot(x="LOCATION_ID", hue="Risk", data=df)
plt.legend(title='Risk', bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize='small', title_fontsize='medium', labelspacing=0.01, borderaxespad=0)
plt.xticks(rotation=90) 
plt.show()
Risk_norm3 = Risk_crosstab3.div(Risk_crosstab3.sum(axis=1) ,axis=0)
Risk_norm3.plot(kind = 'bar', stacked =True)
plt.legend(title='Risk', bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize='small', title_fontsize='medium', labelspacing=0.01, borderaxespad=0)
plt.xticks(rotation=90) 
plt.show()


# In[56]:


Risk_Y = df.loc[df['Risk'] == 0, 'Sector_score']
Risk_N = df.loc[df['Risk'] == 1, 'Sector_score']
plt.hist([Risk_Y, Risk_N], bins=30, stacked=True, alpha=0.7, label=['Risk_Y', 'Risk_N'])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Stacked Histogram')
plt.legend()
plt.show()


# In[57]:


for feature in df.columns:
    if feature != 'Risk':
        sns.boxplot(x='Risk', y=feature, data=df)
        plt.title(f'{feature} by Target Variable')
        plt.show()


# In[58]:


df1=df.drop(columns=['PARA_A', 'LOCATION_ID', 'Score_A', 'PARA_B', 'Score_B', 'TOTAL', 'numbers', 'Score_B.1', 'Score_MV', 'PROB', 'History', 'Prob', 'Score', 'Inherent_Risk', 'CONTROL_RISK', 'Detection_Risk', 'Audit_Risk' ])
sns.pairplot(df1, hue="Risk")


# In[59]:


#correlation matrix below demonstrated strong correlation between following features 
    #Risk_A & PARA_A
    #Risk_B & TOTAL
    #TOTAL  &  PARA_B
    #Score_B.1 & Risk_C
    #Risk_C & Number
    #Risk_B.1 & Number
    #Risk_D & Money Value
    #Risk_E & District_Loss
    #Prob & history
    #History & Risk_F
    #Risk_B and Score

df['LOCATION_ID'] = pd.to_numeric(df['LOCATION_ID'], errors='coerce')
df.fillna(df.mean(), inplace=True)
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), cmap= 'cividis', annot = True, linewidths=0.01)
plt.title('Correlation Matix', fontsize = 20)
plt.show()


# In[69]:


# In order to clean the dataset, non value rows are imputed with number
# correlated features are getting dropped from the dataset to avoid complexity of the algorithm and increasing the risk of errors.

df['LOCATION_ID'] = pd.to_numeric(df['LOCATION_ID'], errors='coerce')
df.fillna(df.mean(), inplace=True)
df2=df.drop(columns=['District_Loss', 'Money_Value', 'PARA_A', 'PARA_B', 'TOTAL', 'Score','Score_B.1', 'numbers' , 'PROB', 'Prob', 'History' ,'Audit_Risk'])
df2.describe()


# In[19]:


# 5 different machine learning algorithms are applied for classification and success metrics such as Confusion Matrix and Area Under the Curve (AUC) used to assess the classification. 

# Decision Tree 
# F1-Score = 0.99
# AUC = 0.9916


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
X = df2.drop('Risk', axis=1)
y = df2['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)


# In[20]:


# Support Vector Machine 
# F1-Score = 0.87
# AUC = 0.825

from sklearn.svm import SVC
model2 = SVC()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)


# In[21]:


# Random Forest 
# F1-Score = 0.99
# AUC = 0.9916

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)


# In[22]:


# Naïve Bayse 
# F1-score = 0.96
# AUC = 0.965


from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)


# In[23]:


# Logistic Regression 
# F1-Score = 0.98
# AUC = 0.9812


from sklearn.linear_model import LogisticRegression
model5 = LogisticRegression()
model5.fit(X_train, y_train)
y_pred = model5.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)

