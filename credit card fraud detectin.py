#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# In[1]:


import sklearn


# In[2]:


import numpy as np
import pandas as pd 
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[3]:


data = pd.read_csv(r"C:\Users\hp\Desktop\projects for business analytics\credit card fraud detection\creditcard.csv") 


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().values.any()


# In[8]:


count_classes = pd.value_counts(data['Class'], sort= True )
count_classes.plot(kind = 'bar' ,rot=0)
plt.title("transaction class destribution ")
plt.xticks(range(2), LABELS)
plt.xlabel("class")
plt.ylabel("frequency")


# In[9]:


Fraud  = data[data['Class'] == 1]
Normal = data[data['Class'] == 0]


# In[10]:


Fraud.shape


# In[11]:


Normal.shape


# In[12]:


Fraud.Amount.describe()


# In[13]:


Normal.Amount.describe()


# In[14]:


Fraud.Amount.plot(kind = 'hist' ,bins = 50)
plt.title('Transaction per Amount by class')
plt.xlabel('Amount $')
plt.ylabel('Number of Transactions')


# In[15]:


Normal.Amount.plot(kind = 'hist' , bins = 50)
plt.title('Transaction per Amount by class')
plt.xlabel('Amount $')
plt.ylabel('Number of Transactions')


# In[16]:


f, (ax1, ax2) = plt.subplots(2, 1,sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(Normal.Time, Normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[17]:


##take sample form the data 
data1 = data.sample(frac = 0.1 , random_state = 1)
data1.shape


# In[18]:


data.shape


# In[19]:


Fraud = data1[data1['Class'] ==1 ]
Valid = data1 [data1['Class'] == 0]
outlier_fraction = len(Fraud)/float(len(Valid))


# In[20]:


print(outlier_fraction)
print("Fraud Cases : ",len(Fraud))

print("Valid Cases : ",len(Valid))


# In[21]:


corrmat = data1.corr()
top_features = corrmat.index
plt.figure(figsize = (25,30))
g=sns.heatmap(data[top_features].corr(),annot=True,cmap="bwr")


# In[22]:


#Create independent and Dependent Features
columns = data1.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[23]:


np.random.seed(42)


# In[24]:


classifier = {
    "Isolation Forest": IsolationForest(n_estimators = 100 , max_samples = len(X),
                                       contamination = outlier_fraction, random_state = state , verbose = 0 ),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors = 0, algorithm = 'auto' ,
                                                 leaf_size = 30, metric = 'minikowski' ),
    "Support Vector Machine" : OneClassSVM(kernel = 'rbf' , degree = 3 , gamma = 0.1 , nu = 0.05,
                                           max_iter = -1 )
    
}


# In[25]:


type(classifier)


# In[26]:


n_outliers = len(Fraud)
for i, (clf_name,clf) in enumerate(classifier.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:    
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y,y_pred))
    print("Classification Report :")
    print(classification_report(Y,y_pred))


# In[ ]:




