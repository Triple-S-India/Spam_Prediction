#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Spam Prediction using NLP


# In[2]:


# First we have to import some basic libreries which we always need


# In[3]:


#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# Now we will Import our given Dataset
dataset = pd.read_csv("D:AI_ML/Theory/ML/spam.csv", encoding='latin_1')   # I have spam.csv file as a given file 


# In[5]:


dataset.head()


# In[6]:


# Find shape of dataset
dataset.shape


# In[7]:


# Find NaN if present in our dataset
dataset.isnull().sum()


# In[8]:


# So, here we see that except 1st and 2nd column remaining columns have almost all the values as NaN,
# So we can drop these columns
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset.head()


# In[9]:


# So now we have a correct dataset for further work


# In[10]:


# We can also change column names for better understanding
dataset = dataset.rename(columns = {'v1':'label','v2':'text'})


# In[11]:


dataset.head()


# In[12]:


#Count observations in each label
dataset.label.value_counts()


# In[13]:


# convert label to a numerical variable
dataset['label_num'] = dataset.label.map({'ham':1, 'spam':0})


# In[14]:


dataset.head()


# In[15]:


# We are cheching total no of words in each row
dataset['length'] = dataset['text'].apply(len)
dataset.head()


# In[16]:


import seaborn as sns
sns.countplot(dataset["label"])
plt.show()


# In[17]:


# Cleaning the texts
# Import required liabraries to clean data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer        #Stemming


# In[18]:


corpus = []
for i in range(0, 5572):
    textt = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    textt = textt.lower()
    textt = textt.split()
    ps = PorterStemmer()
    textt = [ps.stem(word) for word in textt if not word in set(stopwords.words('english'))]
    textt = ' '.join(textt)
    corpus.append(textt)


# In[19]:


corpus[1:10]


# In[20]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 2].values


# In[21]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[22]:


X_train[0:5],y_train[0:5]


# In[23]:


# Fitting Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000)     #n_estimator means no of trees
classifier.fit(X_train, y_train)


# In[24]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred


# In[25]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[26]:


TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP,TN,FP,FN


# In[27]:


Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy


# In[28]:


Precision = TP / (TP + FP)
Precision


# In[29]:


Recall = TP / (TP + FN)
Recall


# In[30]:


F1_Score = 2 * Precision * Recall / (Precision + Recall)
F1_Score


# In[31]:


# I have used here Random_Forest_Classifier. You can also use other classifiers like Naive_Bayes, Decision Tree etc.. 
# Check the result from every model and choose best one.    """ THANKS! """


# In[ ]:




