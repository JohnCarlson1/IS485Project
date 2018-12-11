
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from sklearn import datasets
from sklearn import tree


# <p> Here I load the data in a dataframe assigned to the variable df1. Then I show the top 5 rows of the dataframe to show what kind of data I will be working with. <p> 

# In[4]:


df1 = pd.read_csv('C:/Users/Jjcca/Desktop/IS485/StudentsPerformance.csv')


# In[5]:


df1.head()


# <h6> Now I'm going to pull the columns that I want from df1 and create a new dataset to remove the data I don't want. <h6>

# In[6]:


parent = df1['parental level of education']
math = df1['math score']
reading = df1['reading score']
writing = df1['writing score']


# In[7]:


df_parent = pd.DataFrame(parent)
df_math = pd.DataFrame(math)
df_reading = pd.DataFrame(reading)
df_writing = pd.DataFrame(writing)
df = pd.concat([df_parent, df_math, df_reading, df_writing], axis=1)
print(df)


# <p>Below we can see that there is 1000 rows of data. The data was taken from Kaggle and has already been cleaned. Math Scores for the 1000 participants carry a mean of 66. Reading scores carry a mean of 69, and writing scores carry a mean of 68.<p>

# In[8]:


df.describe(include='all')


# In[9]:


plt.scatter(df['parental level of education'], df['math score'], marker='.')
plt.xticks(rotation=90)
plt.show()


# In[10]:


plt.scatter(df['reading score'],df['parental level of education'], marker='.')
plt.xlabel('Reading Score')
plt.ylabel('Parent Schooling')
plt.show()


# In[11]:


plt.scatter(df['parental level of education'], df['writing score'], marker='.')
plt.xticks(rotation=90)
plt.show()


# In[12]:


df.head()


# In[13]:


from mpl_toolkits.mplot3d import Axes3D


# In[14]:


fig = plt.figure(1, figsize=(12, 9))
ax = Axes3D(fig, elev=-170, azim=240)

ax.scatter(df['math score'], df['reading score'], df['writing score'], s=20, alpha=.5, c='red')
ax.set_xlabel('math score')
ax.set_ylabel('reading score')
ax.set_zlabel('writing score')
plt.show()
plt.close()
fig.clf()


# <p> After doing some exploratory data analysis we can see that the data closely matches one another. The 2D scatter plots explore the direct correlation between parental level of education and one of the testing categories. I wanted to try the 3D plot to see if the data would separate at all. I want to test if the model can correctly predict the level of schooling the parents have finished. In other words, is there a strong enough correlation between the math, writing, and reading scores of these students, to correctly predict what level of schooling the parents completed. <p> 

# This is a supervised learning classification problem. Try decision tree, then K-Nearest Neighbors

# In[15]:


from sklearn import tree
from sklearn.model_selection import train_test_split


# In[16]:


X = df.values[:,1:4]
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state = 42)
assert len(X_train) + len(X_test) == len(X)


# In[17]:


#initiate decision tree classifier and fit training data
jc = tree.DecisionTreeClassifier()
jc = jc.fit(X_train, y_train)


# In[18]:


from sklearn.metrics import classification_report


# In[19]:


y_prediction = jc.predict(X_test)
g = classification_report(y_test, y_prediction)
print(g)


# <b>Thoughts:<b>

# The decision tree classifier did very poor in classifying the parent's degrees, I will try to tune hyperparameters to increase scores.

# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


knn= KNeighborsClassifier(n_neighbors=7)
knn= knn.fit(X_train, y_train)
y_prediction = knn.predict(X_test)
print(classification_report(y_test, y_prediction))


# K-Neighbors came out with a slightly better average/total, however, some individual sections performed worse than before such as high school. K-Neighbors did a much better job of classifying the master's degree than the decision tree did.

# In[1]:


#try stnadardscaler(), (model) logistic regression, regularization, pipeline, clean up page, also try R^2 (accuracy score)


# Now I am going to try using support vector machines model.

# In[34]:


from sklearn import svm


# In[35]:


df.head()


# In[36]:


#initiate model
svm = svm.SVC()

svm = svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))


# The support vector machines model did not classify any of the Master's degree category. Overall, the k-neighbors model worked the best.

# <b> Conclusion: </b>
# Overall, the dataset used was pretty weak with only 1,000 rows of data. Due to this I could only test on 10% of the data. This most likely gave up accuracy. If I were to do this again I would choose a different data set to test on. The results I received did not show any promise. The results received were nothing to base any future decisions on.
