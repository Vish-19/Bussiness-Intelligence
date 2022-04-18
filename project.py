import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("churn.csv")
shape = data.shape #gets the number of columns and rows which are 14 and 10000
stats = data.describe() #gets statistical information about the data set

# DATA CLEANSING

null_check = data.isnull().sum() # gives the number of null values in each column and data.isnull() is data set which replaces null values with true and not null values with false
#if we had null values we we replace it with 0 usually or we drop those rows in some cases we replace these with mean values and etc...
#as our data set has 0 null values we dont have to do any of these operations

#now we need to drop those rows which do no good to the prediction process
columns = data.columns #we realize that Rownumber, CustomerId and Surname do no good for us in the prediction so we will have to drop those columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1) # we specify the axis as 1
#print(data)
#print(data.columns)
# now we have successfully dropped useless columns and are data set is ready

#DATA ENCODING

#there are some columns which are Categorical data(String data) which should be encoded to numeric data
gender = data['Gender'].unique()
#print(gender)
#for example in gender Female has to be encoded to 0 and Male to 1
country = data['Geography'].unique()
#print(country)
#And here France has to be encoded to 0 and Spain to 1 and Germany to 3 but not necessarily in any order
# this is called label encoding but the problem with these is the computer can also compare these label encoded variables which might not make sense

#label encoding can be useful in emotion analysis where words have ranks(ordinal) but not here we use another method called one hot encoding is used
#here the labels to be encoded are added as columns and if in the column to be encoded has the the row equal to the column name the column value for that particular row is given as 1
#for example
#Geography    F G S
#F            1 0 0
#G            0 1 0
#S            0 0 1
#these oder of binary distribution is not fixed it may vary
#these are called dummy values
#to achieve this we use
data = pd.get_dummies(data)
#print(data.columns)

#PREPARING OUR TRAINING DATA SET

X = data.drop(['Exited'],axis=1) #independant variable
Y = data['Exited'] #dependant variables
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
#this method is used to randomly split the data set into testing and training data sets

#HANDLING IMBALANCED DATA
#print(Y.value_counts())
X, Y = SMOTE().fit_resample(X,Y) #performs over sampling
#print(Y.value_counts())
Xtrain = StandardScaler().fit_transform(Xtrain)
Xtest = StandardScaler().fit_transform(Xtest)
#here z-score normalization is used, we can also do it using numpy but StandardScalar() is a faster method and is easier and less amount of work as using numpy first we'll have to convert each array in the 2-D list to numpy array and then perform normalization
#here scaling is important as different scales and when performing linear machine learning algorithms like knn and other algorithms if the scale is not the same the out liers the clusters and all other factors may differ
# for example CreditScore is int the scale above 600 whereas age is below 100 and therefore we need scaling in this case

#FITTING MODEL AND PREDICTING
knn = KNeighborsClassifier()
knn.fit(Xtrain,Ytrain)
Ypredicted1 = knn.predict(Xtest)

#CHECKING ACCURACY
#print("KNN:")
#print(accuracy_score(Ytest,Ypredicted1))
#print(precision_score(Ytest,Ypredicted1))
#print(recall_score(Ytest,Ypredicted1))
#print(f1_score(Ytest,Ypredicted1))
#as our dataset is imbalanced assessing our model based on accuracy isn't the correct way hence we also check for precision score recall score and
#now if we observe the data we have used might have a low scores this might be because of the actual effect of the algorithm used on the data set or probably because the data is imbalanced so make sure that your data is balanced
#for this you can use SMOTE
#this imbalance can be over come by over sampling which means replacing majority data with minority ones
#Or we can use undersampling by removing the majority data but this might lead to loss of data therefore it is better to use oversampling
#if you ahndle the imbalance in your data set we might see a big change in the outcome

#We will try pridicting using random forrest and decision tree

#DECISION TREE
dt = DecisionTreeClassifier()
dt.fit(Xtrain,Ytrain)
Ypredicted2 = dt.predict(Xtest)

#RANDOM FOREST
rf = RandomForestClassifier()
rf.fit(Xtrain,Ytrain)
Ypredicted3 = rf.predict(Xtest)

#Checking accuracy for DECISION TREE
#print("DECISION TREE:")
#print(accuracy_score(Ytest,Ypredicted2))
#print(precision_score(Ytest,Ypredicted2))
#print(recall_score(Ytest,Ypredicted2))
#print(f1_score(Ytest,Ypredicted2))

#Checking accuracy for RANDOM FOREST
#print("RANDOM FOREST:")
#print(accuracy_score(Ytest,Ypredicted3))
#print(precision_score(Ytest,Ypredicted3))
#print(recall_score(Ytest,Ypredicted3))
#print(f1_score(Ytest,Ypredicted3))

#from this we can conclude that random forest has the maximum score in precision accuracy recall and f1
#therefore we can conclude that this is the best prediction method
#we can also try for XG Boost , svm , Logistic regression , Linear regression and etc

#As we are ready with our prediction algorithm its time to deploy our model

#MODEL DEPLOY
X_final = StandardScaler().fit_transform(X) #we transform the whole training set
rf.fit(X_final,Y) #we don't need to scale Y as Y is already in 0's and 1's
joblib.dump(rf,"BANK_CHURN") #this will deploy the model in the name
model_churn = joblib.load("BANK_CHURN")
print(X.columns)
#while prdicting using the model pass it as a 2D array as the training dataset passed in is a 2D data set
result1 = model_churn.predict([[300,42,2,0.0,0,0,0,10143.60,1,0,0,0,1]]) #Customer 1
result2 = model_churn.predict([[608,42,1,83807.86,1,0,1,112542.6,0,0,1,1,0]]) #Customer 2
l = []
l.append(accuracy_score(Ytest,Ypredicted3)*100)
l.append((1 - accuracy_score(Ytest,Ypredicted3))*100)
xpos = np.arange(len(l))
if(result2 == [1]):
    plt.bar(xpos, l, color=['red', 'green'])
    plt.xticks(xpos, ["Churn probability","retension probability"])
else:
    plt.bar(xpos, l, color=['green', 'red'])
    plt.xticks(xpos, ["Retension probability", "Churn probability"])
plt.show()