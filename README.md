# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the dataset .print the head and tail of dataset.
3. import train test split.
4. Import countervector and proceed with the detection of spam mail. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.Sham Rathan 
Register.No : 212221230093  
*/
import chardet 
file='spam.csv'
with open(file,'rb') as rawdata:
  result=chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/c70c6171-ef60-45da-bf97-79b1e914bbb7)

![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/66112a53-59ac-4501-8e72-27b1ff0b98b4)

![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/0c52d190-237c-4066-b9d2-e383ccac91c6)

![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/c57827dd-3e52-4e7c-8c4f-70e5daebb5e9)

![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/9a3ce210-8b48-43bf-ac63-276fa2e1ba57)

![image](https://github.com/ShamRathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93587823/24db9023-9a38-4122-9e11-340f89d6ae73)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
