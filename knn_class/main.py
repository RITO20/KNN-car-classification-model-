import numpy as np 
import pandas as pd 
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

car_data = pd.read_csv('car.data')

x = car_data[[
    'buying',
    'maint',
    'safety'
]].values
y = car_data[['class']]


#converting the data 
#x
le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i]=le.fit_transform(x[:,i])

#y
label_maping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

#mapiing the data 
y['class']=y['class'].map(label_maping)
y = np.array(y)


#create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25 , weights='uniform')

# spliting training data & testign data 
# tese_size=0.2 means 20% of out main data is going to be used for testing 
# the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#traing
knn.fit(x_train,y_train)

#test 
predict = knn.predict(x_test)

#calculate accuracy 
accuracy = metrics.accuracy_score(y_test,predict)

print("predict :", predict)
print("accuracy :", accuracy)

a = int(input("enter a value from 0 to 1727 :"))
print("every number from 0 to 1727 retresents a car")
print("actual value: ", y[a])
print("predicted value: ", knn.predict(x)[a])
print("0 = nuacceptable")
print("1 = acceptable")
print("2 = good")
print("3 = very good")