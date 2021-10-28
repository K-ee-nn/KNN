import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
#---------------------------------------------------------------
# Reading in Data
#---------------------------------------------------------------
data = pd.read_csv('car.data')
#print(data.head())


#-------------------------------------------------------
# transform our non-numerical data into numerical data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
#---------------------------------------------------------

predict = 'class' # Predictive task

X = list(zip(buying, maint, door, persons, lug_boot, safety)) # zip converts everything into one big list
y = list(cls) # Predictive task

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# test X and y based on x_train, x_test, y_train, y_test

model = KNeighborsClassifier(n_neighbors=7) # Amount of neighbors
model.fit(x_train, y_train) # train
acc = model.score(x_test, y_test) # accuracy
print(acc)


predicted = model.predict(x_test) # predicted model
names = ['unacc', 'acc', 'good', 'vgood']# acc in word form
for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], 'Data: ', x_test[x], 'Actual: ', names[y_test[x]])# replace acc with words
    n = model.kneighbors([x_test[x]], 7, True)
    print("N:", n)
