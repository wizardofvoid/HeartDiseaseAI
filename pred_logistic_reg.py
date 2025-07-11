import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#import the data file
heart_data = pd.read_csv("HearDisease/content/heart.csv")

#To check what data are we working with
heart_data.head()
heart_data.tail()
heart_data.describe()
heart_data['target'].value_counts()

#Dividing the data into train and test model
x=heart_data.drop(columns='target',axis=1)
y = heart_data['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, stratify = y, random_state = 2)

#Defining the AI and training it
model = LogisticRegression()
model.fit(x_train,y_train)

#Testing the Trained AI and determing the accuracy score
x_train_prediction=model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy of training data", training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print(test_data_accuracy)

#If you want to enter the data within the code and test the model follow these steps
#Define the data  that you want to test
# input_data = (62,0,0,140,260,0,0,160,0,3.6,0,2,2)
# input_data_as_numpy_array = np.asarray(input_data)
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction = model.predict(input_data_reshaped)
# if(prediction[0] == 0):
#   print("THe person does not have a hear disease")
# else:
#   print("The person has a heart disease")
