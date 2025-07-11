import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

heart_disease = pd.read_csv('./content/heart-disease.csv')
x = heart_disease.drop('target', axis=1)
y = heart_disease['target']

clf = RandomForestClassifier()

x_train,x_test,y_train,y_split = train_test_split(x,y, test_size = 0.2)
clf.fit(x_train, y_train)

#TO KNOW THE SCORE OF THE MODEL
# clf.score(x_test, y_split)

# FOR IMPORTING THE MODEL
# import pickle
# pickle.dump(clf, open("heart_disease_predictor.pkl", "wb"))
