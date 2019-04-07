import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper, PrintfTickFormatter, FactorRange
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform
from bokeh.layouts import gridplot

#Data Loading
dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.head(10))
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Dimensions of input data
dataset.shape
X.shape
Y.shape

#Removing null values
dataset.dropna()

#Data Pre-Processing 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# EDA

#Corelation
corelation = dataset.corr()

test = dataset.groupby(['Geography'])['RowNumber'].agg('count')
factor = test.index.values.tolist()
values = list(test[0:])

p = figure(plot_width=600, plot_height=600)
p.vbar(x=[1,2,3], width=0.5, bottom=0,top=values, color="firebrick")
p.xaxis.axis_label = "Countries"
p.xaxis.major_label_overrides = {1: 'France', 2: 'Germany', 3: 'Spain', 1.5:'',2.5:''}
p.title.text = "Bar Chart for number of customer in specific countries"
show(p)

test = dataset.groupby(['Geography'])['EstimatedSalary'].agg('mean')
factor = test.index.values.tolist()
values = list(test[0:])
values = [i/10000 for i in values]

p = figure(plot_width=600, plot_height=600)
p.line([1,2,3],values, line_width=3)
p.circle([1,2,3],values, size=15, fill_color="orange", line_color="green", line_width=3)
p.xaxis.major_label_overrides = {1: 'France', 2: 'Germany', 3: 'Spain', 1.5:'',2.5:''}
p.xaxis.axis_label = "Countries"
p.yaxis.axis_label = "Salary(1 unit = 10000$)"
p.title.text = "Line Chart for Estimated Salry of customers in specific countries"
show(p)

#list(test.loc[test['gender'] == 'Male']['count'])

test = pd.DataFrame()
test['count'] = dataset.groupby(['Geography', 'Gender'])['RowNumber'].agg('count')
test['country'] = [i[0] for i in test.index.values.tolist()]
test['gender'] = [i[1] for i in test.index.values.tolist()]

country = ['France', 'Germany', 'Spain']
gender = ["Male", "Female"]
colors = ["#c9d9d3", "#718dbf"]
dataBokeh = {'country' : country,
        'Male'   : list(test.loc[test['gender'] == 'Male']['count']),
        'Female' : list(test.loc[test['gender'] == 'Female']['count'])}
p = figure(x_range=country, plot_height=250, title="Gender Counts by Country",
           toolbar_location=None, tools="")

p.vbar_stack(gender, x='country', width=0.9, color=colors, source=dataBokeh,
             legend=["male", "female"])
p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
#show(p)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier = Sequential()

#input layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim = 11))

#output from input layer, input to hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))

#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#fitting ANN
classifier.fit(X_train, Y_train, batch_size = 32, nb_epoch = 100)

#prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test,y_pred)
acc = accuracy_score(Y_test,y_pred)

print ("Accuracy of the model is " + str(acc*100) + "%")

