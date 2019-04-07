# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 02:13:32 2018

@author: aniket
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


LEARNING_RATE = 0.1        # Change the learning rate
EPOCHS_COUNT = 1000        # Change the epochs count

#Setting the credentials for plotly to use Plotly Online Dashboard to display graph
config = {
    'username' : 'aniket.shah',
    'api_key' : '5hWdu5vZY0SrdUFWmJxY'
}
plotly.tools.set_credentials_file(username=config['username'], api_key=config['api_key'])

#Loading Dataset
trainData = np.loadtxt("Google_Stock_Price_Train.csv", delimiter=",")

#Setting Predictors and Target Values
features_data = trainData[:,0:4]
target_data = trainData[:,4]

# Plotting the graph
xd=trainData[:,2]
yd=trainData[:,4]
layout = dict(title = 'Average Open and Volumes of Shares',
              xaxis = dict(title = 'Open'),
              yaxis = dict(title = 'Volume'),
              )
trace = go.Scatter(
    x = xd,
    y = yd,
    mode = 'markers'
)
data = [trace]
py.plot(data, layout)

#Tuning Predictors with the Pytorch datatype
predictors = torch.from_numpy(np.array(features_data)).float()
predictors = Variable(predictors)

outputData = torch.from_numpy(np.array(target_data)).float()
outputData = Variable(outputData)

#Perfomring Linear Regression on the training dataset
linearRegression = nn.Linear(4, 1)

criterion = nn.MSELoss()
optimizer = optim.Adam(linearRegression.parameters(), lr=LEARNING_RATE)

#Running the loop for 1000 times to findout the best accuracy result
for ep in range(EPOCHS_COUNT):
    linearRegression.zero_grad()
    data_output = linearRegression(predictors)
    loss = criterion(data_output.squeeze(0), outputData.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print('epoch {}'.format(ep))

#Using CPU
data_output = data_output.data.cpu().numpy()
outputData = np.array(target_data)

#Plotting the output of Linear Regression
lineChart = plt.figure()
plt.plot(outputData, data_output, "o")
plt.plot([outputData.min(), outputData.max()], [outputData.min(), outputData.max()], 'k--', lw=3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title("Measured Stock Price vs Predicted Stock Prices")
plotly_fig = tls.mpl_to_plotly(lineChart)
plot_url = py.plot(plotly_fig)

