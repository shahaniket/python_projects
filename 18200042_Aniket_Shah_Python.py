# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:14:35 2018

@author: aniket
"""

# =============================================================================
# Importing required libraries that to be used in the project
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Importing given dataset required for implementing GA algorithm
# =============================================================================
data = pd.read_csv('Project_1.csv')
data = data.iloc[:,:-2]
print(data.head(10))        # Printing 10 records

# =============================================================================
# STEP 2
# Initialising the number of functions (equations) to 10
# =============================================================================
numOfFunctions = 10

# =============================================================================
# STEP 3
# Filtering the dataset by selecting first 5 columns as predictors and last 
# column as target column
# =============================================================================
X = data.iloc[:,:5]
Y = data.iloc[:,-1]

# =============================================================================
# STEP 4
# Choosing 25% of the dataset (random) as testing and the rest 75% as training 
# samples
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=1)

# =============================================================================
# STEP 5
# Normalising the training dataset with values between 0 and 1
# =============================================================================
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)

# =============================================================================
# Generating population matrix with random values of 10xPxN bits
# =============================================================================
def generatePopulationMatrix():
    populationMatrix = []
    Npop = 500
    P = 5
    for p in range(Npop):
        w = np.random.uniform(-1,1,(P,numOfFunctions))
        populationMatrix.append(w)
    return populationMatrix

populationMatrix = generatePopulationMatrix()

# =============================================================================
# STEP 7
# Function for generating YHat values, derived from equation 1
# =============================================================================
def generateYHat(populationMatrix):
    YHatPartialList = []
    for i in range(len(populationMatrix)):
        singleWeight = np.array(populationMatrix[i]).reshape(10,5)
        for j in range(len(X_train)):
            dotProduct = singleWeight.dot(X_train[j])
            partialYHatList = []
            for y in range(len(dotProduct)):
                partialYHatList.append((1/(1+np.exp(-dotProduct[y]))))
            YHatPartialList.append(sum(partialYHatList))   
            
    YHatList = np.array_split(YHatPartialList, len(populationMatrix))
    YHatList = [i.tolist() for i in YHatList]
    return YHatList

YHatList = generateYHat(populationMatrix)

# =============================================================================
# STEP 8
# Function for calculating the fitness_values via Eq(2) for each solution
# =============================================================================
def generateFitnessValue(populationMatrix, Y_train, YHatList):
    fitnessValueList = []
    normalizedBodyFat = (Y_train - Y_train.mean()) / (Y_train.max() - Y_train.min())
    for i in range(len(populationMatrix)):
        sampleList = []
        for j in range(len(Y_train)):
            sampleList.append((YHatList[i][j]-normalizedBodyFat.iloc[j]) ** 2)
        fitnessValueList.append((1-(sum(sampleList)/len(Y_train)))*100)
    return fitnessValueList

fitnessValueList = generateFitnessValue(populationMatrix, Y_train, YHatList)

# =============================================================================
# STEP 9
# Function for selecting the highest fitness value and getting the index of that
# value
# =============================================================================
def generateSire(fitnessValueList):
    sire = fitnessValueList[fitnessValueList.index(max(fitnessValueList))]
    return sire

sire = generateSire(fitnessValueList)

def generateParentIndex(fitnessValueList):
    return fitnessValueList.index(max(fitnessValueList))

parentIndex = generateParentIndex(fitnessValueList)

# =============================================================================
# STEP 10
# Function to get the binarized value of all the elements of population matrix
# Normalising all the values and converting it to binary form 
# each binary weight have fix 10 bits allocated
# =============================================================================
def generateBinarization(binarizePopulationMatrix):
    for i in range(len(binarizePopulationMatrix)):
        binarizePopulationMatrix[i] = binarizePopulationMatrix[i].tolist()
        for j in range(len(binarizePopulationMatrix[i])):      
            xmin = min(binarizePopulationMatrix[i][j]) 
            xmax = max(binarizePopulationMatrix[i][j])
            for k in range(len(binarizePopulationMatrix[i][j])):
                binarizePopulationMatrix[i][j][k] = ((binarizePopulationMatrix[i][j][k] - xmin) / (xmax - xmin))
                binarizePopulationMatrix[i][j][k] = (binarizePopulationMatrix[i][j][k] * 1000)
                binarizePopulationMatrix[i][j][k] = (bin(int(binarizePopulationMatrix[i][j][k]))[2:].zfill(10))
    return binarizePopulationMatrix

binarizePopulationMatrix = generateBinarization(populationMatrix)
parent = binarizePopulationMatrix[parentIndex]

# =============================================================================
# STEP 11
# Concatenating all 10-bit weights along each other and make a chromosome
# Function to generate parent chromosome
# =============================================================================
def generateChromosome(chromosomeMatrix):
    for i in range(len(chromosomeMatrix)):
        chromosome = []
        singleChromosome = []
        for j in range(len(chromosomeMatrix[i])):
            chromosome.append(''.join(chromosomeMatrix[i][j]))
        singleChromosome = ''.join(chromosome)
        chromosomeMatrix[i] = singleChromosome
    return chromosomeMatrix

chromosomeMatrix = generateChromosome(binarizePopulationMatrix)
   
# Parent chromosome
def generateParentChromosome(parent):
    parentChromosome = []
    chromosome = []
    for i in range(len(parent)):
        chromosome.append(''.join(parent[i]))
    parentChromosome = ''.join(chromosome)
    return parentChromosome

parentChromosome = generateParentChromosome(parent)

def reshapeMatrix(matrix):
    for i in range(len(matrix)):
        chromosome = []
        singleChromosome = []
        for j in range(len(matrix[i])):
            chromosome.append(''.join(matrix[i][j]))
        singleChromosome = ''.join(chromosome)
        matrix[i] = singleChromosome
    return matrix

# =============================================================================
# STEP 12
# Function to generate CrossOver to create offspring.
# the offspring takes one section of the chromosome from each parent.
# =============================================================================
def generateCrossOverMatrix(chromosomeMatrix, parentChromosome):
    crossOverMatrix = []
    for i in range(len(chromosomeMatrix)):
        C_Point = np.random.randint(2,499)
        parentFirstChromosome = parentChromosome[:C_Point]
        parentSecondChromosome = parentChromosome[C_Point:]
        popFirstChromosome = chromosomeMatrix[i][:C_Point]
        popSecondChromosome = chromosomeMatrix[i][C_Point:]
        firstOff = parentFirstChromosome + popSecondChromosome
        secondOff = parentSecondChromosome + popFirstChromosome
        crossOverMatrix.append(firstOff)
        crossOverMatrix.append(secondOff)
    return crossOverMatrix

crossOverMatrix = generateCrossOverMatrix(chromosomeMatrix, parentChromosome)

# =============================================================================
# STEP 13
# Function to generate Mutation for 5% of the bits
# =============================================================================
def generateMutation(crossOverMatrix):
    for i in range(len(crossOverMatrix)):
        randList = random.sample(range(0, 500), 25)
        chromosomeList = list(crossOverMatrix[i])
        for j in range(len(randList)):
            if chromosomeList[randList[j]] == '0':
                chromosomeList[randList[j]] = '1'
            else:
                chromosomeList[randList[j]] = '0'
        crossOverMatrix[i] = ''.join(chromosomeList)
    return crossOverMatrix

crossOverMatrix = generateMutation(crossOverMatrix)

# =============================================================================
# STEP 14
# function to de-binarize the chromosomes and getting values between -1 and 1
# =============================================================================
def generateDebinarization(crossOverMatrix):
    debinarizeMatrix = []
    for c in range(1000):
        debinarizeList = [crossOverMatrix[c][i:i+10] for i in range(0, len(crossOverMatrix[c]), 10)]
        debinarizeList = [int(b,2) for b in debinarizeList]
        debinarizeList = [(b/1000) for b in debinarizeList]
        debinarizeList = [((b*2)-1) for b in debinarizeList]
        debinarizeMatrix.append(debinarizeList)
    return debinarizeMatrix

debinarizeMatrix = generateDebinarization(crossOverMatrix)

# =============================================================================
# STEP 15
# function to alculate the fitness_value for all population from Eq(2)
# =============================================================================
def generateNewFitnessValue(debinarizeMatrix, parent, sire):
    newYHatList = []
    newPopulationMatrix = []
    for n in range(1000):
        newPopulationMatrix.append((np.array(debinarizeMatrix[n]).reshape(5,10)))
    
    newYHatList = generateYHat(newPopulationMatrix)
    newFitnessValueList = generateFitnessValue(newPopulationMatrix, Y_train, newYHatList)
    
    # STEP 16
    sortedFitnessList = sorted(newFitnessValueList)
    sortedFitnessList = sortedFitnessList[:500]
    newParentIndex = sortedFitnessList.index(max(sortedFitnessList))
    newBinarizePopulationMatrix = generateBinarization(newPopulationMatrix)
    newParent = newBinarizePopulationMatrix[newParentIndex]
    newSire = newFitnessValueList[newParentIndex]
    # STEP 17
    if (sire < newSire):
        parent = newParent
        sire = newSire
    return [newPopulationMatrix, newYHatList, parent, sire, newYHatList]

newFitnessValue = generateNewFitnessValue(debinarizeMatrix, parent, sire)

# =============================================================================
# STEP 18
# recalculating all steps until the highest fitness_value reaches to a plateau
# =============================================================================
xaxis = []
yaxis = []
iteration = 1

while iteration < 18:
    xaxis.append(iteration)
    yaxis.append(newFitnessValue[3])
    plt.scatter(xaxis,yaxis)
    plt.show()
    print('Iteration Count:', iteration)
    iteration = iteration + 1
    
    fitnessValueList = generateFitnessValue(newFitnessValue[0], Y_train, newFitnessValue[1])
    sire = generateSire(fitnessValueList)
    parentIndex = generateParentIndex(fitnessValueList)
    binarizePopulationMatrix = reshapeMatrix(newFitnessValue[0])
    parent = binarizePopulationMatrix[parentIndex]
    chromosomeMatrix = generateChromosome(binarizePopulationMatrix)
    parentChromosome = generateParentChromosome(parent)
    crossOverMatrix = generateCrossOverMatrix(chromosomeMatrix, parentChromosome)
    debinarizeMatrix = generateDebinarization(crossOverMatrix)
    newFitnessValue = generateNewFitnessValue(debinarizeMatrix, parent, sire)

# =============================================================================
# Error Calculation
# Calculating NewYHatValues along with newfitness values for test set
# and utilising it for calculating MSE
# =============================================================================
Y_test = Y_test.tolist()
errorList = []
for i in range(len(Y_test)):
    meanYHat = sum(YHatList[i])/len(YHatList)
    se = (meanYHat - Y_test[i]) ** 2
    errorList.append(se)
error = sum(errorList) / len(Y_test*100)
print("overall error for testing dataset: ", error)

# =============================================================================
# 3d Scatter Plot
# Generating 3d plot for test dataset and YHatvalues
# =============================================================================
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_test['Weight lbs'], X_test['Height inch'], Y_test,c='r',label='y')
ax.scatter(X_test['Weight lbs'], X_test['Height inch'], YHatList[:63],c='b',label='yhat')
ax.set_xlabel('Weights lbs')
ax.set_ylabel('Height inch')
ax.set_zlabel('Y and Yhat')
ax.legend()
ax.grid(True)
plt.show()

