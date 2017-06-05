# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Predicting Household Income From United States Census Data

import os
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree

#Helper function for mapping
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#Read Load Data
fileLoc = 'census-income.data'
numRowTotal = 199523
numCol = 42

#Build dictionary value mapings
dataFile = open(fileLoc, 'r')
mapVals = [dict() for x in range(numCol)]

for line in dataFile:
	line = line.strip()
	line = line.split(',')
	for i in range(0, len(line)):
		currentDict = mapVals[i]
		if(is_number(line[i].strip()) == False): #Only map strings, leave numbers alone
			if not line[i].strip() in currentDict:
				currentDict[line[i].strip()] = len(currentDict)

dataFile.close()

#Load training data into X and Y matrix
xTrainOG = np.zeros((numRowTotal, numCol-1))
yTrain = np.zeros((numRowTotal,1))
dataFile = open(fileLoc, 'r')
loadIndex = 0
for line in dataFile:
	line = line.strip()
	line = line.split(',')
	lineIndex = 0
	#Load row values for X matrix
	while(lineIndex < len(line)-1):
		rawVal = line[lineIndex].strip() #Retreive mapping
		if not rawVal in mapVals[lineIndex]:
			#Must have been a float previously, use float value
			xTrainOG[loadIndex][lineIndex] = float(rawVal)
		else:
			#Has an associated string-value mapping
			mappedVal = mapVals[lineIndex][rawVal]
			xTrainOG[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTrain[loadIndex] = mappedVal
	loadIndex +=1

dataFile.close()

#Load TEST data into X and Y matrix
fileLoc = 'census-income.test'
numRowTotal = 99762
xTestOG = np.zeros((numRowTotal, numCol-1))
yTest = np.zeros((numRowTotal,1))
dataFile = open(fileLoc, 'r')
loadIndex = 0
for line in dataFile:
	line = line.strip()
	line = line.split(',')
	lineIndex = 0
	#Load row values for X matrix
	while(lineIndex < len(line)-1):
		rawVal = line[lineIndex].strip() #Retreive mapping
		if not rawVal in mapVals[lineIndex]:
			xTestOG[loadIndex][lineIndex] = float(rawVal)
		else:
			mappedVal = mapVals[lineIndex][rawVal]
			xTestOG[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTest[loadIndex] = mappedVal
	loadIndex +=1

dataFile.close()

treeArray=[]
for i in range(41):
	# print "-----------------"
	# print i
	# print "-----------------"
	xTrain= np.delete(xTrainOG, i, axis=1)
	xTest= np.delete(xTestOG, i, axis=1)


	yTrain=yTrain.ravel()
	yTest= yTest.ravel()
	#Gaussian
	# print "Gaussian Naive-Bayes:"
	gnb= GaussianNB()
	gnb.fit(xTrain, yTrain)
	gnb_predictions= gnb.predict(xTest)
	# print "Train score: ",  gnb.score(xTrain, yTrain)
	# print "Test score: ", gnb.score(xTest, yTest)
	# print gnb.theta_, gnb.sigma_

	# print "Multinomial Naive-Bayes:"
	mnb=MultinomialNB()
	mnb.fit(xTrain, yTrain)
	mnb_predictions= mnb.predict(xTest)
	# print "Train score: ", mnb.score(xTrain, yTrain)
	# print  "Test score: ", mnb.score(xTest, yTest)

	# print "Bernoulli Naive-Bayes:"
	bnb=MultinomialNB()
	bnb.fit(xTrain, yTrain)
	bnb.predictions= bnb.predict(xTest)
	# print "Train score: ", bnb.score(xTrain, yTrain)
	# print "Test score: ", bnb.score(xTest, yTest)

	# print "Decision Tree Classifier:"
	treeClass=  tree.DecisionTreeClassifier()
	treeClass= treeClass.fit(xTrain, yTrain)
	tc_predict=treeClass.predict(xTest)
	treeScore= treeClass.score(xTest, yTest)
	# print "Train score: ", treeClass.score(xTrain, yTrain)
	# print "Test score: ", treeScore

	treeArray.append(treeScore)

max=0
index=0

for i in range(41):
	if treeArray[i]>max:
		max=treeArray[i]
		index=i

print max, " index: ", index