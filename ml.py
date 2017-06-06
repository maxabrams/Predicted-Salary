# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Predicting Household Income From United States Census Data

import os
import numpy as np
import math
import collections
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

#Fake People Test
fakeTest = False
#Load TEST data into X and Y matrix
if(fakeTest):
	fileLoc = 'FakePeople.data'
	numRowTotal = 3
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

scoreArr=[]
printAll = True
deleteArr = []
numColTest = 1#41 - len(deleteArr)

for i in range(numColTest):
	# print "-----------------"
	# print i
	# print "-----------------"
	if deleteArr:
		for val in deleteArr:
			xTrain= np.delete(xTrainOG, val, axis=1)
			xTest= np.delete(xTestOG, val, axis=1)
	else:
		xTrain = xTrainOG
		xTest = xTestOG
	#xTrain= np.delete(xTrainOG, i, axis=1)
	#xTest= np.delete(xTestOG, i, axis=1)

	yTrain=yTrain.ravel()
	yTest= yTest.ravel()
	#Gaussian

	gnb= GaussianNB()
	gnb.fit(xTrain, yTrain)
	gnb_predictions= gnb.predict(xTest)
	if (printAll):
		print "Gaussian Naive-Bayes:"
		print "Train score: ",  gnb.score(xTrain, yTrain)
		print "Test score: ", gnb.score(xTest, yTest)
	#print gnb.theta_, gnb.sigma_

	#Multinomial
	mnb=MultinomialNB()
	mnb.fit(xTrain, yTrain)
	mnb_predictions= mnb.predict(xTest)
	mnbScore = mnb.score(xTest, yTest)
	if (printAll):
		print "Multinomial Naive-Bayes:"
		print "Train score: ", mnb.score(xTrain, yTrain)
		print  "Test score: ", mnbScore

	
	#Bernoulli
	bnb=MultinomialNB()
	bnb.fit(xTrain, yTrain)
	bnb.predictions= bnb.predict(xTest)
	if (printAll):
		print "Bernoulli Naive-Bayes:"
		print "Train score: ", bnb.score(xTrain, yTrain)
		print "Test score: ", bnb.score(xTest, yTest)

	
	#Decision Tree Classifier
	treeClass=  tree.DecisionTreeClassifier()
	treeClass= treeClass.fit(xTrain, yTrain)
	tc_predict=treeClass.predict(xTest)
	treeScore= treeClass.score(xTest, yTest)
	if (printAll):
		print "Decision Tree Classifier:"
		print "Train score: ", treeClass.score(xTrain, yTrain)
		print "Test score: ", treeScore

	scoreArr.append(mnbScore)
#Determine highest scores and associated attributes
mapOut = {}

for i in range(numColTest):
	mapOut[i]=scoreArr[i]

#Orded most important to least important index
print collections.OrderedDict(sorted(mapOut.items(), key=lambda t: t[1]))
# Not working # tree.export_graphviz(treeClass, out_file='tree.dot')  



