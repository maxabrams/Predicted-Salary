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
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

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
ages = [0] * 100
for line in dataFile:
	line = line.strip()
	line = line.split(',')
	for i in range(0, len(line)):
		if(i==0):
			ages[int(line[i])] += 1
		currentDict = mapVals[i]
		if(is_number(line[i].strip()) == False): #Only map strings, leave numbers alone
			if not line[i].strip() in currentDict:
				currentDict[line[i].strip()] = len(currentDict)

printi = 0
while(printi < 100):
	print str(printi),',',str(ages[printi])
	printi += 1
dataFile.close()

#Load training data into X and Y matrix
xTrainOG = np.zeros((numRowTotal, numCol-1))
yTrain = np.zeros((numRowTotal,1))
dataFile = open(fileLoc, 'r')
loadIndex = 0


xTrainMatrix = [] #Use array for vectorization
yTrainMatrix = []
for line in dataFile:
	line = line.strip()
	line = line.split(',')
	lineIndex = 0
	lineArr = {}
	#Load row values for X matrix
	while(lineIndex < len(line)-1):
		rawVal = line[lineIndex].strip() #Retreive mapping
		if is_number(rawVal):
			lineArr[str(lineIndex)] = float(rawVal)
			#addItem = {str(lineIndex):float(rawVal)}
		else:
			lineArr[str(lineIndex)] = rawVal
			#addItem = {str(lineIndex):rawVal}
		#lineArr += addItem
		if not rawVal in mapVals[lineIndex]:
			#Must have been a float previously, use float value
			xTrainOG[loadIndex][lineIndex] = float(rawVal)
		else:
			#Has an associated string-value mapping
			mappedVal = mapVals[lineIndex][rawVal]
			xTrainOG[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	xTrainMatrix.append(lineArr)
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTrainMatrix.append(mappedVal)
	yTrain[loadIndex] = mappedVal
	loadIndex +=1

#Convert string values to their own coloums with vectorization
#print XTrainMatrix
xVec = DictVectorizer()
xTrainMatrix = xVec.fit_transform(xTrainMatrix).toarray()
#print XTrainMatrix
#print xVec.get_feature_names()
dataFile.close()
xTrainOG = xTrainMatrix

#Load TEST data into X and Y matrix
fileLoc = 'census-income.test'
numRowTotal = 99762
xTestOG = np.zeros((numRowTotal, numCol-1))
yTest = np.zeros((numRowTotal,1))
dataFile = open(fileLoc, 'r')
loadIndex = 0
xTestMatrix = [] #Use array for vectorization
yTestMatrix = []
for line in dataFile:
	line = line.strip()
	line = line.split(',')
	lineIndex = 0
	lineArr = {}
	#Load row values for X matrix
	while(lineIndex < len(line)-1):
		rawVal = line[lineIndex].strip() #Retreive mapping
		if is_number(rawVal):
			lineArr[str(lineIndex)] = float(rawVal)
			#addItem = {str(lineIndex):float(rawVal)}
		else:
			lineArr[str(lineIndex)] = rawVal
		if not rawVal in mapVals[lineIndex]:
			xTestOG[loadIndex][lineIndex] = float(rawVal)
		else:
			mappedVal = mapVals[lineIndex][rawVal]
			xTestOG[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	xTestMatrix.append(lineArr)
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTestMatrix.append(mappedVal)
	yTest[loadIndex] = mappedVal
	loadIndex +=1

xTestMatrix = xVec.transform(xTestMatrix).toarray()
dataFile.close()
xTestOG = xTestMatrix

print str(len(xTrainMatrix))
print str(len(yTrainMatrix))
print str(len(xTestMatrix))
print str(len(yTestMatrix))

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

	#Convert to NP array
	xTrainMatrix = np.array(xTrainMatrix)
	yTrainMatrix = np.array(yTrainMatrix)
	xTestMatrix = np.array(xTestMatrix)
	yTestMatrix = np.array(yTestMatrix)


	logreg=LogisticRegression()
	logreg.fit(xTrainMatrix, yTrainMatrix)
	if (printAll):
		print "Logistic Regression:"
		print "RMSE Train: ", mean_squared_error(yTrainMatrix, logreg.predict(xTrainMatrix))
		print "RMSE Test: ", mean_squared_error(yTestMatrix, logreg.predict(xTestMatrix))
		print "Train score: ",  logreg.score(xTrainMatrix, yTrainMatrix)
		print "Test score: ", logreg.score(xTestMatrix, yTestMatrix)



	#Gaussian
	gnb= GaussianNB()
	gnb.fit(xTrainMatrix, yTrainMatrix)
	#gnb_predictions= gnb.predict(xTestMatrix)
	if (printAll):
		print "Gaussian Naive-Bayes:"
		print "Train score: ",  gnb.score(xTrainMatrix, yTrainMatrix)
		print "RMSE Train: ", mean_squared_error(yTrainMatrix, gnb.predict(xTrainMatrix))
		print "Test score: ", gnb.score(xTestMatrix, yTestMatrix)
		print "RMSE Test: ", mean_squared_error(yTestMatrix, gnb.predict(xTestMatrix))

	#print gnb.theta_, gnb.sigma_

	#Multinomial
	mnb=MultinomialNB()
	mnb.fit(xTrainMatrix, yTrainMatrix)
	#mnb_predictions= mnb.predict(xTestMatrix)
	mnbScore = mnb.score(xTestMatrix, yTestMatrix)
	if (printAll):
		print "Multinomial Naive-Bayes:"
		print "Train score: ", mnb.score(xTrainMatrix, yTrainMatrix)
		print "RMSE Train: ", mean_squared_error(yTrainMatrix, mnb.predict(xTrainMatrix))

		print  "Test score: ", mnbScore
		print "RMSE Test: ", mean_squared_error(yTestMatrix, mnb.predict(xTestMatrix))

	
	#Bernoulli
	bnb=MultinomialNB()
	bnb.fit(xTrainMatrix, yTrainMatrix)
	#bnb.predictions= bnb.predict(xTestMatrix)
	if (printAll):
		print "Bernoulli Naive-Bayes:"
		print "Train score: ", bnb.score(xTrainMatrix, yTrainMatrix)
		print "RMSE Train: ", mean_squared_error(yTrainMatrix, bnb.predict(xTrainMatrix))

		print "Test score: ", bnb.score(xTestMatrix, yTestMatrix)
		print "RMSE Test: ", mean_squared_error(yTestMatrix, bnb.predict(xTestMatrix))

	
	#Decision Tree Classifier
	treeClass=  tree.DecisionTreeClassifier()
	treeClass= treeClass.fit(xTrainMatrix, yTrainMatrix)
	#tc_predict=treeClass.predict(xTestMatrix)
	treeScore= treeClass.score(xTestMatrix, yTestMatrix)
	if (printAll):
		print "Decision Tree Classifier:"
		print "Train score: ", treeClass.score(xTrainMatrix, yTrainMatrix)
		print "RMSE Train: ", mean_squared_error(yTrainMatrix, treeClass.predict(xTrainMatrix))

		print "Test score: ", treeScore
		print "RMSE Test: ", mean_squared_error(yTestMatrix, treeClass.predict(xTestMatrix))

	
	scoreArr.append(mnbScore)
#Determine highest scores and associated attributes
mapOut = {}

for i in range(numColTest):
	mapOut[i]=scoreArr[i]

#Orded most important to least important index
print collections.OrderedDict(sorted(mapOut.items(), key=lambda t: t[1]))
# Not working # tree.export_graphviz(treeClass, out_file='tree.dot')  



