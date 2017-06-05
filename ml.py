# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Predicting Household Income From United States Census Data

import os
import numpy as np
import math

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
xTrain = np.zeros((numRowTotal, numCol-1))
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
			xTrain[loadIndex][lineIndex] = float(rawVal)
		else:
			#Has an associated string-value mapping
			mappedVal = mapVals[lineIndex][rawVal]
			xTrain[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTrain[loadIndex] = mappedVal
	loadIndex +=1

dataFile.close()
print xTrain[0]


#Load TEST data into X and Y matrix
fileLoc = 'census-income.test'
numRowTotal = 99762
xTest = np.zeros((numRowTotal, numCol-1))
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
			xTest[loadIndex][lineIndex] = float(rawVal)
		else:
			mappedVal = mapVals[lineIndex][rawVal]
			xTest[loadIndex][lineIndex] = mappedVal
		lineIndex += 1
	#Load associated value for Y Matrix
	rawVal = line[lineIndex].strip()
	mappedVal = mapVals[lineIndex][rawVal]
	yTest[loadIndex] = mappedVal
	loadIndex +=1

dataFile.close()

print xTest[0]