# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Predicting Household Income From United States Census Data

import os
import numpy as np
import math

#Read Load Data
fileLoc = 'census-income.data'
numRowTotal = 199523
numCol = 42

#Make dictionary value mapings
dataFile = open(fileLoc, 'r')
mapVals = [dict() for x in range(numCol)]

for line in dataFile:
	line = line.strip()
	line = line.split(',')
	for i in range(0, len(line)):
		#print i
		#print line[i]
		currentDict = mapVals[i]
		#print currentDict
		if not line[i] in currentDict:
			currentDict[line[i].strip()] = len(currentDict)

dataFile.close()