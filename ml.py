# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Predicting Household Income From United States Census Data

import os
import numpy as np
import math

#Read Load Data
fileLoc = 'census-income.data'
dataFile = open(fileLoc, 'r')
numRowTotal = 199523
numCol = 41

#Use 90% as training data, 10% as test data
numRowTrain = math.floor(numRowTotal * .9)
numRowTest = numRowTotal - numRowTrain
