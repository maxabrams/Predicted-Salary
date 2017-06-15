import os
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import math

#---1
fileLoc = 'iris.data'
dataFile = open(fileLoc, 'r')

#Load Setosa
setosa_training = np.zeros((40,4))
setosa_u = np.zeros((1,4))
for i in range (0, 40):
	line = dataFile.readline()
	line = line.split(",")
	setosa_training[i] = [line[0],line[1],line[2],line[3]]
	setosa_u[0] = [setosa_u[0][0]+float(line[0]),setosa_u[0][1]+float(line[1]),setosa_u[0][2]+float(line[2]),setosa_u[0][3]+float(line[3])]
print "Iris-Setosa Training Data:\n", setosa_training

setosa_u[0] = setosa_u[0]/40.0 #calculate mu #[setosa_u[0][0]/40.0,setosa_u[0][1]/40.0,setosa_u[0][2]/40.0,setosa_u[0][3]/40.0]
print "Setosa Mu:", setosa_u

setosa_test = np.zeros((10,4))
for i in range (0, 10):
	line = dataFile.readline()
	line = line.split(",")
	setosa_test[i] = [line[0],line[1],line[2],line[3]]
print "Iris-Setosa Test Data:\n", setosa_test

#Load Versicolor
versicolor_training = np.zeros((40,4))
versicolor_u = np.zeros((1,4))
for i in range (0, 40):
	line = dataFile.readline()
	line = line.split(",")
	versicolor_training[i] = [line[0],line[1],line[2],line[3]]
	versicolor_u[0] = [versicolor_u[0][0]+float(line[0]),versicolor_u[0][1]+float(line[1]),versicolor_u[0][2]+float(line[2]),versicolor_u[0][3]+float(line[3])]
print "Iris-Versicolor Training Data:\n", versicolor_training

versicolor_u[0] = versicolor_u[0]/40.0 #calculate mu
print "Versicolor Mu:", versicolor_u

versicolor_test = np.zeros((10,4))
for i in range (0, 10):
	line = dataFile.readline()
	line = line.split(",")
	versicolor_test[i] = [line[0],line[1],line[2],line[3]]
print "Iris-Versicolor Test Data:\n", versicolor_test

#Load Virginica
virginica_training = np.zeros((40,4))
virginica_u = np.zeros((1,4))
for i in range (0, 40):
	line = dataFile.readline()
	line = line.split(",")
	virginica_training[i] = [line[0],line[1],line[2],line[3]]
	virginica_u[0] = [virginica_u[0][0]+float(line[0]),virginica_u[0][1]+float(line[1]),virginica_u[0][2]+float(line[2]),virginica_u[0][3]+float(line[3])]
print "Iris-Virginica Training Data:\n", virginica_training

virginica_u[0] = virginica_u[0]/40.0 #calculate mu
print "Virginica Mu:", virginica_u

virginica_test = np.zeros((10,4))
for i in range (0, 10):
	line = dataFile.readline()
	line = line.split(",")
	virginica_test[i] = [line[0],line[1],line[2],line[3]]
print "Iris-Virginica Test Data:\n", virginica_test

#Mu is already calculated, so now we can caluclate sigma
print

setosa_s = np.zeros((4,4))
for i in range(0,40):
	setosa_s += np.outer((setosa_training[i]-setosa_u),np.transpose(setosa_training[i]-setosa_u))
setosa_s = setosa_s / 40.0
print "Setosa sigma:\n", setosa_s 

versicolor_s = np.zeros((4,4))
for i in range(0,40):
	versicolor_s += np.outer((versicolor_training[i]-versicolor_u),np.transpose(versicolor_training[i]-versicolor_u))
versicolor_s = versicolor_s / 40.0
print "Versicolor sigma:\n", versicolor_s 

virginica_s = np.zeros((4,4))
for i in range(0,40):
	virginica_s += np.outer((virginica_training[i]-virginica_u),np.transpose(virginica_training[i]-virginica_u))
virginica_s = virginica_s / 40.0
print "Virginica sigma:\n",virginica_s 

lda_s = (setosa_s + versicolor_s + virginica_s) / 3.0
print "LDA sigma:\n", lda_s

# HELPER functions / models for #2 and #3

#Model function used for LDA and QUDA
def model(x, mu, sigma):
	#print "model call"
	#print sigma
	normal = x-mu
	#print "normal" , normal
	transpose = np.transpose(x-mu)
	#print "transpose", transpose
	div = math.pow((2.0 * math.pi),2.0) * np.sqrt(det(sigma))
	#print "div" , div
	#print "first" , np.dot(normal, inv(sigma))
	expVal = -1.0/2.0 * np.dot(np.dot(normal, inv(sigma)),transpose)
	#print "exp", expVal
	output = (1.0 / div) * np.exp(expVal)
	#print "output" , output
	return output

#Method for LDA
def LDA(x):
	o1 = model(x, setosa_u, lda_s)
	o2 = model(x, virginica_u, lda_s)
	o3 = model(x, versicolor_u, lda_s)
	outputs = [o1, o2, o3]
	#print "calculated:" , outputs
	if(max(outputs)==o1):
		return "setosa"
	elif(max(outputs)==o2):
		return "virginica"
	else:
		return "versicolor"

#Method for QDA
def QDA(x):
	o1 = model(x, setosa_u, setosa_s)
	o2 = model(x, virginica_u, virginica_s)
	o3 = model(x, versicolor_u, versicolor_s)
	outputs = [o1, o2, o3]
	#print "calculated:" , outputs
	if(max(outputs)==o1):
		return "setosa"
	elif(max(outputs)==o2):
		return "virginica"
	else:
		return "versicolor"

#---2
print

#Run error tests for testing data LDA
errorSetosa = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorSetosa[LDA(setosa_training[i])] +=1
print "LDA Setosa training results: ", errorSetosa

errorVirginica = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorVirginica[LDA(virginica_training[i])] +=1
print "LDA Virginica training results: ", errorVirginica

errorVersicolor= {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorVersicolor[LDA(versicolor_training[i])] +=1
print "LDA Versicolor training results: ", errorVersicolor

error = errorSetosa['virginica'] + errorSetosa['versicolor'] + errorVirginica['setosa'] + errorVirginica['versicolor'] + errorVersicolor['setosa'] + errorVersicolor['virginica']
error = error / 120.0
print "--LDA training data error:" , error

#Run error on test data LDA
errorTestSetosa = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestSetosa[LDA(setosa_test[i])] +=1
print "LDA Setosa test results: ", errorTestSetosa

errorTestVirginica = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestVirginica[LDA(virginica_test[i])] +=1
print "LDA Virginica test results: ", errorTestVirginica

errorTestVersicolor= {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestVersicolor[LDA(versicolor_test[i])] +=1
print "LDA Versicolor test results: ", errorTestVersicolor

error = errorTestSetosa['virginica'] + errorTestSetosa['versicolor'] + errorTestVirginica['setosa'] + errorTestVirginica['versicolor'] + errorTestVersicolor['setosa'] + errorTestVersicolor['virginica']
error = error / 30.0
print "--LDA test error rate:" , error

#---3 
print

#Run error tests for testing data QDA
errorSetosa = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorSetosa[QDA(setosa_training[i])] +=1
print "QDA Setosa training results: ", errorSetosa

errorVirginica = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorVirginica[QDA(virginica_training[i])] +=1
print "QDA Virginica training results: ", errorVirginica

errorVersicolor= {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,40):
	errorVersicolor[QDA(versicolor_training[i])] +=1
print "QDA Versicolor training results: ", errorVersicolor

error = errorSetosa['virginica'] + errorSetosa['versicolor'] + errorVirginica['setosa'] + errorVirginica['versicolor'] + errorVersicolor['setosa'] + errorVersicolor['virginica']
error = error / 120.0
print "--QDA training data error:" , error

#Run error on test data QDA
errorTestSetosa = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestSetosa[QDA(setosa_test[i])] +=1
print "QDA Setosa test results: ", errorTestSetosa

errorTestVirginica = {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestVirginica[QDA(virginica_test[i])] +=1
print "QDA Virginica test results: ", errorTestVirginica

errorTestVersicolor= {'setosa':0, 'virginica':0, 'versicolor':0}
for i in range(0,10):
	errorTestVersicolor[QDA(versicolor_test[i])] +=1
print "QDA Versicolor test results: ", errorTestVersicolor

error = errorTestSetosa['virginica'] + errorTestSetosa['versicolor'] + errorTestVirginica['setosa'] + errorTestVirginica['versicolor'] + errorTestVersicolor['setosa'] + errorTestVersicolor['virginica']
error = error / 30.0
print "--QDA test error rate:" , error
