# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Determine the number of options for every row
import collections

fileLoc = 'census-income.data'
dataFile = open(fileLoc, 'r')
numRowTotal = 199523
numCol = 42

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

vals = [dict() for x in range(numCol)]

valD = []

for line in dataFile:
	line = line.strip()
	line = line.split(',')
	for i in range(0, len(line)):
		#print i
		#print line[i]
		if (i == 5):
			valD.append(float(line[i]))
		currentDict = vals[i]
		#print currentDict
		if(is_number(line[i].strip()) == False): #Only map strings, leave numbers alone
			if not line[i].strip() in currentDict:
				currentDict[line[i].strip()] = len(currentDict) # =1 --for count
		#else:
			#currentDict[line[i]] += 1 -- for count

for i in range(numCol):
	print str(len(vals[i])) + '\t'"distinct values in row " + str(i)
	print vals[i]
	print

valD.sort()
print valD