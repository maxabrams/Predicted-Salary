# Max Abrams & Cynthia Le
# COEN 129 Final Project
# Determine the number of options for every row

fileLoc = 'census-income.data'
dataFile = open(fileLoc, 'r')
numRowTotal = 199523
numCol = 42

vals = [dict() for x in range(numCol)]

for line in dataFile:
	line = line.strip()
	line = line.split(',')
	for i in range(0, len(line)):
		#print i
		#print line[i]
		currentDict = vals[i]
		#print currentDict
		if not line[i] in currentDict:
			currentDict[line[i].strip()] = len(currentDict) # =1 --for count
		#else:
			#currentDict[line[i]] += 1 -- for count

for i in range(numCol):
	print str(len(vals[i])) + '\t'"distinct values in row " + str(i)
	print vals[i]
	print