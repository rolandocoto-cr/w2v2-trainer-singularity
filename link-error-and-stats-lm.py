#- make a python program with two parameters
#	- name of the errorfile
#	- name of the stats file
#- put the WER and CER numbers in a vector
#- find the validation WER lines from the errorfile
#- see if i have the same number of items as in the CER/WER
#- match them
#- add metadata (e.g. type of run, language)
#- output a csv with this
# Calculate duration of the run

#python3 link-error-and-stats.py /home/rolando/Desktop/deploy-asr-hpc/dz-fromScratch-noAugmentation-005mins-01-stats-median.txt dz-fromScratch-noAugmentation-005mins-01-output.txt dz-fromScratch-noAugmentation-005mins-01-stats.csv

#python3 link-error-and-stats.py /home/rolando/Desktop/deploy-asr-hpc/dz-fromScratch-noAugmentation-251mins-01-stats-median.txt dz-fromScratch-noAugmentation-251mins-01-output.txt dz-fromScratch-noAugmentation-251mins-01-stats.csv

#python3 link-error-and-stats.py /home/rolando/Desktop/deploy-asr-hpc/dz-transferTibetanJuly3200-noAugmentation-251mins-01-stats-median.txt dz-transferTibetanJuly3200-noAugmentation-251mins-01-output.txt dz-transferTibetanJuly3200-noAugmentation-251mins-01-stats.csv

import sys

statsPath = sys.argv[1]
outputPath = sys.argv[2]
writePath = sys.argv[3]

#========================================
# Read the stats
#========================================

statsFile = open(statsPath, "r")
statsLines = statsFile.read()
statsLines = statsLines.split("\n")

epoch = []
cer  = []
wer = []

epochLM = []
cerLM  = []
werLM = []

for l in statsLines:
	
	if (("CER" in l) or ("WER" in l)) and ("/" in l) and ("withoutLM" in l):
		
		t = l.split("/")
		#print(t[1])
		data = t[1].split("\t")
		tNumber = data[1]
		tEpoch = data[0].split(" ")[0]
		tType = data[0].split(" ")[2].replace(":","")
		
		print(tNumber + " - " + tEpoch + " - " + tType)
		
		if (tEpoch not in epoch): epoch.append(tEpoch)
		if (tType == "WER"): wer.append(tNumber)
		elif (tType == "CER"): cer.append (tNumber)

	if (("CER" in l) or ("WER" in l)) and ("/" in l) and ("withLM" in l):
		
		t = l.split("/")
		#print(t[1])
		data = t[1].split("\t")
		tNumber = data[1]
		tEpoch = data[0].split(" ")[0]
		tType = data[0].split(" ")[2].replace(":","")
		
		print(tNumber + " - " + tEpoch + " - " + tType)
		
		if (tEpoch not in epochLM): epochLM.append(tEpoch)
		if (tType == "WER"): werLM.append(tNumber)
		elif (tType == "CER"): cerLM.append (tNumber)

print(cer)
print(cerLM)

tArr = []
tArrLM = []

for i in range(0,len(epoch)):
	print(epoch[i] + " - " + cer[i] + " - " + wer[i])
	tArr.append([int(epoch[i]), [cer[i], wer[i]]])

for i in range(0,len(epochLM)):
	print(epochLM[i] + " - " + cerLM[i] + " - " + werLM[i])
	tArrLM.append([int(epochLM[i]), [cerLM[i], werLM[i]]])
	
tArr.sort()
tArrLM.sort()
		
print(tArr)
print(tArrLM)

epoch = []
cer  = []
wer = []

epochLM = []
cerLM  = []
werLM = []

for r in tArr:
	epoch.append(str(r[0]))
	cer.append(r[1][0])
	wer.append(r[1][1])
    
for r in tArrLM:
	epochLM.append(str(r[0]))
	cerLM.append(r[1][0])
	werLM.append(r[1][1])

print("========= stats File ===========")
print(epochLM)
print(cerLM)
print(werLM)

#========================================
# Read the error file
#========================================

outputFile = open(outputPath, "r")
outputLines = outputFile.read()
outputLines = outputLines.split("\n")

counter = 1

validWer = []

for l in outputLines:
	if "eval_wer" in l:
		print(str(counter))
		if (counter%4 == 0):
			print("***")
			print(l.split(" "))
			print(l.split(" ")[3])
			validWer.append(l.split(" ")[3].replace(",",""))
		print(l)
		counter = counter+1

print("========= error File ===========")
print(len(tArr))
print(len(validWer))


#========================================
# Calculate duration
#========================================

from datetime import datetime

startString = ""
endString = ""

for l in statsLines:
	if ("Start" in l):
		startString = l.replace("Start: ","")
	if ("End" in l):
		endString = l.replace("End: ", "")
		
print(startString)
print(endString)

startTime = datetime.strptime(startString, '%Y-%m-%d %H:%M:%S.%f')
endTime = datetime.strptime(endString, '%Y-%m-%d %H:%M:%S.%f')
print(startTime)
print(endTime)

deltaTime = (endTime - startTime).total_seconds() / 60
print(deltaTime)

#========================================
# Add metadata
#========================================

print(statsPath)
splitPath = statsPath.split("/")
filename = splitPath[len(splitPath)-1]
print(filename)
fileInfo = filename.split("-")
print(fileInfo)

language = fileInfo[0]
condition = fileInfo[1]
augmentation = fileInfo[2]
mins = int(fileInfo[3].replace("mins",""))
run = fileInfo[4]

transferLang = ""
transferModel = ""

isTransfer = 0
if ("transfer" in filename): isTransfer = 1

if (isTransfer == 1):
	tTransferInfo = condition.replace("transfer", "")
	if ("Tibetan" in tTransferInfo):
		transferLang = "tib"
		tTransferInfo = condition.replace("Tibetan", "")
	transferModel = tTransferInfo

if (augmentation == "noAugmentation"): augmentation = "none"

print(language)
print(condition)
print(augmentation)
print(mins)
print(run)

csvOutput = "language" + "\t" + "condition" + "\t" + "augmentation" + "\t" + "mins" + "\t" + "run" + "\t" + "epoch" + "\t" + "cer" + "\t" + "wer" + "\t" + "validWer" + "\t" + "convergeModel" + "\ttrainingMins\ttransferLang\ttransferModel\n"

augmentation = "withoutLM"
for i in range(0,len(epoch)):
	csvOutput = csvOutput + language + "\t" + condition + "\t" + augmentation + "\t" + str(mins) + "\t" + run + "\t" + epoch[i] + "\t" + cer[i] + "\t" + wer[i] + "\t" + validWer[i] + "\t\t" + str(deltaTime) + "\t" + transferLang + "\t" + transferModel + "\n"
    
augmentation = "withLM"
for i in range(0,len(epoch)):
	csvOutput = csvOutput + language + "\t" + condition + "\t" + augmentation + "\t" + str(mins) + "\t" + run + "\t" + epochLM[i] + "\t" + cerLM[i] + "\t" + werLM[i] + "\t" + validWer[i] + "\t\t" + str(deltaTime) + "\t" + transferLang + "\t" + transferModel + "\n"


#========================================
# Write CSV file
#========================================

f = open(writePath, "w")
f.write(csvOutput)
f.close()

statsFile.close()
outputFile.close()
