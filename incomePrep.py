#Prepare the dataset for imcome preparation 
import os 
import numpy as np 
import pandas as pd 

#Read and Parse the data 
dataTrain= []; dataTest = []
columns = ['Age','WorkClass','FinalWeight','Education','EducationNumber','MaritalStatus','Job','Relationship'
			,'Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek','Origin','IncomeLevel']
fileTrain = 'adult.data'
fileTest = 'adult.test'
lines = open(fileTrain,'r')
for line in lines.readlines():
	l = line.strip().split(',')
	l = [x.strip() for x in l]
	if len(l) != len(columns):
		continue
	dataTrain.append(l)
dfTrain = pd.DataFrame(dataTrain,columns = columns)


lines = open(fileTest,'r')
for line in lines.readlines():
	l = line.strip().split(',')
	l = [x.strip() for x in l]
	if len(l) != len(columns):
		continue
	dataTest.append(l)
dfTest = pd.DataFrame(dataTest,columns = columns)


#Delete the arrays of test and train
del dataTrain,dataTest 
for col in columns:
	dfTrain = dfTrain[dfTrain[col] != '?']
	dfTest = dfTest[dfTest[col] != '?']

#Exploratory data analysis 
#Lets look at some plots to understand the data before applying a ML/DS model 
continuousCols = ['Age','FinalWeight','CapitalGain','CapitalLoss','HoursPerWeek','EducationNumber']
catColumns = list(set(columns) - set(continuousCols))


#Transforming the Continuous variables 
#We can either normalize them or use Z-score transformation here
for col in continuousCols:
	dfTrain[col] = dfTrain[col].astype('float32')
	dfTest[col] = dfTest[col].astype('float32')

print(dfTrain.info())
print(dfTest.info())
dfTrain.to_csv('incomeTrain.csv')
dfTest.to_csv('incomeTest.csv')

trainData = pd.read_csv('incomeTrain.csv')
testData = pd.read_csv('incomeTest.csv')

print(trainData.info())
print(testData.info())

#Transforming the Categorical variables 
#Categorical variables can be either label-encoded or one-hot encoded 






