import time
import collections

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR


def randomForest(xTrain, xTest, yTrain, yTest):
	rfr = RandomForestRegressor(n_estimators=100, criterion='mae', n_jobs=2, max_depth=7)

	rfr.fit(xTrain, yTrain)
	yPred = rfr.predict(xTest)
	mae = mean_absolute_error(yTest, yPred)

	data = {"Model": rfr, "MAE": mae}
	return data

def ridgeRegression(xTrain, xTest, yTrain, yTest):
	rdg = Ridge(alpha=1.2, normalize=True)

	rdg.fit(xTrain, yTrain)
	yPred = rdg.predict(xTest)
	mae = mean_absolute_error(yTest, yPred)

	data = {"Model": rdg, "MAE": mae}
	return data

def elasticNet(xTrain, xTest, yTrain, yTest):
	els = ElasticNet()

	els.fit(xTrain, yTrain)
	yPred = els.predict(xTest)
	mae = mean_absolute_error(yTest, yPred)

	data = {"Model": els, "MAE": mae}
	return data

def supportVectorRegressor(xTrain, xTest, yTrain, yTest):
	svr = SVR(kernel='rbf')

	svr.fit(xTrain, yTrain)
	yPred = svr.predict(xTest)
	mae = mean_absolute_error(yTest, yPred)

	data = {"Model": svr, "MAE": mae}
	return data




def main():
	trainRaw = pd.read_csv('data/mobiticket/processed_train.csv')
	testRaw = pd.read_csv('data/mobiticket/processed_test.csv')


	# Convert travel_from to unique IDs on both sets
	trainRaw['travel_from'] = trainRaw['travel_from'].astype('category')
	originCategories = trainRaw['travel_from'].cat.categories
	trainRaw['origin'] = trainRaw['travel_from'].cat.codes

	testRaw['origin'] = testRaw['travel_from'].astype('category', categories=originCategories).cat.codes

	trainRaw.drop(['travel_from', 'ride_id'], axis=1, inplace=True)
	testRaw.drop(['travel_from'], axis=1, inplace=True)

	# Read X, y for training
	X = trainRaw.drop(['tickets'], axis=1)
	Y = trainRaw['tickets']

	XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

	# Try different algorithms
	algols = [randomForest, ridgeRegression, elasticNet, supportVectorRegressor]
	results = []
	for algol in algols:
		out = algol(XTrain, XTest, yTrain, yTest)
		results.append(out)

	# Get lowest MAE algol and use it
	i = 0
	mn = 100
	pt = 0
	for result in results:
		mae = result['MAE']
		if(mae < mn):
			mn = mae
			pt = i
		i = i+1

	

	# Predict
	x = testRaw.drop(['ride_id'], axis=1)

	model = results[pt]['Model']
	y = model.predict(x)

	preds = {'ride_id': testRaw['ride_id'], 'number_of_ticket': y}
	predDf = pd.DataFrame.from_dict(preds)
	
	predDf.to_csv('data/mobiticket/test_questions_predictions.csv', index=False)







if __name__ == "__main__":
	main()
