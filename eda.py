import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import json
import datetime
import time

pd.options.mode.chained_assignment = None

# Returns (lat, long) i.e. centroid for hexagon defining a region
def getRegions(data):
	features = data['features']
	place = dict()
	place['movement_id'] =[]
	place['name'] =[]
	place['lat'] = []
	place['long'] = []

	for feature in features:
		prop = feature['properties']
		place['movement_id'].append(prop['MOVEMENT_ID'])
		place['name'].append(prop['DISPLAY_NAME'])

		geo = feature['geometry']['coordinates'][0]
		lat = []
		lng = []
		for pt in geo:
			lat.append(pt[1])
			lng.append(pt[0])

		place['lat'].append(np.mean(lat))
		place['long'].append(np.mean(lng))

	df = pd.DataFrame.from_dict(place, orient='columns')
	return df

def featuriseDates(df):
	# Split the times into different features
	df['travel_date'] = pd.to_datetime(df['travel_date'], infer_datetime_format=True)
	df['travel_time'] = pd.to_datetime(df['travel_time'])
	df['year'] = df['travel_date'].dt.year
	df['month'] = df['travel_date'].dt.month
	df['day'] = df['travel_date'].dt.day
	df['day_of_week'] = df['travel_date'].dt.dayofweek
	df['hour'] = df['travel_time'].dt.hour
	df['minute'] = df['travel_time'].dt.minute

	return df

def countTickets(df):
	# Generate number of tickets per ride
	df['tickets'] = df.groupby(['ride_id'])['ride_id'].transform('size')
	# Remove duplicate Ids
	df.drop_duplicates(['ride_id'], keep='first', inplace=True)

	return df




def main():
	trainDf = pd.read_csv('data/mobiticket/train_revised.csv')
	testDf = pd.read_csv('data/mobiticket/test_questions.csv')

	datedTrain = featuriseDates(trainDf)
	ticketedTrain = countTickets(datedTrain)

	datedTest = featuriseDates(testDf)

	
	# Drop unnecessary columns for train
	ticketedTrain.drop(['seat_number', 'payment_method', 'payment_receipt', 'car_type', 'travel_to', 'travel_date', 'travel_time'], axis=1, inplace=True)
	cols = ['ride_id', 'travel_from', 'max_capacity', 'year', 'month', 'day', 'day_of_week', 'hour', 'minute', 'tickets']
	finalTrain = ticketedTrain[cols]

	# Drop for test
	datedTest.drop(['car_type', 'travel_to', 'travel_date', 'travel_time'], axis=1, inplace=True)


	# Write to files
	finalTrain.to_csv('data/mobiticket/processed_train.csv', index=False)
	datedTest.to_csv('data/mobiticket/processed_test.csv', index=False)

	


	


if __name__=="__main__":
	main()