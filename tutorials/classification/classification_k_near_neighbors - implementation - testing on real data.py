from math import sqrt 
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
	if(len(data) >= k) :
		warnings.warn('K is set to a value less than total voting groups')
		
	distances = []
	for group in data:
		for features in data[group]:
			#euclidian_distance = sqrt( (features[0] - predict[0])**2 + (features[1] - predict[1])**2 )
			#euclidian_distance = np.sqrt( np.sum( ( np.array(features) - np.array(predict) ) **2 ) )
			euclidian_distance = np.linalg.norm( np.array(features) - np.array(predict) )
			distances.append([euclidian_distance, group])
			
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][0] / k
	#KNNAlgo
	return vote_result, confidence
	
df = pd.read_csv("D:/IMP/ML/PyTut/classification/breast-cancer-wisconsin.data.txt")
df.replace("?",-99999, inplace=True) 
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2

train_set = {2: [], 4:[]}
test_set = {2: [], 4:[]}

train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for data in train_data:
	train_set[data[-1]].append(data[:-1])
	
for data in test_data:
	test_set[data[-1]].append(data[:-1])
	
correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(test_set, data, k=5)
		if vote == group:
			correct += 1
		else:
			print(confidence)
		total += 1
		
print("Accuracy: ", correct/total)