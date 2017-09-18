from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

#euclidian_distance = sqrt( (plot1[0] - plot2[0]) **2 + (plot1[1] - plot2[1]) **2 )

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

test = [[plt.scatter(ii[0],ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
print(test)
plt.scatter(new_features[0], new_features[1], s = 100, color = 'g')
plt.show()

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
	print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	#KNNAlgo
	return vote_result
	
result = k_nearest_neighbors(dataset, [4,4])
print(result)