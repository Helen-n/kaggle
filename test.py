#import pandas as pd
#import numpy as np
#from sklearn.cross_validation import cross_val_score, ShuffleSplit
#from sklearn.ensemble import RandomForestRegressor
#from loaddata1 import getDataSets

#trainData, testData  ,submiison = getDataSets()
#Y = np.array(trainData['shot_made_flag'])
#X = np.array(trainData.drop('shot_made_flag',axis=1))



#rf = RandomForestRegressor(n_estimators=20, max_depth=4)
#scores = []
#for i in range(X.shape[1]):
#    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2", cv=ShuffleSplit(len(X), 3, 3))
#    scores.append((round(np.mean(score), 3),i))
#map(abs, scores)
#print sorted(scores, reverse = True)
#import numpy as np
#from sklearn.cross_validation import cross_val_score, ShuffleSplit
#from sklearn.datasets import load_boston
#from sklearn.ensemble import RandomForestRegressor

##Load boston housing dataset as an example
#boston = load_boston()
#X = boston["data"]
#Y = boston["target"]
#names = boston["feature_names"]

#rf = RandomForestRegressor(n_estimators=20, max_depth=4)
#scores = []
#for i in range(X.shape[1]):
#     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
#                              cv=ShuffleSplit(len(X), 3, .3))
#     scores.append((round(np.mean(score), 3), names[i]))
#print sorted(scores, reverse=True)

## SVM
#import csv
#import pandas as pd
#import numpy as np
#from sklearn import svm
#from loaddata1 import getDataSets
#trainData, testData  ,submission = getDataSets()
#Y = np.array(trainData['shot_made_flag'])
#X = np.array(trainData.drop('shot_made_flag',axis=1))
#testData = np.array(testData.drop('shot_made_flag',axis=1))
#clf = svm.SVC()
#clf.fit(X,Y)
#output = clf.predict(testData)
#map(int,output)
#ids = submission['shot_id']
# # write results
#predictions_file = open("mysubmission.csv", "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["shot_id","shot_made_flag"])
#open_file_object.writerows(zip(ids, output))
#predictions_file.close()
#print 'Done.'

import pandas as pd
import numpy as np
import csv
submission = pd.read_csv('mysubmission.csv')
ids = submission['shot_id']
output = submission.shot_made_flag.astype(np.int)
predictions_file = open("mysubmission2.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["shot_id","shot_made_flag"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'done.'

