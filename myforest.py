import loaddata1
import csv
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter

if __name__ == '__main__':
    # Do all the feature engineering
    print "Gender the initial training/test sets"
    train_df , test_df ,submit_df = titanic2.getDataSets()
    y = train_df['shot_made_flag']
    #print(len(y))
    train_df.drop('shot_made_flag',axis=1, inplace=True)
    X  = train_df
    #print(len(X))
    test_df.drop('shot_made_flag', axis=1, inplace=True)
    test_data = test_df
    ids = submit_df['shot_id']
    
    print 'Training...'
    forest = RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=6,min_samples_leaf=2)
    forest = forest.fit( X, y)

    print 'Predicting...'
    output = forest.predict(test_data).astype(int)

    predictions_file = open("mysubmission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["shot_id","shot_made_flag"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'



     