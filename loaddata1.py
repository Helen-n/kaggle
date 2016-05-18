import re
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing


def processActionType():
    global df
    df['action_type'] = pd.factorize(df['action_type'])[0]

def processCombinedShotType():
    global df
    df['combined_shot_type'] = pd.factorize(df['combined_shot_type'])[0]

def processShotType():
    global df
    df['shot_type'] = pd.factorize(df['shot_type'])[0]

def processShotZoneArea():
    global df
    df['shot_zone_area'] = pd.factorize(df['shot_zone_area'])[0]

def processShotZoneBasic():
    global df
    df['shot_zone_basic'] = pd.factorize(df['shot_zone_basic'])[0]

def processShotZoneRange():

    global df
    df['shot_zone_range'] = pd.factorize(df['shot_zone_range'])[0]

def processOpponent():
    global df
    df['opponent'] = pd.factorize(df['opponent'])[0]

def processTimeRemaining():
    global df
    minutes_remaining = np.array(df['minutes_remaining'])
    seconds_remaining = np.array(df['seconds_remaining'])
    df['time_remaining'] = minutes_remaining*60 + seconds_remaining

def processSeason():
    global df
    df['season'] = pd.factorize(df['season'])[0]

def processDrops():
    global df
    rawDropList = ['minutes_remaining', 'seconds_remaining', 'game_date', 'game_event_id', 'shot_id']
    df.drop(rawDropList, axis=1, inplace=True)

def reduceAndCluster(input_df, submit_df, clusters=3):
    """
    Takes the train and test data frames and performs dimensionality reduction with PCA and clustering
    
    This was part of some experimentation and wasn't used for top scoring submissions. Leaving it in for reference
    """
    
    # join the full data together
    df = pd.concat([input_df, submit_df])
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df = df.reindex_axis(input_df.columns, axis=1)
    
    # Series of labels
    survivedSeries = pd.Series(df['shot_made_flag'], name='shot_made_flag')
    
    print df.head()
    
    # Split into feature and label arrays
    X = df.values[:, 1::]
    y = df.values[:, 0]
    
    print X[0:5]
    
    
    # Minimum percentage of variance we want to be described by the resulting transformed components
    variance_pct = .99
    
    # Create PCA object
    pca = PCA(n_components=variance_pct)
    
    # Transform the initial features
    X_transformed = pca.fit_transform(X,y)
    
    # Create a data frame from the PCA'd data
    pcaDataFrame = pd.DataFrame(X_transformed)
    
    print pcaDataFrame.shape[1], " components describe ", str(variance_pct)[1:], "% of the variance"
    
    
    
    # use basic clustering to group similar examples and save the cluster ID for each example in train and test
    kmeans = KMeans(n_clusters=clusters, random_state=np.random.RandomState(4), init='random')
     
    #==============================================================================================================
    # # Perform clustering on labeled AND unlabeled data
    # clusterIds = kmeans.fit_predict(X_pca)
    #==============================================================================================================
    
    # Perform clustering on labeled data and then predict clusters for unlabeled data
    trainClusterIds = kmeans.fit_predict(X_transformed[:input_df.shape[0]])
    print "clusterIds shape for training data: ", trainClusterIds.shape
    #print "trainClusterIds: ", trainClusterIds
     
    testClusterIds = kmeans.predict(X_transformed[input_df.shape[0]:])
    print "clusterIds shape for test data: ", testClusterIds.shape
    #print "testClusterIds: ", testClusterIds
     
    clusterIds = np.concatenate([trainClusterIds, testClusterIds])
    print "all clusterIds shape: ", clusterIds.shape
    #print "clusterIds: ", clusterIds
    
    
    # construct the new DataFrame comprised of "Survived", "ClusterID", and the PCA features
    clusterIdSeries = pd.Series(clusterIds, name='ClusterId')
    df = pd.concat([survivedSeries, clusterIdSeries, pcaDataFrame], axis=1)
    
    # split into separate input and test sets again
    input_df = df[:input_df.shape[0]]
    submit_df = df[input_df.shape[0]:]
    submit_df.reset_index(inplace=True)
    submit_df.drop('index', axis=1, inplace=True)
    submit_df.drop('shot_made_flag', axis=1, inplace=1)
    
    return input_df, submit_df

def getDataSets():

    global  df
    df = pd.read_csv('data/data.csv', header=0)

    # process the individual variables present in the raw data
    processActionType()
    processCombinedShotType()
    processShotType()
    processShotZoneArea()    
    processShotZoneBasic()    
    processShotZoneRange()
    processOpponent()
    processShotZoneRange()
    processSeason()
    processDrops()

    df_train = df[df['shot_made_flag'].notnull()]
    df_test  = df[df['shot_made_flag'].isnull()]
    df_submission = pd.read_csv('data/sample_submission.csv', header=0)
    return df_train, df_test, df_submission
