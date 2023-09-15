'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn
import mlflow
import mlflow
from pycaret.classification import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from Lead_scoring_training_pipeline.constants import *
from constants import DB_PATH,DB_FILE_NAME,FEATURES_TO_ENCODE,ONE_HOT_ENCODED_FEATURES,TRACKING_URI,EXPERIMENT,DB_FILE_MLFLOW

print("DB_PATH : {} \nDB_FILE_NAME: {}\nFEATURES_TO_ENCODE: {}\nONE_HOT_ENCODED_FEATURES: {} \nTRACKING_URI:{}\nEXPERIMENT:{}\nDB_FILE_MLFLOW: {}\n".format(DB_PATH,DB_FILE_NAME,FEATURES_TO_ENCODE,ONE_HOT_ENCODED_FEATURES,TRACKING_URI,EXPERIMENT,DB_FILE_MLFLOW))

import sqlite3
from sqlite3 import Error
import os 


#create a sqlite db fo storing all the model artifacts etc

def create_sqlit_connection(db_path,db_file):
     """ create a database connection to a SQLite database """
     print(db_path,db_file)
     conn = None
     # opening the conncetion for creating the sqlite db
     try:
         conn = sqlite3.connect(db_path+db_file)
         print(sqlite3.version)
     # return an error if connection not established
     except Error as e:
         print(e)
     # closing the connection once the database is created
     finally:
         if conn:
             conn.close()

###############################################################################
# Define the function to encode features
# ##############################################################################

def encode_features():
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''

    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    query = "SELECT * FROM interactions_mapped" 
    data = pd.read_sql_query(query, conn)
    #print(data.head(5))
    print("Before encoding data shape ({}): {}".format(len(data.columns),','.join(data.columns)))
    print("Before encoding data shape : {}".format(data.shape))
    for f in FEATURES_TO_ENCODE:
        if(f in data.columns):
            encoded = pd.get_dummies(data[f])
            encoded = encoded.add_prefix(f + '_')
            data = pd.concat([data,encoded],axis=1)
        else:
            print("Feature {} not found".format(f))
    data.drop(columns=FEATURES_TO_ENCODE,inplace=True)
    print("After encoding data shape ({}): {}".format(len(data.columns),','.join(data.columns)))
    print("After encoding data shape : {}".format(data.shape))
    # Save the encoded features into a new table named 'features
    db_file_path = os.path.join(DB_PATH,DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    df_encoded = data[ONE_HOT_ENCODED_FEATURES]
    df_encoded.to_sql('features', conn, if_exists='replace', index=False)
    #print(df_encoded)
    # Close the database connection


    target_df = data['app_complete_flag']
    # Save the target variable into a new table named 'target'
    target_df.to_sql('target', conn, if_exists='replace', index=False)
    
    data.to_sql('data', conn, if_exists='replace', index=False)
    # Close the database connection again
    conn.close()


###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model():
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    # Start an MLflow experiment
    print(EXPERIMENT)
    # Create a connection to the SQLite database
    conn = sqlite3.connect(DB_FILE_MLFLOW)
    conn.close()
    
    # Load the features and target variable from the database
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    query = "SELECT * FROM dataset" 
    data = pd.read_sql_query(query, conn)
    #print(data)
        

    #print(data.head())

    tracking_uri ="'{}'".format(TRACKING_URI)
    print(tracking_uri)
    mlflow.set_tracking_uri("'{}'".format(tracking_uri))
    #mlflow.set_tracking_uri("TRACKING_URI")
    mlflow.set_tracking_uri("http://0.0.0.0:6006")
    Lead_Scoring_Training_Pipeline = setup(data = data,target='app_complete_flag',session_id = 42,fix_imbalance=False,n_jobs=-1,use_gpu=False,log_experiment=True,experiment_name=EXPERIMENT,log_plots=True, log_data=True,silent=True, verbose=True,log_profile=False)
    # Create a list of models to exclude
    exclude_models = ['gbc', 'knn', 'qda', 'dummy', 'svm', 'ada']
    #top_models = compare_models(include = [ 'xgboost'])
    # Compare models with exclusion
    top_models = compare_models(exclude=exclude_models)
    lgbm  = create_model('lightgbm', fold = 5) 
        
    # Close the database connection
    conn.close()
         
