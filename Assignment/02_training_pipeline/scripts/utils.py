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
import lightgbm as lgb
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
#from Lead_scoring_training_pipeline.constants import *
from constants import DB_PATH,DB_FILE_NAME,FEATURES_TO_ENCODE,ONE_HOT_ENCODED_FEATURES,TRACKING_URI,EXPERIMENT,DB_FILE_MLFLOW,model_config

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
    print("Experiment name : {}".format(EXPERIMENT))
    print("MLFlow DB  name : {}".format(DB_FILE_MLFLOW))
         
    conn = sqlite3.connect(DB_FILE_MLFLOW)
    print(dir(mlflow))

    #print(mlflow.get_artifact_uri())
    #mlflow.end_run()
    mlflow.set_tracking_uri("http://0.0.0.0:6006")  
    experiment_name = f"{EXPERIMENT}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        # Try to create the experiment
        mlflow.create_experiment(experiment_name,artifact_location="/home/Shajiv_Assignment/Assignment/02_training_pipeline/scripts/mlruns_trainings/")
        print(f"Created experiment: {experiment_name}")
        
    except mlflow.exceptions.MlflowException as e:
            # Handle other MlflowExceptions if necessary
            print(f"Error: {str(e)}")

    #print(mlflow.get_artifact_uri())
    #experiment = mlflow.get_experiment(experiment_name)
    #print(f"Name: {experiment.name}")
    #print(f"Experiment_id: {experiment.experiment_id}")
    #print(f"Artifact Location: {experiment.artifact_location}")
    #print(f"Tags: {experiment.tags}")
    #print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    #print(f"Creation timestamp: {experiment.creation_time}")  
    
    mlflow.set_experiment(experiment_name)
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    cnx  = sqlite3.connect(db_file_path)
    X = pd.read_sql('select * from features', cnx)
    y = pd.read_sql('select * from target', cnx)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



    #Model Training
    runname = 'LightGB_{}'.format(experiment_name)
    with mlflow.start_run(run_name=runname) as run:
     
        clf = lgb.LGBMClassifier()
        clf.set_params(**model_config)  # Coming from the constants.py
        clf.fit(X_train, y_train)
        print(clf)
        mlflow.sklearn.log_model(sk_model=clf,artifact_path="models", registered_model_name=runname)
        mlflow.log_params(model_config)    

        # predict the results on training dataset
        y_pred=clf.predict(X_test)

        # # view accuracy
        # acc=accuracy_score(y_pred, y_test)
        # conf_mat = confusion_matrix(y_pred, y_test)
        # mlflow.log_metric('test_accuracy', acc)
        # mlflow.log_metric('confustion matrix', conf_mat)


        #Log metrics
        acc=accuracy_score(y_pred, y_test)
        conf_mat = confusion_matrix(y_pred, y_test)
        precision = precision_score(y_pred, y_test,average= 'macro')
        recall = recall_score(y_pred, y_test, average= 'macro')
        f1 = f1_score(y_pred, y_test, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
        class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)
        auc_value = roc_auc_score(y_pred, y_test)
        

        mlflow.log_metric('AUC', auc_value)
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision_0", class_zero[0])
        mlflow.log_metric("Precision_1", class_one[0])
        mlflow.log_metric("Recall_0", class_zero[1])
        mlflow.log_metric("Recall_1", class_one[1])
        mlflow.log_metric("f1_0", class_zero[2])
        mlflow.log_metric("f1_1", class_one[2])
        mlflow.log_metric("False Negative", fn)
        mlflow.log_metric("True Negative", tn)
        # mlflow.log_metric("f1", f1_score)

        runID = run.info.run_uuid
        print("Inside MLflow Run with id {}".format(runID))
        mlflow.end_run()