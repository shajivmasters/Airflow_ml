'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3
import json

import os
import logging

from datetime import datetime
from constants import DB_PATH,DB_FILE_NAME,FEATURES_TO_ENCODE,ONE_HOT_ENCODED_FEATURES,TRACKING_URI,EXPERIMENT,DB_FILE_MLFLOW,model_config,PREDICTION_LOG,INPUT_FEATURE_LOG,MODEL_NAME,STAGE

#print("DB_PATH : {} \nDB_FILE_NAME: {}\nFEATURES_TO_ENCODE: {}\nONE_HOT_ENCODED_FEATURES: {} \nTRACKING_URI:{}\nEXPERIMENT:{}\nDB_FILE_MLFLOW:{}\nmodel_config : {}\n".format(DB_PATH,DB_FILE_NAME,FEATURES_TO_ENCODE,ONE_HOT_ENCODED_FEATURES,TRACKING_URI,EXPERIMENT,DB_FILE_MLFLOW,json.dumps(model_config,indent=4)))

import logging 
import time


inputlogger = logging.getLogger("inputlogger")
inputlogger.setLevel(logging.INFO)

inputlogger_fh = logging.FileHandler(INPUT_FEATURE_LOG)
inputlogger_fh.setLevel(logging.INFO) 


# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
inputlogger_fh.setFormatter(formatter)

# Add the file handler to logger1
inputlogger.addHandler(inputlogger_fh)


predictionlogger = logging.getLogger("predictionlogger")
predictionlogger.setLevel(logging.INFO)

predictionlogger_fh = logging.FileHandler(INPUT_FEATURE_LOG)
predictionlogger_fh.setLevel(logging.INFO) 


# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
predictionlogger_fh.setFormatter(formatter)

# Add the file handler to logger1
predictionlogger.addHandler(inputlogger_fh)


###############################################################################
# Define the function to train the model
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
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''
    
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    query = "SELECT * FROM interactions_mapped" 
    data = pd.read_sql_query(query, conn)
    #print(data.head(5))
    #print("Before encoding data Columns ({}): {}".format(len(data.columns),','.join(data.columns)))
    print("Before encoding data shape : {}".format(data.shape))
    if 'app_complete_flag' in data.columns:
        print(data.columns)
        data.drop(['app_complete_flag'],axis=1,inplace=True)
        print("After droping the 'app_complete_flag' column ({}) :{}".format(len(data.columns),','.join(data.columns)))
    for f in FEATURES_TO_ENCODE:
        if(f in data.columns):
            encoded = pd.get_dummies(data[f])
            encoded = encoded.add_prefix(f + '_')
            data = pd.concat([data,encoded],axis=1)
        else:
            print("Feature {} not found".format(f))
    data.drop(columns=FEATURES_TO_ENCODE,inplace=True)
    #print("After encoding data Columns  ({}): {}".format(len(data.columns),','.join(data.columns)))
    
    print("After encoding data shape : {}".format(data.shape))
    # Save the encoded features into a new table named 'features
    db_file_path = os.path.join(DB_PATH,DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    df_encoded = data[ONE_HOT_ENCODED_FEATURES]
    df_encoded.to_sql('Inference_features', conn, if_exists='replace', index=False)
    #print(df_encoded)
    # Close the database connection
    
    data.to_sql('data', conn, if_exists='replace', index=False)
    # Close the database connection again
    conn.close()
    
###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def get_models_prediction():
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    cnx = sqlite3.connect(DB_FILE_MLFLOW)
    mlflow.set_tracking_uri("http://0.0.0.0:6006")
    logged_model = "{}/{}".format(MODEL_NAME,STAGE)
    print(logged_model)
    # Load model as a PyFuncModel.
    loaded_model = mlflow.sklearn.load_model(logged_model)
    # Predict on a Pandas DataFrame.
    print("Loadded Model Information :{}".format(loaded_model))
    
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    print("Loading the data from {}".format(db_file_path))
    query = "SELECT * FROM Inference_features" 
    X = pd.read_sql_query(query, conn)
    conn.close()
    #print("The Inference data")
    #print(X.head(2))
    predictions_proba = loaded_model.predict_proba(X)
    predictions = loaded_model.predict(X)
    pred_df = X.copy()    
    pred_df['churn'] = predictions
    pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
    #print("The Prediction dataframe")
    #print(pred_df.head(2))

    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    pred_df.to_sql('Final_Predictions', conn, if_exists='replace', index=False)
    conn.close()
    print("Predictions are done and save in Final_Predictions Table")
    
###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check():
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)
    conn = sqlite3.connect(db_file_path)
    print("Loading the data from {}".format(db_file_path))
    query = "SELECT * FROM Final_Predictions" 
    prediction_data = pd.read_sql_query(query, conn)
    conn.close()
    prediction_result = prediction_data[["churn","Prob of Not Churn","Prob of Churn"]]

    prediction_result.to_csv('prediction_distribution.txt', index=False,mode='w')
    print("Please check the prediction_distribution.txt file")
    
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################
   

def input_features_check():
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)

    # Check if the database file exists
    if not os.path.isfile(db_file_path):
        print(f"Database file '{DB_FILE_NAME}' does not exist in the directory '{DB_PATH}'.")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        # Fetch the table schema from the 'model_input' table
        cursor.execute("PRAGMA table_info('Inference_features')")
        table_columns = [row[1] for row in cursor.fetchall()]
        #print(table_columns)
        # Check if the table columns match the provided model_input_schema
        if sorted(table_columns) == sorted(ONE_HOT_ENCODED_FEATURES):
            print("All the models input are present")
            inputlogger.info("All the models input are present")
        else:
            print(set(table_columns) - set(ONE_HOT_ENCODED_FEATURES))
            print("Some of the models inputs are missing")
            inputlogger.error("Some of the models inputs are missing")

    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    finally:
        if conn:
            conn.close()
