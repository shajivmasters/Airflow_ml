"""
Import necessary modules
############################################################################## 
"""

import pandas as pd
from constants import DATA_DIRECTORY,DB_PATH,DB_FILE_NAME
from schema import raw_data_schema,model_input_schema
import sys 
import os 
import csv
import sqlite3
from sqlite3 import Error

###############################################################################
# Define function to validate raw data's schema
############################################################################### 

def return_data(DB_PATH ,DB_FILE_NAME,tablename):
    print("Processing the file",DB_PATH + DB_FILE_NAME,tablename)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    query = "SELECT * FROM {}".format(tablename)  # Assuming you have a table named 'loaded_data'
    data = pd.read_sql_query(query, conn,index_col='created_date')
    conn.close()
    #return data.sort_index(axis=1,ascending=True)
    return data


def raw_data_schema_check():
    '''
    This function check if all the columns mentioned in schema.py are present in
    leadscoring.csv file or not.

   
    INPUTS
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        raw_data_schema : schema of raw data in the form oa list/tuple as present 
                          in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Raw datas schema is in line with the schema present in schema.py' 
        else prints
        'Raw datas schema is NOT in line with the schema present in schema.py'

    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    # Define the path to the CSV file
    csv_file_path = os.path.join(DATA_DIRECTORY, 'leadscoring.csv')
    
    # Check if the CSV file exists
    if not os.path.isfile(csv_file_path):
        print(f"File 'leadscoring.csv' does not exist in the directory '{DATA_DIRECTORY}'.")
        return

    # Read the header row from the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)

    if header is None:
        print(f"The CSV file '{csv_file_path}' is empty.")
        return
    header = header[1:] # This is because there is a index without the column name in the source 
    #print(header)
    
    
    #print("_____")
    #print(raw_data_schema)
    # Check if the header matches the schema
    if header == raw_data_schema:
        print('Raw data schema is in line with the schema present in schema.py')
    else:
        print('Raw data schema is NOT in line with the schema present in schema.py')


###############################################################################
# Define function to validate model's input schema
############################################################################### 

def model_input_schema_check():
    '''
    This function check if all the columns mentioned in model_input_schema in 
    schema.py are present in table named in 'model_input' in db file.

   
    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        model_input_schema : schema of models input data in the form oa list/tuple
                          present as in 'schema.py'

    OUTPUT
        If the schema is in line then prints 
        'Models input schema is in line with the schema present in schema.py'
        else prints
        'Models input schema is NOT in line with the schema present in schema.py'
    
    SAMPLE USAGE
        raw_data_schema_check
    '''
    
        # Combine the DB_PATH and DB_FILE_NAME to get the full database file path
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
        cursor.execute("PRAGMA table_info('model_input')")
        table_columns = [row[1] for row in cursor.fetchall()]
        #print(table_columns)
        # Check if the table columns match the provided model_input_schema
        if sorted(table_columns) == sorted(model_input_schema):
            print("Model input schema is in line with the schema present in schema.py")
        else:
            print("Model input schema is NOT in line with the schema present in schema.py")

    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    finally:
        if conn:
            conn.close()



    
    
