##############################################################################
# Import necessary modules and files
# #############################################################################


import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from significant_categorical_level import list_platform, list_medium, list_source
from  city_tier_mapping import city_tier_mapping
from constants import DB_FILE_NAME,DB_PATH,DATA_DIRECTORY,INTERACTION_MAPPING,INDEX_COLUMNS_TRAINING,INDEX_COLUMNS_INFERENCE,NOT_FEATURES
import sys 

###############################################################################
# Define the function to build database
# ##############################################################################

def build_dbs():
    '''
    This function checks if the db file with specified name is present 
    in the /Assignment/01_data_pipeline/scripts folder. If it is not present it creates 
    the db file with the given name at the given path. 


    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should exist  


    OUTPUT
    The function returns the following under the given conditions:
        1. If the file exists at the specified path
                prints 'DB Already Exists' and returns 'DB Exists'

        2. If the db file is not present at the specified loction
                prints 'Creating Database' and creates the sqlite db 
                file at the specified path with the specified name and 
                once the db file is created prints 'New DB Created' and 
                returns 'DB created'


    SAMPLE USAGE
        build_dbs()
    '''
    # Check if the database file already exists at the specified path
    db_file_path = os.path.join(DB_PATH, DB_FILE_NAME)

    if os.path.exists(db_file_path):
        print('DB Already Exists')
        return 'DB Exists'
    else:
        print('Creating Database')
        # Create the SQLite database file
        conn = sqlite3.connect(db_file_path)
        conn.close()
        print('New DB Created')
        return 'DB created'
    

###############################################################################
# Define function to load the csv file to the database
# ##############################################################################

def load_data_into_db():
    '''
    Thie function loads the data present in data directory into the db
    which was created previously.
    It also replaces any null values present in 'toal_leads_dropped' and
    'referred_lead' columns with 0.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        DATA_DIRECTORY : path of the directory where 'leadscoring.csv' 
                        file is present
        

    OUTPUT
        Saves the processed dataframe in the db in a table named 'loaded_data'.
        If the table with the same name already exsists then the function 
        replaces it.


    SAMPLE USAGE
        load_data_into_db()
    '''
    data_file = "leadscoring_test.csv"
    print(DATA_DIRECTORY + data_file)
    # Load data from CSV file
    data = pd.read_csv(DATA_DIRECTORY + data_file)
    #print(data.columns)
    #print(data.head())
    if data_file == "leadscoring.csv":
        data = data.drop(data.columns[0], axis=1)
    #print(data.columns)
    # Replace null values in specified columns with 0
    #data['toal_leads_dropped'].fillna(0, inplace=True)
    data['referred_lead'].fillna(0, inplace=True)

    # Connect to the database and save the data
    print(DB_PATH + DB_FILE_NAME)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    data.to_sql('loaded_data', conn, if_exists='replace', index=False)
    conn.close()

###############################################################################
# Define function to map cities to their respective tiers
# ##############################################################################

    
def map_city_tier():
    '''
    This function maps all the cities to their respective tier as per the
    mappings provided in the city_tier_mapping.py file. If a
    particular city's tier isn't mapped(present) in the city_tier_mapping.py 
    file then the function maps that particular city to 3.0 which represents
    tier-3.


    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be
        city_tier_mapping : a dictionary that maps the cities to their tier

    
    OUTPUT
        Saves the processed dataframe in the db in a table named
        'city_tier_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_city_tier()

    '''
    # Create a DataFrame from the 
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    query = "SELECT * FROM loaded_data"  # Assuming you have a table named 'loaded_data'
    data = pd.read_sql_query(query, conn)
    conn.close()

    # Map cities to tiers based on the provided dictionary
    ###data['city_tier'] = data['city'].map(city_tier_mapping).fillna(3.0)  # Default to tier-3 if not mapped
    data["city_tier"] = data["city_mapped"].map(city_tier_mapping)
    data["city_tier"] = data["city_tier"].fillna(3.0)
    data = data.drop(['city_mapped'], axis = 1)
    
    # Connect to the database and save the mapped data
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    data.to_sql('city_tier_mapped', conn, if_exists='replace', index=False)
    conn.close()

###############################################################################
# Define function to map insignificant categorial variables to "others"
# ##############################################################################


def map_categorical_vars():
    '''
    This function maps all the insignificant variables present in 'first_platform_c'
    'first_utm_medium_c' and 'first_utm_source_c'. The list of significant variables
    should be stored in a python file in the 'significant_categorical_level.py' 
    so that it can be imported as a variable in utils file.
    

    INPUTS
        DB_FILE_NAME : Name of the database file
        DB_PATH : path where the db file should be present
        list_platform : list of all the significant platform.
        list_medium : list of all the significat medium
        list_source : list of all rhe significant source

        **NOTE : list_platform, list_medium & list_source are all constants and
                 must be stored in 'significant_categorical_level.py'
                 file. The significant levels are calculated by taking top 90
                 percentils of all the levels. For more information refer
                 'data_cleaning.ipynb' notebook.
  

    OUTPUT
        Saves the processed dataframe in the db in a table named
        'categorical_variables_mapped'. If the table with the same name already 
        exsists then the function replaces it.

    
    SAMPLE USAGE
        map_categorical_vars()
    '''
    # Create a DataFrame with categorical data (assuming you have these columns in your database)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    query = "SELECT * FROM city_tier_mapped"  # Assuming you have a table named 'loaded_data'
    data = pd.read_sql_query(query, conn)
    conn.close()

    print(list_platform)
    print(list_medium)
    print(list_source)
    
    # all the levels below 90 percentage are assgined to a single level called others
    new_df = data[~data['first_platform_c'].isin(list_platform)] # get rows for levels which are not present in list_platform
    new_df['first_platform_c'] = "others" # replace the value of these levels to others
    old_df = data[data['first_platform_c'].isin(list_platform)] # get rows for levels which are present in list_platform
    df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe


    # all the levels below 90 percentage are assgined to a single level called others
    new_df = df[~df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are not present in list_medium
    new_df['first_utm_medium_c'] = "others" # replace the value of these levels to others
    old_df = df[df['first_utm_medium_c'].isin(list_medium)] # get rows for levels which are present in list_medium
    df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe


    # all the levels below 90 percentage are assgined to a single level called others
    new_df = df[~df['first_utm_source_c'].isin(list_source)] # get rows for levels which are not present in list_source
    new_df['first_utm_source_c'] = "others" # replace the value of these levels to others
    old_df = df[df['first_utm_source_c'].isin(list_source)] # get rows for levels which are present in list_source
    df = pd.concat([new_df, old_df]) # concatenate new_df and old_df to get the final dataframe
    
    df['total_leads_droppped'] = df['total_leads_droppped'].fillna(0)
    df['referred_lead'] = df['referred_lead'].fillna(0)
    # Connect to the database and save the mapped data
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    df.to_sql('categorical_variables_mapped', conn, if_exists='replace', index=False)
    conn.close()


##############################################################################
# Define function that maps interaction columns into 4 types of interactions
# #############################################################################
def interactions_mapping():
    '''
    This function maps the interaction columns into 4 unique interaction columns
    These mappings are present in 'interaction_mapping.csv' file. 


    INPUTS
        DB_FILE_NAME: Name of the database file
        DB_PATH : path where the db file should be present
        INTERACTION_MAPPING : path to the csv file containing interaction's
                                   mappings
        INDEX_COLUMNS_TRAINING : list of columns to be used as index while pivoting and
                                 unpivoting during training
        INDEX_COLUMNS_INFERENCE: list of columns to be used as index while pivoting and
                                 unpivoting during inference
        NOT_FEATURES: Features which have less significance and needs to be dropped
                                 
        NOTE : Since while inference we will not have 'app_complete_flag' which is
        our label, we will have to exculde it from our features list. It is recommended 
        that you use an if loop and check if 'app_complete_flag' is present in 
        'categorical_variables_mapped' table and if it is present pass a list with 
        'app_complete_flag' column, or else pass a list without 'app_complete_flag'
        column.

    
    OUTPUT
        Saves the processed dataframe in the db in a table named 
        'interactions_mapped'. If the table with the same name already exsists then 
        the function replaces it.
        
        It also drops all the features that are not requried for training model and 
        writes it in a table named 'model_input'

    
    SAMPLE USAGE
        interactions_mapping()
    '''
    
    # Create a DataFrame with categorical data (assuming you have these columns in your database)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    query = "SELECT * FROM categorical_variables_mapped"  # Assuming you have a table named 'loaded_data'
    data = pd.read_sql_query(query, conn)
    conn.close()
    #print("Before : {}".format(len(data)))
    data = data.drop_duplicates()
    #print("After  : {}".format(len(data)))
          
    if INDEX_COLUMNS_INFERENCE and INDEX_COLUMNS_TRAINING:
        sys.exit("INDEX_COLUMNS_TRAINING and INDEX_COLUMNS_TRAINING both has value. Please provide either one depending training or inference")
          
    
    # Read the interaction mappings from the CSV file
    interaction_map_file = "interaction_mapping.csv"
    df_event_mapping = pd.read_csv(interaction_map_file)
    #print(df_event_mapping)
    # unpivot the interaction columns and put the values in rows
    if INDEX_COLUMNS_TRAINING: 
        df_unpivot = pd.melt(data, id_vars=INDEX_COLUMNS_TRAINING, var_name='interaction_type', value_name='interaction_value')
        # handle the nulls in the interaction value column
        df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)

        df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
    
        #dropping the interaction type column as it is not needed
        df = df.drop('interaction_type', axis=1)
        # pivoting the interaction mapping column values to individual columns in the dataset
        df_pivot = df.pivot_table(
            values='interaction_value', index=INDEX_COLUMNS_TRAINING, columns='interaction_mapping', aggfunc='sum')
        
        
    if INDEX_COLUMNS_INFERENCE:
        if 'app_complete_flag' in data.columns:
            data.drop(column_to_check, axis=1, inplace=True)
    
        # unpivot the interaction columns and put the values in rows
        
        df_unpivot = pd.melt(data, id_vars=INDEX_COLUMNS_INFERENCE, var_name='interaction_type', value_name='interaction_value')
        # handle the nulls in the interaction value column
        df_unpivot['interaction_value'] = df_unpivot['interaction_value'].fillna(0)
        df = pd.merge(df_unpivot, df_event_mapping, on='interaction_type', how='left')
        #dropping the interaction type column as it is not needed
        df = data.drop(['interaction_type'], axis=1)
        # pivoting the interaction mapping column values to individual columns in the dataset
        df_pivot = df.pivot_table(
        values='interaction_value', index=INDEX_COLUMNS_INFERENCE, columns='interaction_mapping', aggfunc='sum')
        
    df_pivot = df_pivot.reset_index()
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    df_pivot.to_sql('interactions_mapped', conn, if_exists='replace', index=False)
    df_pivot.to_csv('Clean_model_unit_test_input.csv',index=False)
    conn.close()
    
    #print(df_pivot.columns)

    df_model = df_pivot.drop(NOT_FEATURES,axis=1)
    #print(df_model.columns)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    df_model.to_sql('model_input', conn, if_exists='replace', index=False)
    conn.close()