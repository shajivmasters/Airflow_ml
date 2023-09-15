##############################################################################
# Import the necessary modules
# #############################################################################
import pytest
from utils import load_data_into_db, map_city_tier, map_categorical_vars, interactions_mapping
import sqlite3
from significant_categorical_level import list_platform, list_medium, list_source
from  city_tier_mapping import city_tier_mapping
from constants import DB_FILE_NAME,DB_PATH,DATA_DIRECTORY,INTERACTION_MAPPING,INDEX_COLUMNS_TRAINING,INDEX_COLUMNS_INFERENCE,NOT_FEATURES,UNIT_TEST_DB_FILE_NAME
import sys 
import pandas as pd 
###############################################################################
# Write test cases for load_data_into_db() function
# ##############################################################################
def return_data(DB_PATH ,DB_FILE_NAME,tablename):
    print("Processing the file",DB_PATH + DB_FILE_NAME,tablename)
    conn = sqlite3.connect(DB_PATH + DB_FILE_NAME)
    query = "SELECT * FROM {}".format(tablename)  # Assuming you have a table named 'loaded_data'
    data = pd.read_sql_query(query, conn,index_col='created_date')
    conn.close()
    #return data.sort_index(axis=1,ascending=True)
    return data

def test_load_data_into_db():
    """_summary_
    This function checks if the load_data_into_db function is working properly by
    comparing its output with test cases provided in the db in a table named
    'loaded_data_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_get_data()

    """


    data = return_data(DB_PATH,DB_FILE_NAME,'loaded_data')

    data_to_compare = return_data(DB_PATH,UNIT_TEST_DB_FILE_NAME,'loaded_data_test_case')

    
    #df3   ={'A': [1, 2, 3], 'B': [4, 5, 7]}
    #data_to_compare = pd.DataFrame(df3)
    
    #print(data.head(5))
    #print(data_to_compare.head(5))
    try:
        pd.testing.assert_frame_equal(data, data_to_compare)
    except AssertionError as e:
        raise AssertionError("DataFrames are not the same.") from e
        
    print("All assertions passed!")



    

###############################################################################
# Write test cases for map_city_tier() function
# ##############################################################################
def test_map_city_tier():
    """_summary_
    This function checks if map_city_tier function is working properly by
    comparing its output with test cases provided in the db in a table named
    'city_tier_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_map_city_tier()

    """

    data = return_data(DB_PATH,DB_FILE_NAME,'city_tier_mapped')

    data_to_compare = return_data(DB_PATH,UNIT_TEST_DB_FILE_NAME,'city_tier_mapped_test_case')

    
    #df3   ={'A': [1, 2, 3], 'B': [4, 5, 7]}
    #data_to_compare = pd.DataFrame(df3)
    
    #print(data.head(5))
    #print(data_to_compare.head(5))
    try:
        pd.testing.assert_frame_equal(data, data_to_compare)
    except AssertionError as e:
        raise AssertionError("DataFrames are not the same.") from e
        
    print("All assertions passed!")

###############################################################################
# Write test cases for map_categorical_vars() function
# ##############################################################################    
def test_map_categorical_vars():
    """_summary_
    This function checks if map_cat_vars function is working properly by
    comparing its output with test cases provided in the db in a table named
    'categorical_variables_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'
    
    SAMPLE USAGE
        output=test_map_cat_vars()

    """    
    data = return_data(DB_PATH,DB_FILE_NAME,'categorical_variables_mapped')

    data_to_compare = return_data(DB_PATH,UNIT_TEST_DB_FILE_NAME,'categorical_variables_mapped_test_case')

    data = data.sort_index()
    data_to_compare = data_to_compare.sort_index()
    #df3   ={'A': [1, 2, 3], 'B': [4, 5, 7]}
    #data_to_compare = pd.DataFrame(df3)
    #print(data.head(5))
    #print(data_to_compare.head(5))

    try:
        pd.testing.assert_frame_equal(data, data_to_compare,check_like=True)
    except AssertionError as e:
        raise AssertionError("DataFrames are not the same.") from e

    print("All assertions passed!")   

###############################################################################
# Write test cases for interactions_mapping() function
# ##############################################################################    
def test_interactions_mapping():
    """_summary_
    This function checks if test_column_mapping function is working properly by
    comparing its output with test cases provided in the db in a table named
    'interactions_mapped_test_case'

    INPUTS
        DB_FILE_NAME : Name of the database file 'utils_output.db'
        DB_PATH : path where the db file should be present
        UNIT_TEST_DB_FILE_NAME: Name of the test database file 'unit_test_cases.db'

    SAMPLE USAGE
        output=test_column_mapping()

    """ 
    data = return_data(DB_PATH,DB_FILE_NAME,'interactions_mapped')

    data_to_compare = return_data(DB_PATH,UNIT_TEST_DB_FILE_NAME,'interactions_mapped_test_case')

    data = data.sort_index(axis=1,ascending=True)
    data_to_compare = data_to_compare.sort_index(axis=1,ascending=True)
    #df3   ={'A': [1, 2, 3], 'B': [4, 5, 7]}
    #data_to_compare = pd.DataFrame(df3)
    #print(data.head(5))
    #print(data_to_compare.head(5))

    try:
        pd.testing.assert_frame_equal(data, data_to_compare)
        
    except AssertionError as e:
        raise AssertionError("DataFrames are not the same.") from e

    print("All assertions passed!")