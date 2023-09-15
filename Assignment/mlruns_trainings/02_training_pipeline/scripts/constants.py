DB_PATH = "/home/airflow/dags/Lead_scoring_data_pipeline/"
DB_FILE_NAME = "lead_scoring_data_cleaning.db"


DB_FILE_MLFLOW = "/home/Shajiv_Assignment/Assignment/02_training_pipeline/scripts/Lead_scoring_mlflow_production.db"

TRACKING_URI = "http://0.0.0.0:6006"
EXPERIMENT = "Lead_Scoring_Training_Pipeline"


# model config imported from pycaret experimentation
model_config = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'importance_type': 'split',
        'learning_rate': 0.01,  # Reduced learning rate
        'max_depth': 10,  # Increased max depth
        'min_child_samples': 50,  # Reduced min_child_samples
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 200,
        'n_jobs': -1,
        'num_leaves': 50,  # Increased num_leaves
        'objective': None,
        'random_state': 42,
        'reg_alpha': 0.1,  # Added regularization
        'reg_lambda': 0.1,  # Added regularization
        'silent': 'warn',
        'subsample': 0.9,  # Reduced subsample
        'subsample_for_bin': 300000,
        'subsample_freq': 0
    }

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = ['city_tier', 'total_leads_droppped', 'referred_lead',
        'assistance_interaction', 'career_interaction',
       'payment_interaction', 'social_interaction', 'syllabus_interaction',
       'first_platform_c_Level0', 'first_platform_c_Level1',
       'first_platform_c_Level2', 'first_platform_c_Level3',
       'first_platform_c_Level7', 'first_platform_c_Level8',
       'first_platform_c_others', 'first_utm_medium_c_Level0',
       'first_utm_medium_c_Level10', 'first_utm_medium_c_Level11',
       'first_utm_medium_c_Level13', 'first_utm_medium_c_Level15',
       'first_utm_medium_c_Level16', 'first_utm_medium_c_Level2',
       'first_utm_medium_c_Level20', 'first_utm_medium_c_Level26',
       'first_utm_medium_c_Level3', 'first_utm_medium_c_Level30',
       'first_utm_medium_c_Level33', 'first_utm_medium_c_Level4',
       'first_utm_medium_c_Level43', 'first_utm_medium_c_Level5',
       'first_utm_medium_c_Level6', 'first_utm_medium_c_Level8',
       'first_utm_medium_c_Level9', 'first_utm_medium_c_others',
       'first_utm_source_c_Level0', 'first_utm_source_c_Level14',
       'first_utm_source_c_Level16', 'first_utm_source_c_Level2',
       'first_utm_source_c_Level4', 'first_utm_source_c_Level5',
       'first_utm_source_c_Level6', 'first_utm_source_c_Level7',
       'first_utm_source_c_others']
# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = [ 'first_platform_c', 'first_utm_medium_c' ,'first_utm_source_c']
