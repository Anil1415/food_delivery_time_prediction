import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import time
import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import handle_column_name

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('data transform initiated')
            numerical_columns = ['delivery_person_age','delivery_person_ratings','vehicle_condition','multiple_deliveries','preparation_time_min','distance_to_delivery_km']
            categorical_columns = ['weather_conditions','road_traffic_density','type_of_vehicle','festival','city']

            # define which columns should be ordinal_encoded and which should be scaled

            # categorical_columns = X.select_dtypes(include = 'object').columns
            # numerical_columns = X.select_dtypes(exclude = 'object').columns

            # define custom ranking for each ordinal variable

            weather_category = ['Sunny','Fog','Cloudy','Windy','Sandstorms','Stormy']
            road_traffic_category = ['Low','Medium','High', 'Jam']
            vehilcle_type = ['bicycle','electric_scooter','scooter','motorcycle']
            festival_category = ['No', 'Yes']
            city_category = ['Urban','Semi-Urban','Metropolitian']

            logging.info('pipeline initiated')

            ## Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                       ('imputer', SimpleImputer(strategy='median')),
                       ('scalar', StandardScaler())
                       ]
                    )
            ##  categorical_pipeline

            categorical_pipeline = Pipeline(
                steps= [
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder', OrdinalEncoder(categories=[weather_category,road_traffic_category,vehilcle_type,festival_category,city_category])),
                        ('scaler', StandardScaler())
                        ]
                    )
            
            ## preprocessor

            preprocessor = ColumnTransformer(
                            [
                             ('numerical_pipeline', numerical_pipeline, [i for i in numerical_columns]),
                             ('categorical_pipeline', categorical_pipeline, [i for i in categorical_columns])
                            ]
                        )   
            return preprocessor
            logging.info('Pipeline completed')

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:

            train_df =pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dateframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'time_taken_min'
            drop_columns = [target_column_name,
                            'id','delivery_person_id' ,'restaurant_latitude','restaurant_longitude','delivery_location_latitude','delivery_location_longitude','order_date','time_orderd','time_order_picked','type_of_order']


            input_feature_train_df = train_df.drop(columns = drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Transforming using preprocessor obj

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and test datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('preprocessor pickle file saved')

            return(train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info('Exception occured in the initiate_datatransformation')
            raise CustomException(e, sys)
       






""""

---
             # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # Read train and test data

            # data  = pd.read_csv(raw_data_path)

            logging.info('Reading raw data completed')
            logging.info(f"Raw Data Frame Head: \n {data.head().to_string()}")

            logging.info("Data Cleaning, EDA and Feature Engineering started")
            # Handling column name
            obj_handling = handle_column_name()
            obj_handling.lower_column_names()
            obj_handling.fill_column_names()


            if data.duplicated().any():
                 # drop duplicates
                 data = data.drop_duplicates()
            else:
                 data

            

            # pattern match time format are kept and others are dropped


            data = data[~((~data.time_orderd.str.contains(r'[0-9]{2}:[0-9]{2}', na = False)) | (~data.time_order_picked.str.contains(r'[0-9]{2}:[0-9]{2}', na=False)))]
            data['time_order_picked'] = data.time_order_picked.str.replace('24','00')   

            # converting to datetime64[ns] format
            data['time_orderd'] = pd.to_datetime(data['order_date']+ ' ' +data['time_orderd'])

            data['time_order_picked'] = pd.to_datetime(data['order_date'] + ' '+data['time_order_picked'])
            # 45581   2022-11-03 00:05:00 # picked_up time
            # 45581   2022-11-03 23:50:00 # ordered time

            ###  date need to be changed even the diff getting 15 min

            difference = (pd.to_datetime(data['time_order_picked']) - pd.to_datetime(data['time_orderd'])).dt.seconds/60 

            data.insert(loc=19, column='preparation_time_min', value=difference)
            distance_km = haversine_np(data['restaurant_longitude'], data['restaurant_latitude'], data['delivery_location_longitude'], data['delivery_location_latitude'])
            data.insert(loc=20, column='distance_to_delivery_km', value=distance_km.round(2))

            train_df =pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dateframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_objects()

            target_column_name = 'time_taken_min'
            drop_columns = [target_column_name, 'id']





"""