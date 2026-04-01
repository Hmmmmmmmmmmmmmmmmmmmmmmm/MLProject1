import os
import sys

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_training import ModelTrainConfig, ModelTrainer

log = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        basic method to accept, save and split data
        (stores data in artifacts directory)
        returns train and test paths
        '''
        log.info("Entered the Data Ingestion Component")
        try:
            # below part can be replaced with any
            # form of data acceptance/ reading aka
            # method be it mongo DB or any other database
            # df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            df = pd.read_csv(os.path.join("notebook", "data", "StudentsPerformance.csv"))
            log.info('Exported/ Read the Dataset')


            # Could improve by versioning artifacts like this:
            # from datetime import datetime
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # base_dir = os.path.join("artifact", timestamp)
            # os.makedirs(base_dir, exist_ok=True)


            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path
                ),exist_ok=True
            )
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )
            log.info("Train Test Spilt initiated (RS:42)")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            log.info("Data ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.train_data_path,
            )
        except Exception as e:
            log.error(f"Error occurred: {e}", exc_info=True)
            raise CustomException(e,sys)



if __name__ == '__main__':
    print('MEWMEW')
    log.info("TESTING data transformation via data ingestion by passing test and train data")
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, p = data_transformation.initiate_data_transformer(train_data, test_data)
    model_trainer = ModelTrainer()
    print("Initiating model Training")
    print(model_trainer.initiate_model_trainer(
        train_array=train_arr,
        test_array=test_arr,
        preprocessor_path=p
    ))
    print("MEWMEWMEMWEMWWEMWEM")