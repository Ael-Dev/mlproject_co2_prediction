import os
import sys
#sys.path.insert(0, '../src')
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split

# Importar el módulo dataclass para crear clases con atributos de datos
from dataclasses import dataclass

# Definir una clase DataIngestionConfig con los atributos raw_data_path, train_data_path y test_data_path con valores por defecto
@dataclass
class DataIngestionConfig:
    # Establecer las rutas de la carpeta donde se almacenaran los datos 
    # y el nombre con el q se guardaran
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")

# Definir una clase DataIngestion 
class DataIngestion:
    # Definir el método constructor que crea un objeto DataIngestionConfig y lo asigna al atributo ingestion_config
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Definir el método initiate_data_ingestion que lee el archivo de datos y crea el directorio para los archivos de entrenamiento y prueba
    def initiate_data_ingestion(self):
        # Registrar un mensaje de información indicando que se ingresó al método de ingestión de datos
        logging.info("Entered the data ingestion method or component")
        try:
            # Leer los datos que puede encontrarse en una cualquier fuente(BDs) y formato (csv, data, etc) 
            path_raw_data = "C:/Users/Alex/Desktop/R/machine learning/10.Projects/mlproject_co2_prediction/notebook/data/DATA_CLEANED.csv"
            df = pd.read_csv(path_raw_data)
            # Registrar un mensaje de información indicando que se leyó el conjunto de datos
            logging.info("Read the dataset")

            # ------------------------------------------------------------------------------------------------
            # Crear el directorio para los archivos de entrenamiento y prueba si no existe
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # Guardar en la ruta establecida el dataset obtenido desde otras fuentes
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # ------------------------------------------------------------------------------------------------
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=123)
            
            # ------------------------------------------------------------------------------------------------
            # Guardar el dataset se entrenamiento en la ruta establecida
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # Guardar el dataset se test en la ruta establecida
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed!")

            # ------------------------------------------------------------------------------------------------
            # Retornar las rutas de donde se almacenaron los datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Manejar la excepción si ocurre algún error
        except Exception as e:
            # Lanzar una excepción personalizada con el error  y el sistema  como argumentos
            raise CustomException(e, sys)


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------
    # 1. Crear un objeto de la clase creada DataIngestion
    obj = DataIngestion()
    # 1.1. Inicializamos la ingestion de datos 
    # y recuperamos la rutas donde se guardaron los datos obtenidos
    train_data,test_data = obj.initiate_data_ingestion()
    
    # ------------------------------------------------------------------------------------------------
    # 2. Crea un objeto de la clase creada DataTransformation
    data_transformation = DataTransformation()
    # 2.1. Inicializamos la transformation de datos 
    # y recuperamos la data transformada
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    # ------------------------------------------------------------------------------------------------
    # Crear un objeto de la clase ModelTrainer
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

