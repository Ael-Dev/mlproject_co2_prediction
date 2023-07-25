import sys 
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    # Definir el método que hace una predicción a partir de unas características
    def predict(self,features):
        try:
            # Definir las rutas de los archivos donde se guardan el modelo y el preprocesador
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")

            # Cargar el modelo y el preprocesador usando la función load_object
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            # Aplicar el preprocesador a las características para escalarlas
            data_scaled=preprocessor.transform(features)

            # Aplicar el modelo a las características escaladas para obtener las predicciones
            preds=model.predict(data_scaled)
            # Retornar las predicciones
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    # Definir el método constructor de la clase CustomData
    def __init__(self,
                 engine_size: float,
                 cylinders: float,
                 transmission: str,
                 fuel_type: str,
                 fuel_consumption_city: float,
                 fuel_consumption_hwy: float,
                 fuel_consumption_comb: float,
                 fuel_consumption_comb_mpg: float,
                 make_type: str,
                 vehicle_class_type: str):
        # Asignar los argumentos a los atributos de la instancia
        self.engine_size = engine_size
        self.cylinders = cylinders
        self.transmission = transmission
        self.fuel_type = fuel_type
        self.fuel_consumption_city = fuel_consumption_city
        self.fuel_consumption_hwy = fuel_consumption_hwy
        self.fuel_consumption_comb = fuel_consumption_comb
        self.fuel_consumption_comb_mpg = fuel_consumption_comb_mpg
        self.make_type = make_type
        self.vehicle_class_type = vehicle_class_type

    # Definir el método que devuelve los datos de la instancia como un dataframe de pandas
    def get_data_as_data_frame(self):
        try:
            # Crear un diccionario con los nombres y valores de los atributos
            custom_data_input_dict = {
                "engine_size": [self.engine_size],
                "cylinders": [self.cylinders],
                "transmission": [self.transmission],
                "fuel_type": [self.fuel_type],
                "fuel_consumption_city": [self.fuel_consumption_city],
                "fuel_consumption_hwy": [self.fuel_consumption_hwy],
                "fuel_consumption_comb": [self.fuel_consumption_comb],
                "fuel_consumption_comb_mpg": [self.fuel_consumption_comb_mpg],
                "make_type": [self.make_type],
                "vehicle_class_type": [self.vehicle_class_type],
            }
            # Retornar un dataframe de pandas a partir del diccionario
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)