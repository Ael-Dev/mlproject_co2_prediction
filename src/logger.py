import logging
import os
from datetime import datetime

# Definir el formato del nombre del archivo de log con la fecha y hora actual
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Definir el directorio de los logs con el directorio actual y el nombre del archivo de log
logs_path =  os.path.join(os.getcwd(), 'logs', LOG_FILE)
# Crear el directorio de los logs si no existe
os.makedirs(logs_path,exist_ok=True)

# Definir la ruta completa del archivo de log con el directorio y el nombre del archivo de log
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configurar el logging con el nombre del archivo, el formato del mensaje y el nivel de logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s -%(message)s",
    level=logging.INFO,
)
