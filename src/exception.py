from src.logger import logging
import sys

# Definir una función que recibe un error y un detalle del error
def error_message_detail(error, error_detail:sys):
    # Obtener la información de la traza del error
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  )
    return error_message

# Definir una clase de excepción personalizada que hereda de Exception
class CustomException(Exception):
    # Definir el método constructor que recibe un mensaje de error y un detalle del error
    def __init__(self, error_message, error_detail:sys):
        # Llamar al método constructor de la clase base con el mensaje de error
        super().__init__(error_message)
        # Asignar el atributo error_message con el resultado de la función error_message_detail
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    # Definir el método que devuelve la representación del objeto como una cadena
    def __str__(self):
        # Devolver el atributo error_message
        return self.error_message