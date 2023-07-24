import sys

# Definir una función que recibe un error y un detalle del error
def error_message_detail(error, error_detail:sys):
    # Obtener la información de la traza del error
    _,_,exc_tb = error_detail.exc_info()
    # Obtener el nombre del archivo donde ocurrió el error
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Obtener el número de línea donde ocurrió el error
    line_number = exc_tb.tb_lineno
    # Obtener el mensaje del error
    error_message = str(error)
    # Formatear el mensaje de error con el nombre del archivo, el número de línea y el mensaje del error
    return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message[{error_message}]"


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