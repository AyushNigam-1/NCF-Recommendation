import sys
from src.logging import logger

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        exc_type, exc_value, exc_tb = sys.exc_info()

        # Check if traceback exists
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None
    
    def __str__(self):
        # Handle case where no traceback information is available
        if self.lineno is None or self.file_name is None:
            return f"Error occurred: {self.error_message}"
        else:
            return f"Error occurred in python script name [{self.file_name}] line number [{self.lineno}] error message [{str(self.error_message)}]"

# if __name__ == '__main__':
#     try:
#         logger.logging.info("Enter the try block")
#         a = 1 / 0  # Division by zero will trigger an exception
#         print("This will not be printed", a)
#     except Exception as e:
#         raise CustomException(e, sys)  # Pass the exception and sys to the custom exception
