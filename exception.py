import sys

def error_message_detail(error, error_detail: sys):
    # Extract traceback information
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format the error message
    error_message = (
        f"Error occurred in script: [{file_name}], "
        f"line: [{exc_tb.tb_lineno}], "
        f"message: [{str(error)}]"
    )
    
    return error_message

class AFException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize base exception
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message