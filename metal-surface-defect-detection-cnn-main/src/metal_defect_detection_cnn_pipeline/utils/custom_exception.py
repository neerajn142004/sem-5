import sys

# Function to generate a detailed error message
def error_message_detail(error, error_detail: sys):
    # Extract the exception details using sys.exc_info()
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Get the line number where the exception occurred
    line_number = exc_tb.tb_lineno
    
    # Format the error message with the file name, line number, and error description
    error_message = f"Error occurred in Python script name {file_name} " \
                    f"line number {line_number} error message {str(error)}"
    
    # Return the formatted error message
    return error_message

# Custom exception class to provide detailed error information
class CustomException(Exception):
    # Constructor for the custom exception class
    def __init__(self, error_message, error_detail: sys):
        # Call the base Exception class constructor with the error message
        super().__init__(error_message)
        
        # Generate a detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    # Override the __str__ method to return the custom error message when the exception is printed
    def __str__(self):
        return self.error_message