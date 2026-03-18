import sys
from logger import get_logger

def error_msg_details(error, error_detail:sys):
    '''
    Custom error msg formatter
    '''

    _,_,exc_tb = error_detail.exc_info()
    # file_name = exc_tb.tb_frame.f_code.co_filename
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_no = "Unknown"
    # error_msg = f"Error occurred in python script name [{0}] line [{1}] error msg [{2}]"(
    #     file_name, exc_tb.tb_lineno, str(error)
    # )

    error_msg = (
        f"Error occurred in python script [{file_name}]"
        f"line [{line_no}] error message [{error}]"
    )

    return error_msg



class CustomException(Exception):
    def __init__(self, error_msg, error_details:sys):
        super().__init__(error_msg)

        self.error_msg = error_msg_details(
            error=error_msg,
            error_detail=error_details
        )

    def __str__(self):
        return self.error_msg


if __name__ == "__main__":
    log = get_logger(__name__)
    try:
        a = 1/0
    except Exception as e:
        log.error("Divided by Zero")
        raise CustomException(e,sys)