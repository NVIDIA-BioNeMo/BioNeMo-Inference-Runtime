import os
import inspect

from tensorrt_llm_lite.logger import logger as tllm_logger

logger = tllm_logger


def trtbnm_logger(message: str = "trt_bnm_logger", do_print: bool=False):
    
    trtbnm_log_level_from_env: str = os.getenv("TRTBNM_LOG_LEVEL", "CRITICAL").upper()
    
    if trtbnm_log_level_from_env in ["DEBUG", "INFO"]:
        caller_info_: dict = caller_info()
        formatted_message = \
        f"""
        ******** trtbnm, begin ******
        caller {caller_info_["function_name"]}
            {message}
        at {caller_info_["filename"]}: {caller_info_["line"]}
        ******** trtbnm, end ********
        """
        if do_print:
            print(formatted_message)
        logger.log(logger.INFO, formatted_message)


def caller_info(levels_up: int = 1) -> str:
    """Return 'ClassName.method_name' or 'function_name' of the caller."""
    frame = inspect.stack()[levels_up + 1]  # +1 to skip this function
    locals_ = frame.frame.f_locals
    
    calling_function_name = f"{frame.function}"
    if 'self' in locals_:
        calling_function_name =  f"{locals_['self'].__class__.__name__}.{frame.function}"
    elif 'cls' in locals_:
        calling_function_name =  f"{locals_['cls'].__name__}.{frame.function}"

    return {
        "function_name": calling_function_name,
        "filename": frame.filename,
        "line": frame.lineno,
    }
