import functools
import datetime
import logging

logger_yq = logging.getLogger('yq')

def timeit(func):
    """
    A decorator that logs the execution time of a function.

    Parameters:
    func (callable): The function to be timed.

    Returns:
    callable: The wrapper function.

    This decorator logs the runtime of the function in hours, minutes, seconds, and milliseconds.
    It can be applied to any function using the @timeit syntax.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # Argument representation
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # Keyword-argument representation
        signature = ", ".join(args_repr + kwargs_repr)  # Formatting

        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        # Convert to hours, minutes, seconds, and milliseconds
        hours, rem = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = (seconds - int(seconds)) * 1000
        seconds = int(seconds)

        logger_yq.info(f"Runtime of {func.__name__}: {int(hours)}h {int(minutes)}m {seconds}s {milliseconds:.6f}ms")
        return result
    return wrapper

if __name__ == "__main__":
    # @timeit
    # funcname()
    pass