import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    This function configures a logger with a specified name and log file. It formats the log messages to 
    include the timestamp, logger name, log level, and the log message itself. 

    Parameters:
    name (str): Name of the logger. This is useful when you have multiple loggers in your application.
    log_file (str/Path): Path to the log file where log messages will be written.
    level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.

    Returns:
    logging.Logger: A configured logger that writes messages to the specified log file.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    pass
    # # Put this somewhere in the main file (logging is global)
    # cur_dir = Path(__file__).parent
    # logger_yq = log.setup_logger('yq', yq_path.get_logs_path(cur_dir=cur_dir).joinpath('log_file.log'))
    
    # # Insert this in some files (can be logger_heston or anything else but 
    # # if the file path need to be different then need to set up differently)
    # logger_yq = logging.getLogger('yq')

    # TRYING OUT DIFFERENT LOGGERS (to prevent huge log files during simulations)
    # # Main logger
    # main_logger = log.setup_logger('main', 'main.log', level=logging.INFO)

    # # Debug logger
    # debug_logger = log.setup_logger('debug', 'debug.log', level=logging.DEBUG)

    # # Simulation logger
    # simulation_logger = log.setup_logger('simulation', 'simulation.log', level=logging.WARNING)

    # # Example usage
    # main_logger.info("This is an info message from the main logger.")
    # debug_logger.debug("This is a debug message from the debug logger.")
    # simulation_logger.warning("This is a warning message from the simulation logger.")

    # # Muting a logger
    # simulation_logger.setLevel(logging.CRITICAL)  # Only CRITICAL messages will be logged