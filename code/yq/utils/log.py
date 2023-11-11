import logging

def setup_logger(name, log_file, level=logging.INFO):
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