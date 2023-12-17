def get_logs_path(cur_dir: str) -> str:
    """
    This function finds the root directory of the project and constructs a path to the 'logs' directory
    within the 'code/yq' folder, creating the directory if it doesn't exist.

    Parameters:
    cur_dir (str): The current directory from which to start searching for the root directory.

    Returns:
    str: The path to the logs directory.
    """
    root_dir = get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('code', 'yq', 'logs')
    tar_dir.mkdir(parents=True, exist_ok=True)
    return tar_dir

def get_plots_path(cur_dir: str) -> str:
    """
    This function finds the root directory of the project and constructs a path to the 'plots' directory
    within the 'code/yq' folder, creating the directory if it doesn't exist.

    Parameters:
    cur_dir (str): The current directory from which to start searching for the root directory.

    Returns:
    str: The path to the plots directory.
    """
    root_dir = get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('code', 'yq', 'plots')
    tar_dir.mkdir(parents=True, exist_ok=True)
    return tar_dir

def get_hparams(cur_dir: str) -> str:
    """
    This function finds the root directory of the project and constructs a path to the 'hparams' directory
    within the 'code/yq/scripts' folder, creating the directory if it doesn't exist.

    Parameters:
    cur_dir (str): The current directory from which to start searching for the root directory.

    Returns:
    str: The path to the hyperparameters directory.
    """
    root_dir = get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('code', 'yq', 'scripts', 'hparams')
    tar_dir.mkdir(parents=True, exist_ok=True)
    return tar_dir

def get_root_dir(cur_dir: str) -> str:
    """
    This function finds and returns the root directory of the project. It traverses up the directory tree 
    starting from 'cur_dir' until it finds the 'simulation-in-finance' directory, which is considered the 
    root of the project. If it reaches the filesystem root without finding this directory, 
    it raises a FileNotFoundError.

    Parameters:
    cur_dir (str): The current directory from which to start searching for the root directory.

    Returns:
    str: The root directory of the project.
    """
    while cur_dir.name != 'simulation-in-finance':    
            cur_dir = cur_dir.parent
            # print(f"Now we are at {cur_dir.name}\n")
            if cur_dir == cur_dir.parent:
                # We have reached the filesystem root without finding the marker
                raise FileNotFoundError("Root directory marker not found.")
    return cur_dir
