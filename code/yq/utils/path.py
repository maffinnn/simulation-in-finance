def get_logs_path(cur_dir: str) -> str:
    root_dir = get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('code', 'yq', 'logs')
    tar_dir.mkdir(parents=True, exist_ok=True)
    return tar_dir

def get_plots_path(cur_dir: str) -> str:
    root_dir = get_root_dir(cur_dir=cur_dir)
    tar_dir = root_dir.joinpath('code', 'yq', 'plots')
    tar_dir.mkdir(parents=True, exist_ok=True)
    return tar_dir

def get_root_dir(cur_dir: str) -> str:
    while cur_dir.name != 'simulation-in-finance':    
            cur_dir = cur_dir.parent
            # print(f"Now we are at {cur_dir.name}\n")
            if cur_dir == cur_dir.parent:
                # We have reached the filesystem root without finding the marker
                raise FileNotFoundError("Root directory marker not found.")
    return cur_dir
