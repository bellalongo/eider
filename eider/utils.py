import os
import traceback
import logging
from typing import List, Dict, Optional, Union, Any

def create_directories(path_config):
    """
        Creates output directories based on the path configuration.
        Checks for directory keys in the path_config dictionary and creates
        directories if they don't already exist in the filesystem.
        Arguments:
            path_config (dict): Dictionary containing path configurations where keys
                               with 'dir' in their name will be created as directories
        Returns:
            None
        Raises:
            OSError: If directory creation fails due to permissions or other OS issues
    """
    for directory in path_config:
        if not (os.path.exists(path_config[directory])) and ('dir' in directory):
            os.makedirs(path_config[directory])
            print(f"Created directory: {path_config[directory]}")

def check_environment(path_config):
    """
        Verifies and sets the XUVTOP environment variable needed for CHIANTI.
        The XUVTOP environment variable is required for ChiantiPy to locate atomic
        data files. This function ensures it's set, using a default from path_config
        if needed.
        Arguments:
            path_config (dict): Dictionary containing path configurations,
                               must include a 'chianti_path' key with path to CHIANTI database
        Returns:
            None
        Raises:
            KeyError: If 'chianti_path' is not in path_config
    """
    if 'XUVTOP' not in os.environ:
        print("Warning: XUVTOP environment variable not set.")
        os.environ['XUVTOP'] = path_config['chianti_path']
    print(f"EXUVTOP environment set to {path_config['chianti_path']}")
