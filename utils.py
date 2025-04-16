import os
import traceback
import logging
from typing import List, Dict, Optional, Union, Any

def create_directories(path_config):
    """

    """
    for directory in path_config:
        if not (os.path.exists(path_config[directory])) and ('dir' in directory):
            os.makedirs(path_config[directory])
            print(f"Created directory: {path_config[directory]}")

def check_environment(path_config):
    """

    """
    if 'XUVTOP' not in os.environ:
        print("Warning: XUVTOP environment variable not set.")
        os.environ['XUVTOP'] = path_config['chianti_path']
    print(f"EXUVTOP environment set to {path_config['chianti_path']}")
