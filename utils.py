import os

class ErrorHandler:
    def __init__(self):
        self.errors = []

    def log_error(self, error_message, exception=None):
        error_entry = {
            "message": error_message,
            "exception": str(exception) if exception else None
        }
        self.errors.append(error_entry)
        print(f"[ERROR] {error_message}")
        if exception:
            print(f"        Exception: {exception}")

    def has_errors(self):
        return len(self.errors) > 0

    def get_errors(self):
        return self.errors

    def clear_errors(self):
        self.errors.clear()


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
