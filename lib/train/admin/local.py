from .environment import default_paths


class EnvironmentSettings:
    def __init__(self):
        self.__dict__.update(default_paths())
