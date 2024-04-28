
INFO=3
WARN=2
ERROR=1
QUIET=0

verbosity=3

def set_verbosity(level):
    global verbosity
    verbosity = level

def _log(message, endl, status):
    global verbosity
    if verbosity >= status:
        print(message, end=endl)    

def info(message="", endl="\n"):
    _log(message, endl, INFO)

def warn(message="", endl="\n"):
    _log(message, endl, WARN)

def err(message="", endl="\n"):
    _log(message, endl, ERROR)

