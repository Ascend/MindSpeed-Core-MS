import datetime
import re
import sys
import os
import site
import warnings
from collections import namedtuple


def check_path_owner_consistent(path: str):
    if not os.path.exists(path):
        msg = f"The path does not exist: {path}"
        raise RuntimeError(msg)
    if os.stat(path).st_uid != os.getuid():
        warnings.warn(f"Warning: The {path} owner does not match the current owner.")


def check_directory_path_readable(path):
    check_path_owner_consistent(path)
    if os.path.islink(path):
        msg = f"Invalid path is a soft chain: {path}"
        raise RuntimeError(msg)
    if not os.access(path, os.R_OK):
        msg = f"The path permission check failed: {path}"
        raise RuntimeError(msg)


def get_cann_version():
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH", "")
    cann_version = "not known"
    check_directory_path_readable(os.path.realpath(ascend_home_path))
    for dirpath, _, filenames in os.walk(os.path.realpath(ascend_home_path)):
        install_files = [file for file in filenames if re.match(r"ascend_.*_install\.info", file)]
        if install_files:
            filepath = os.path.realpath(os.path.join(dirpath, install_files[0]))
            check_directory_path_readable(filepath)
            with open(filepath, "r") as f:
                for line in f:
                    if line.find("version") != -1:
                        cann_version = line.strip().split("=")[-1]
                        break
    return cann_version
