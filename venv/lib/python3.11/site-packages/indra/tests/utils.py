import pytest
import os
import pathlib

import sys
from io import StringIO
from contextlib import contextmanager
import deeplake


@pytest.fixture(scope="session")
def tmp_datasets_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


_THIS_FILE = pathlib.Path(__file__).parent.absolute()


def get_data_path(subpath: str = ""):
    return os.path.join(_THIS_FILE, "data" + os.sep, subpath)


def data_path(file_name: str = ""):
    path = get_data_path()
    return os.path.join(path, file_name)


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def agree(path, token: str):
    dataset_name = path.split("/")[-1]
    with replace_stdin(StringIO(dataset_name)):
        print(path, " ", token)
        ds = deeplake.load(path, token=token, read_only=True)

    return deeplake.load(path, token=token, read_only=True)
