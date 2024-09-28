import os
import pytest
from uuid import uuid4
import posixpath
import deeplake

from .constants import *


from deeplake.core.storage.local import LocalProvider


SESSION_ID = "tmp" + str(uuid4())[:4]


@pytest.fixture(scope="session")
def get_pytest_local_path():
    return posixpath.join(PYTEST_LOCAL_PROVIDER_BASE_ROOT, SESSION_ID)


@pytest.fixture
def local_path(request):
    path = posixpath.join(PYTEST_LOCAL_PROVIDER_BASE_ROOT, SESSION_ID)

    LocalProvider(path).clear()
    yield path

    LocalProvider(path).clear()


@pytest.fixture(scope="session")
def hub_cloud_dev_credentials(request):
    username = os.getenv(ENV_HUB_DEV_USERNAME)

    assert (
        username is not None
    ), f"Deep Lake dev username was not found in the environment variable '{ENV_HUB_DEV_USERNAME}'. This is necessary for testing deeplake cloud datasets."

    return username, None


@pytest.fixture(scope="session")
def hub_cloud_dev_token(hub_cloud_dev_credentials):
    token = os.getenv(ENV_ACTIVELOOP_TOKEN)

    assert (
        token is not None
    ), f"Deep Lake dev token was not found in the environment variable ${ENV_ACTIVELOOP_TOKEN}. This is necessary for testing deeplake cloud datasets."

    return token


@pytest.fixture
def local_auth_ds_generator(local_path, hub_cloud_dev_token):
    def generate_local_auth_ds(**kwargs):
        return deeplake.dataset(local_path, token=hub_cloud_dev_token, **kwargs)

    return generate_local_auth_ds


@pytest.fixture
def local_auth_ds(local_auth_ds_generator):
    return local_auth_ds_generator()
