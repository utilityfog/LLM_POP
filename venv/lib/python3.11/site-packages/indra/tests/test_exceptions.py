from .constants import MNIST_DS_NAME
from indra import api
import pytest


def test_authorization_exception():
    with pytest.raises(api.api.AuthorizationError):
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        api.dataset("hub://activeloop-test/coco-train", token=token)


def test_wrong_commit_id():
    ds = api.dataset(MNIST_DS_NAME)[0:100]
    assert len(ds) == 100

    with pytest.raises(api.api.wrong_commit_id):
        ds.checkout("some_hash")

    ds.checkout("firstdbf9474d461a19e9333c2fd19b46115348f")

