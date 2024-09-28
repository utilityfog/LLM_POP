from indra import api
import deeplake
import numpy as np
from .utils import tmp_datasets_dir
from .constants import (
    MNIST_DS_NAME,
)


def test_indra():
    ds = api.dataset(MNIST_DS_NAME)
    tensors = ds.tensors
    assert isinstance(tensors, list)


def test_headless_tensor(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "headless-tensor-ds", overwrite=True)
    with ds:
        ds.create_tensor("labels", dtype=np.uint8, htype="class_label")
        ds.labels.append(1)
        ds.labels.append(2)
        ds.labels.append(3)
    ids = api.dataset(str(tmp_datasets_dir / "headless-tensor-ds"))
    t = ids.labels
    del ids
    assert len(t) == 3
    assert np.all(t[0].numpy() == [1])
    assert np.all(t[1].numpy() == [2])
    assert np.all(t[2].numpy() == [3])
    tt = t[0:2]
    del t
    assert len(tt) == 2
