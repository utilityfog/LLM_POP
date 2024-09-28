from indra import api
import numpy as np
import deeplake
import pytest
from .utils import tmp_datasets_dir
from .constants import (
    MNIST_DS_NAME,
)
import random
import gc


def check_equality(dsv, ds, slice):
    for i, x in enumerate(slice):
        assert dsv.tensors[1][i].numpy() == ds.tensors[1][x].numpy()


def test_dataset_slicing():
    ds = api.dataset(MNIST_DS_NAME)
    dsv = ds[0:1000]
    check_equality(dsv, ds, range(0, 1000))
    dsv = ds[0:1000:7]
    check_equality(dsv, ds, range(0, 1000, 7))
    dsv = ds[0:1000:5]
    check_equality(dsv, ds, range(0, 1000, 5))
    dsv = ds[337:5647:13]
    check_equality(dsv, ds, range(337, 5647, 13))

    dsvv = dsv[1:50:6]
    check_equality(dsvv, dsv, range(1, 50, 6))
    dsvv = dsv[[5, 3, 8, 1, 90, 80, 70],]
    check_equality(dsvv, dsv, [5, 3, 8, 1, 90, 80, 70])
    with pytest.raises(IndexError):
        dsvv = dsv[[5, 3, 8, 1, 2000, 90, 80, 70],]
        print(len(dsvv))

    dsv = ds[[1, 59999, 49999, 4999, 399, 29],]
    check_equality(dsv, ds, [1, 59999, 49999, 4999, 399, 29])
    dsvv = dsv[[5, 3, 1, 4, 2, 0],]
    check_equality(dsvv, dsv, [5, 3, 1, 4, 2, 0])
    dsvv = dsv[1:5:2]
    check_equality(dsvv, dsv, range(1, 5, 2))


def test_advanced_slicing_and_equality():
    ds = api.dataset(MNIST_DS_NAME)
    deeplake_ds = deeplake.dataset(MNIST_DS_NAME, read_only=True)
    assert (
        len(ds[slice(None, None, None)])
        == len(deeplake_ds[slice(None, None, None)])
        == 60000
    )
    assert (
        len(ds[slice(None, None, -1)])
        == len(deeplake_ds[slice(None, None, -1)])
        == 60000
    )
    assert len(ds[slice(None, -1, -1)]) == len(deeplake_ds[slice(None, -1, -1)]) == 0
    assert len(ds[slice(None, -2, -1)]) == len(deeplake_ds[slice(None, -2, -1)]) == 1

    ds_view = ds[slice(None, -3, -1)]
    dp_ds_view = deeplake_ds[slice(None, -3, -1)]
    assert len(ds_view) == len(dp_ds_view)

    for i in range(len(ds_view)):
        assert np.array_equal(
            ds_view.tensors[0][i].numpy(), dp_ds_view.images[i].numpy()
        )


def test_tensors_slicing(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "tmp_dataset_slicing", overwrite=True)
    with ds:
        ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            ds.label.append(int(1000 * random.uniform(0.0, 1.0)))

    ids = api.dataset(str(tmp_datasets_dir / "tmp_dataset_slicing"))

    for i in range(10):
        s = int(1000 * random.uniform(0.0, 1.0))
        e = int(1000 * random.uniform(0.0, 1.0))
        if e < s:
            s, e = e, s
        t1 = ds.label[s:e]
        t2 = ids.label[s:e]
        v = ids[s:e]
        for i in range(e - s):
            assert np.all(t1[i].numpy() == t2[i].numpy())
            assert np.all(v.label[i].numpy() == t2[i].numpy())


def test_sequence_tensors_slicing(tmp_datasets_dir):
    ds = deeplake.dataset(
        tmp_datasets_dir / "tmp_sequence_dataset_slicing", overwrite=True
    )
    with ds:
        ds.create_tensor("label", htype="generic", dtype=np.int32)
        for i in range(1000):
            ds.label.append(int(100 * random.uniform(0.0, 1.0)))

    ids = api.dataset(str(tmp_datasets_dir / "tmp_sequence_dataset_slicing")).query(
        "SELECT * GROUP BY label"
    )

    tt = ids.label
    del ids
    gc.collect()
    for i in range(10):
        x = int(random.uniform(2.0 / len(tt), 1.0) * len(tt))
        assert tt[slice(0, x), slice(None), slice(0, 10)].shape[0] == x

    t = tt[slice(0, 10), slice(None), slice(0, 10)]
    del tt
    gc.collect()
    assert len(t) == 10
    for i in range(len(t)):
        arr = t[i].numpy()
        l = len(arr)
        for j in range(l):
            assert arr[j].shape == (1,)
