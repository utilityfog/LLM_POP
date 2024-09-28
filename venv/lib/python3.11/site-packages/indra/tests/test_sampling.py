from indra import api
import os
import pytest
import deeplake
import sys

from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
from .constants import MNIST_DS_NAME, IMAGENET_DS_NAME
from .utils import tmp_datasets_dir, agree

from .dataset_fixtures import *

ds = dataset_to_libdeeplake(
    deeplake.load(MNIST_DS_NAME, token=os.getenv(ENV_ACTIVELOOP_TOKEN), read_only=True)
)


def test_dataset_query_sampling():
    dsv = ds.sample("max_weight(labels==1: 9, true: 1)")
    assert len(dsv) == len(ds)

    dsv = ds.sample("max_weight(labels==1: 8, labels==2: 8)")
    assert len(dsv) == len(ds)
    for i in range(0, len(dsv)):
        assert dsv.tensors[1][i].numpy()[0] == 1 or dsv.tensors[1][i].numpy()[0] == 2

    dsv = ds.sample("sum_weight(labels==1: 8, labels==2: 8)", replace=False)
    ss = list(dsv.indexes)
    ss.sort()
    assert ss == list(range(0, len(ds)))

    dsv = ds.sample("sum_weight(labels==1: 8, labels==2: 8)", size=100)
    assert len(dsv) == 100
    for i in range(0, len(dsv)):
        assert dsv.tensors[1][i].numpy()[0] == 1 or dsv.tensors[1][i].numpy()[0] == 2

    dsv = ds.sample("labels")
    assert len(dsv) == len(ds)
    for i in range(0, len(dsv)):
        assert dsv.tensors[1][i].numpy()[0] != 0

    dsv = ds.sample("labels", size=2 * len(ds))
    assert len(dsv) == 2 * len(ds)
    for i in range(0, len(dsv)):
        assert dsv.tensors[1][i].numpy()[0] != 0

    dsv = ds.sample("labels", replace=False, size=2 * len(ds))
    assert len(dsv) == len(ds)
    ss = list(dsv.indexes)
    ss.sort()
    assert ss == list(range(0, len(ds)))


def test_dataset_weighted_sampling():
    new_weights = list()
    for i in range(0, len(ds)):
        if ds.tensors[1][i].numpy()[0] == 5:
            new_weights.append(9)
        elif ds.tensors[1][i].numpy()[0] == 7:
            new_weights.append(7)
        else:
            new_weights.append(0)
    dsv = ds.sample(new_weights)
    assert len(dsv) == len(ds)
    for i in range(0, len(dsv)):
        assert dsv.tensors[1][i].numpy()[0] == 5 or dsv.tensors[1][i].numpy()[0] == 7

    dsv = ds.sample(new_weights, replace=False)
    assert len(dsv) == len(ds)
    ss = list(dsv.indexes)
    ss.sort()
    assert ss == list(range(0, len(ds)))

    dsv = ds.sample(new_weights, replace=False, size=10000)
    assert len(dsv) == 10000
    ss = set(dsv.indexes)
    assert len(ss) == 10000


def test_dataset_query_sample_chaining():
    dsv = ds.sample("sum_weight(labels==1: 5, labels==2: 5, labels==3: 5)", size=1000)
    assert len(dsv) == 1000
    new_weights = list()
    for i in range(0, len(dsv)):
        assert (
            dsv.tensors[1][i].numpy()[0] == 1
            or dsv.tensors[1][i].numpy()[0] == 2
            or dsv.tensors[1][i].numpy()[0] == 3
        )
        if dsv.tensors[1][i].numpy()[0] == 1:
            new_weights.append(1)
        elif dsv.tensors[1][i].numpy()[0] == 3:
            new_weights.append(1)
        else:
            new_weights.append(0)

    dsvv = dsv.sample(new_weights)
    assert len(dsvv) == len(dsv)
    for i in range(0, len(dsvv)):
        assert dsvv.tensors[1][i].numpy()[0] == 1 or dsvv.tensors[1][i].numpy()[0] == 3

    dsvv = dsv.sample(tuple(new_weights))
    assert len(dsvv) == len(dsv)
    for i in range(0, len(dsvv)):
        assert dsvv.tensors[1][i].numpy()[0] == 1 or dsvv.tensors[1][i].numpy()[0] == 3


def test_empty_dataset(local_auth_ds):
    ds = local_auth_ds
    ds.create_tensor("images")
    ds.create_tensor("caption")
    ds.create_tensor("ds_split", htype="text")
    assert len(ds) == 0
    with pytest.raises(RuntimeError):
        ds.query("select * where ds_split == 'train' sample by 1.0 replace false")


def imagenet_sampling_bug(ids):
    weights = list()
    for i in range(1000):
        weights.append(i % 5)

    dsv = ids.sample(weights)
    assert len(dsv) == len(ids)

    dsv = ids.sample(weights, replace=False)
    assert len(dsv) == len(ids)


def imagenet_query_bug(ids):
    print(type(ids))
    dsv = ids.query(
        "SELECT * SAMPLE BY sum_weight(labels == 'fur coat': 10, labels == 'bikini': 3) limit 0.1 PERCENT"
    )
    assert len(dsv) == 1281


def empty_weights(ids):
    weights = list()

    dsv = ids.sample(weights, replace=False)
    assert len(dsv) == len(ids)

    dsv = ids.sample(weights, replace=True)
    assert len(dsv) == len(ids)


def test_imagenet():
    ids = dataset_to_libdeeplake(
        agree(IMAGENET_DS_NAME, token=os.getenv(ENV_ACTIVELOOP_TOKEN))
    )

    imagenet_sampling_bug(ids)
    imagenet_query_bug(ids)
    empty_weights(ids)
