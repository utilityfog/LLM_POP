from indra import api

import pytest
from .utils import tmp_datasets_dir
from .dataset_fixtures import *
import numpy as np

from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake


def test_empty_dataset_with_empty_tensor_materaialization(
    local_auth_ds, local_auth_ds_generator
):
    with local_auth_ds as ds:
        ds.create_tensor("val")
        ds.create_tensor("id")
        ds.id.extend([i for i in range(10)])

    dest = deeplake.empty(
        posixpath.join(PYTEST_LOCAL_PROVIDER_BASE_ROOT, SESSION_ID + str(1))
    )

    source_paths = [ds.path, ds.path]
    queries = [f'SELECT * FROM "{path}"' for path in source_paths]
    query = " union ".join(queries)
    view = deeplake.query(query)
    print("---------- VIEW SUMMARY BEFORE MATERIALIZATION---------")
    print(view.summary())

    target_ds = deeplake.load(dest.path, read_only=True, indra=True)
    print("---------- TARGET_DS SUMMARY BEFORE MATERIALIZATION---------")
    print(target_ds.summary())

    view.indra_ds.materialize(target_ds.indra_ds)

    print("---------- TARGET_DS SUMMARY AFTER MATERIALIZATION---------")
    print(target_ds.summary())
    assert len(target_ds) == 20
    view.indra_ds.materialize(target_ds.indra_ds)
    assert len(target_ds) == 40



def test_virtual_tensors(local_auth_ds):
    with local_auth_ds as deeplake_ds:
        deeplake_ds.create_tensor("label", htype="generic", dtype=np.int32)
        deeplake_ds.create_tensor("embeddings", htype="generic", dtype=np.float32)
        deeplake_ds.create_tensor("text", htype="text")
        deeplake_ds.create_tensor("json", htype="json")
        for i in range(100):
            count = i % 5
            deeplake_ds.label.append([int(i % 100)] * count)
            deeplake_ds.embeddings.append(
                [1.0 / float(i + 1), 0.0, -1.0 / float(i + 1)]
            )
            deeplake_ds.text.append(f"Hello {i}")
            deeplake_ds.json.append('{"key": "val"}')

    source_ds = dataset_to_libdeeplake(deeplake_ds).query(
        "SELECT *, shape(label)[0] as num_labels"
    )

    new_ds = deeplake.dataset(local_auth_ds.path + "/new_ds")
    dest_ds = dataset_to_libdeeplake(new_ds)
    source_ds.materialize(dest_ds)

    materialized = deeplake.load(local_auth_ds.path + "/new_ds")

    assert len(materialized.num_labels) == len(source_ds.num_labels) == 100
    assert np.all(materialized.num_labels.numpy() == source_ds.num_labels.numpy())
