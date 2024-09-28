from indra import api
import numpy as np
import deeplake
import math
from .utils import tmp_datasets_dir


def test_basic(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "basic_shape_test_dataset", overwrite=True)
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.images.append(np.random.randint(0, 255, (200, 300, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (200, 300, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (200, 300, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (200, 300, 3), np.uint8))

    ds2 = api.dataset(str(tmp_datasets_dir / "basic_shape_test_dataset"))
    assert len(ds2) == 4
    assert ds2.images.shape == ds.images.shape
    assert ds2.images.shape == ds.images.shape
    assert ds2.images[0:2].shape == ds.images[0:2].shape
    assert ds2[0:2].images.shape == ds[0:2].images.shape


def test_ordinary_shape(tmp_datasets_dir):
    tmp_ds = deeplake.dataset(tmp_datasets_dir / "shape_test_dataset", overwrite=True)
    with tmp_ds as ds:
        ds.create_tensor("labels", htype="class_label")
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.labels.append([1, 2, 3])
        ds.labels.append([1])
        ds.images.append(np.random.randint(0, 255, (400, 600, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (200, 300, 3), np.uint8))

    ds2 = api.dataset(str(tmp_datasets_dir / "shape_test_dataset"))
    assert len(ds2) == 2
    assert ds2.tensors[0].name == "labels"
    assert ds2.labels[0].shape == tmp_ds.labels[0].shape
    assert ds2.labels[1].shape == tmp_ds.labels[1].shape
    assert ds2.tensors[1].name == "images"
    assert ds2.images[0].shape == tmp_ds.images[0].shape
    assert ds2.images[1].shape == tmp_ds.images[1].shape


def test_dynamic_shape(tmp_datasets_dir):
    tmp_ds = deeplake.dataset(
        tmp_datasets_dir / "shape_test_dataset_sequential", overwrite=True
    )
    with tmp_ds as ds:
        ds.create_tensor("labels", htype="sequence[class_label]")
        ds.create_tensor("images", htype="sequence[image]", sample_compression="jpeg")
        ds.labels.append([[1, 2, 3], [1], [2, 3]])
        ds.labels.append([[1, 2, 3, 2, 1]])
        a = list()
        for i in range(5):
            a.append(np.random.randint(0, 255, (400, 600, 3), np.uint8))
        ds.images.append(a)

    ds2 = api.dataset(str(tmp_datasets_dir / "shape_test_dataset_sequential"))
    assert ds2.labels[0].shape == tmp_ds.labels[0].shape
    assert ds2.labels[1].shape == tmp_ds.labels[1].shape
    assert ds2.images[0].shape == tmp_ds.images[0].shape


def test_polygon_shape(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "polygon_shape_dataset", overwrite=True)
    with ds:
        ds.create_tensor(
            "polygon",
            dtype=np.float32,
            htype="polygon",
            sample_compression=None,
        )
        for i in range(10):
            polygons = []
            for j in range(i):
                p = np.ndarray((3 + j, 2))
                c = (3 + j) / 2
                for k in range(3 + j):
                    p[k] = [
                        200 * (j % 3) + 150 * (math.sin(6.28 * k / (3 + j)) + 1) / 2,
                        200 * (j / 3) + 150 * (math.cos(6.28 * k / (3 + j)) + 1) / 2,
                    ]
                polygons.append(p)
            ds.polygon.append(polygons)

    ids = api.dataset(str(tmp_datasets_dir / "polygon_shape_dataset"))
    assert ids.polygon.shape == (10, None, None, None)
    assert ids.polygon[0].shape == (0, 0, 0)
    for i in range(1, 10):
        assert ids.polygon[i].shape == (i, i + 2, 2)
