from indra import api
import pytest
from .dataset_fixtures import get_pytest_local_path

def test_class_label(get_pytest_local_path):
    ds = api.dataset_writer(get_pytest_local_path)
    with ds:
        ds.create_tensor("labels", htype="class_label", class_names=["a", "b", "c"])
        ds.create_tensor("labels2", htype="class_label")
        assert len(ds.labels.info) == 1
        assert len(ds.labels2.info) == 1
        assert (
            ds.labels.info.class_names == ds.labels.info["class_names"] == ["a", "b", "c"]
        )
        assert ds.labels2.info.class_names == ds.labels2.info["class_names"] == []
        ds.labels.info.class_names = ["c", "b", "a"]

    # TODO check after write
    # ds = api.dataset_writer("./test_class_label_1/")
    # assert len(ds.labels.info) == 1
    # assert len(ds.labels2.info) == 1
    # assert (
    #     ds.labels.info.class_names == ds.labels.info["class_names"] == ["c", "b", "a"]
    # )
    # assert ds.labels2.info.class_names == ds.labels2.info["class_names"] == []


def test_bbox(get_pytest_local_path):
    ds = api.dataset_writer(get_pytest_local_path)
    with ds:
        ds.create_tensor("bboxes", htype="bbox", coords={"type": 0, "mode": 2})
        ds.create_tensor("bboxes1", htype="bbox", coords={"type": 1})
        ds.create_tensor("bboxes2", htype="bbox")
        assert len(ds.bboxes.info) == 1
        assert len(ds.bboxes2.info) == 1
        assert ds.bboxes.info.coords == ds.bboxes.info["coords"] == {"type": 0, "mode": 2}
        assert ds.bboxes1.info.coords == ds.bboxes1.info["coords"] == {"type": 1}
        assert ds.bboxes2.info.coords == ds.bboxes2.info["coords"] == {}
        ds.bboxes.info.coords = {"type": 3}

    # ds = local_ds_generator()
    # assert len(ds.bboxes.info) == 1
    # assert len(ds.bboxes2.info) == 1
    # assert ds.bboxes.info.coords == ds.bboxes.info["coords"] == {"type": 3}
    # assert ds.bboxes2.info.coords == ds.bboxes2.info["coords"] == {}
    # with pytest.raises(TypeError):
    #     ds.create_tensor("bboxes3", htype="bbox", coords=[1, 2, 3])

    # with pytest.raises(KeyError):
    #     ds.create_tensor("bboxes4", htype="bbox", coords={"random": 0})



def test_info_new_methods(get_pytest_local_path):
    ds = api.dataset_writer(get_pytest_local_path)
    ds.create_tensor("x")

    ds.info["0"] = "hello"
    ds.info["1"] = "world"
    assert len(ds.info) == 2
    assert set(ds.info.keys()) == {'0', '1'}
    assert 0 in ds.info
    assert 1 in ds.info

    assert ds.info[0] == "hello"
    assert ds.info[1] == "world"

    del ds.info[0]
    assert len(ds.info) == 1
    assert 1 in ds.info
    assert ds.info[1] == "world"



def test_info_persistence_bug(get_pytest_local_path):
    ds = api.dataset_writer(get_pytest_local_path)
    with ds:
        ds.create_tensor("xyz")
    # ds.commit()
        ds.xyz.info.update(abc=123)
        assert ds.xyz.info.abc == 123
    # ds = local_ds_generator()
    # assert ds.xyz.info.abc == 123