from indra import api
import deeplake
import numpy as np

from .utils import tmp_datasets_dir

MERGE_DATASET_PATH = "fast_merge_dataset"
MERGE_RENAME_DATASET_PATH = "fast_merge_rename"


def test_mege_strategy(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / MERGE_DATASET_PATH, overwrite=True)
    with ds:
        ds.create_tensor("x")
        ds.x.extend(list(range(10)))
        pending_commit = ds.pending_commit_id
        ds.commit()
        ds.checkout("alt", create=True)
        ds.create_tensor("y")
        ds.y.extend(list(range(10)))
        ds.commit()
        ds.checkout("main")
        ds.merge("alt")
    cpp_ds = api.dataset(str(tmp_datasets_dir / MERGE_DATASET_PATH))
    assert len(cpp_ds.tensors) == 2
    assert cpp_ds.tensors[0].name == "x"
    assert cpp_ds.tensors[1].name == "y"

    for i in range(len(ds)):
        assert np.all(cpp_ds.tensors[0][i].numpy() == ds.x[i].numpy())
        assert np.all(cpp_ds.tensors[1][i].numpy() == ds.y[i].numpy())

    cpp_ds.checkout(pending_commit)
    assert len(cpp_ds.tensors) == 1
    assert cpp_ds.tensors[0].name == "x"

    for i in range(len(ds)):
        assert np.all(cpp_ds.tensors[0][i].numpy() == ds.x[i].numpy())


def test_rename_merge(tmp_datasets_dir):
    local_ds = deeplake.dataset(
        tmp_datasets_dir / MERGE_RENAME_DATASET_PATH, overwrite=True
    )
    with local_ds as ds:
        # no conflicts
        ds.create_tensor("abc")
        ds.abc.append([1, 2, 3])
        first = ds.commit()
        ds.checkout("alt", create=True)
        ds.rename_tensor("abc", "xyz")
        ds.xyz.append([3, 4, 5])
        second = ds.commit()
        ds.checkout("main")
        ds.merge("alt")
        assert "abc" not in ds.tensors
        np.testing.assert_array_equal(ds.xyz.numpy(), np.array([[1, 2, 3], [3, 4, 5]]))

        # tensor with same name on main
        ds.create_tensor("red")
        ds.red.append([2, 3, 4])
        third = ds.commit()
        ds.checkout("alt2", create=True)
        ds.rename_tensor("red", "blue")
        ds.blue.append([1, 0, 0])
        forth = ds.commit()
        ds.checkout("main")
        ds.create_tensor("blue")
        ds.blue.append([0, 0, 1])
        fifth = ds.commit()
        # resolve
        ds.merge("alt2", force=True)
        np.testing.assert_array_equal(ds.red.numpy(), np.array([[2, 3, 4]]))
        np.testing.assert_array_equal(
            ds.blue.numpy(), np.array([[0, 0, 1], [2, 3, 4], [1, 0, 0]])
        )
        sixt = ds.pending_commit_id

        cpp_ds = api.dataset(str(tmp_datasets_dir / MERGE_RENAME_DATASET_PATH))
        cpp_ds.checkout(first)
        assert cpp_ds.tensors[0].name == "abc"
        assert len(cpp_ds) == 1

        cpp_ds.checkout(second)
        assert cpp_ds.tensors[0].name == "xyz"
        assert len(cpp_ds) == 2
        assert len(cpp_ds.tensors) == 1

        cpp_ds.checkout(third)
        assert len(cpp_ds.tensors) == 2
        assert cpp_ds.tensors[1].name == "red"
        assert np.all(cpp_ds.tensors[1][0].numpy() == [2, 3, 4])

        cpp_ds.checkout(forth)
        assert len(cpp_ds.tensors) == 2
        assert cpp_ds.tensors[1].name == "blue"
        assert len(cpp_ds.tensors[1]) == 2
        assert np.all(cpp_ds.tensors[1][0].numpy() == [2, 3, 4])
        assert np.all(cpp_ds.tensors[1][1].numpy() == [1, 0, 0])

        cpp_ds.checkout(fifth)
        assert np.all(cpp_ds.tensors[1][0].numpy() == [2, 3, 4])
        assert len(cpp_ds.tensors[1]) == 1

        cpp_ds.checkout(sixt)
        assert cpp_ds.tensors[0].name == "xyz"
        assert cpp_ds.tensors[1].name == "red"
        assert cpp_ds.tensors[2].name == "blue"
        assert len(cpp_ds.tensors[2]) == 3

        assert np.all(cpp_ds.tensors[1][0].numpy() == [2, 3, 4])
        assert np.all(cpp_ds.tensors[2][0].numpy() == [0, 0, 1])
        assert np.all(cpp_ds.tensors[2][1].numpy() == [2, 3, 4])
        assert np.all(cpp_ds.tensors[2][2].numpy() == [1, 0, 0])
