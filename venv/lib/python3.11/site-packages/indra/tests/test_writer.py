from indra import api
import numpy as np
import pytest
import deeplake
from .utils import tmp_datasets_dir, data_path

def test_class_lavel_extend(tmp_datasets_dir):
    dw = api.dataset_writer(str(tmp_datasets_dir / "class_label/"))

    dw.create_tensor("label", htype="class_label", dtype="bool")
    dw.extend({"label": ["train"]*50})
    dw.extend({"label": ["test"]*50})
    dw.finish()

    ds = api.dataset(str(tmp_datasets_dir / "class_label/"))
    print(ds.label.info['class_names'])
    assert len(ds.label.info['class_names']) == 2
    assert np.all(ds.label.numpy() == [False]*50 + [True]*50)
    print(ds.label.dtype)
    assert ds.label.dtype == bool
    assert ds.label.info["class_names"][0] == "train"
    assert ds.label.info["class_names"][1] == "test"


def test_empty_shape_data_append(tmp_datasets_dir):
    dw = api.dataset_writer(str(tmp_datasets_dir / "empty_shape/"))

    dw.create_tensor("labels", htype="class_label")

    with dw:
        dw.labels.extend(1)
        dw.labels.extend(11)
        dw.labels.extend(111)
        dw.labels.extend(1111)

    ds = deeplake.load(tmp_datasets_dir / "empty_shape")

    assert len(ds) == 4
    assert ds.labels[0].numpy() == np.uint32(1)
    assert ds.labels[1].numpy() == np.uint32(11)
    assert ds.labels[2].numpy() == np.uint32(111)
    assert ds.labels[3].numpy() == np.uint32(1111)


def test_string_data_append(tmp_datasets_dir):
    dw = api.dataset_writer(str(tmp_datasets_dir / "string_dataset/"))

    print(str(tmp_datasets_dir / "string_dataset/"))
    dw.create_tensor("sentences", htype="text")

    sentences = [
        "This is a sentence.",
        "This is another sentence.",
        "This is a third sentence.",
        "This is a fourth sentence.",
    ]

    with dw:
        dw.sentences.extend(sentences)
        dw.sentences.extend(["This is a fifth sentence."])
        sentences.append("This is a fifth sentence.")

    ds = deeplake.load(tmp_datasets_dir / "string_dataset")

    assert len(ds) == 5
    assert ds.sentences[0].tobytes() == sentences[0].encode("utf-8")
    assert ds.sentences[1].tobytes() == sentences[1].encode("utf-8")
    assert ds.sentences[2].tobytes() == sentences[2].encode("utf-8")
    assert ds.sentences[3].tobytes() == sentences[3].encode("utf-8")
    assert ds.sentences[4].tobytes() == sentences[4].encode("utf-8")


@pytest.mark.performance
def test_speed(tmp_datasets_dir):
    from time import time
    from tqdm import tqdm
    start = time()
    dw = api.dataset_writer(str(tmp_datasets_dir / "large_ds/"))

    dw.create_tensor("labels", htype="class_label", dtype="uint32")
    with dw:
        for i in tqdm(range(1000)):
            dw.labels.extend(np.random.randint(0, 50000, size=(1000, 100), dtype=np.uint32))

    print(f"Indra {time() - start}")

    dw = deeplake.empty(str(tmp_datasets_dir / "large_ds/"), overwrite=True)

    start = time()
    dw.create_tensor("labels", htype="class_label", dtype="uint32")

    with dw:
        for i in tqdm(range(1000)):
            dw.labels.extend(np.random.randint(0, 50000, size=(1000, 100), dtype=np.uint32))

    print(f"Indra {time() - start}")
