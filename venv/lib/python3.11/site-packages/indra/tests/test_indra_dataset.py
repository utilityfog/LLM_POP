import deeplake
from indra import api
import numpy as np
import io
from indra import api
from deeplake.core.sample import Sample
from .utils import tmp_datasets_dir
from PIL import Image


def test_png_chunk_bug(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "png_chunk_issue", overwrite=True)
    with ds:
        ds.create_tensor("image", htype="image", chunk_compression="png")
        ds.image.append(np.random.randint(0, 255, (200, 200, 4), np.uint8))
    ids = api.dataset(str(tmp_datasets_dir / "png_chunk_issue"))
    assert np.all(ids.tensors[0][0].numpy() == ds.image[0].numpy())


def test_tiled_bmask_bug(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "tiled_mask_issue", overwrite=True)
    with ds:
        ds.create_tensor(
            "masks",
            htype="binary_mask",
            sample_compression="lz4",
            tiling_threshold=1024 * 1024,
        )
        ds.masks.append(np.random.randint(0, 1, (640, 640, 80), np.bool_))
        assert len(ds.masks.chunk_engine.tile_encoder.entries) == 1
    ids = api.dataset(str(tmp_datasets_dir / "tiled_mask_issue"))
    assert np.all(ids.tensors[0][0].numpy() == ds.masks[0].numpy())


def test_tiled_samples_reading(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "tiles_samples", overwrite=True)
    with ds:
        ds.create_tensor(
            "masks",
            htype="binary_mask",
            sample_compression="lz4",
            tiling_threshold=1024 * 1024,
        )
        ds.masks.append(np.random.randint(0, 1, (640, 640, 80), np.bool_))
        assert len(ds.masks.chunk_engine.tile_encoder.entries) == 1
        # Two dimentional tiling
        ds.create_tensor("some", sample_compression=None, tiling_threshold=512 * 512)
        ds.some.append(np.random.randint(0, 256, (3000, 4000), np.int32))

        # Three dimentional tiling
        ds.create_tensor("small", sample_compression=None, tiling_threshold=128 * 128)
        ds.small.append(np.random.randint(0, 250, (128, 128, 3), np.int32))

    cpp_ds = api.dataset(str(tmp_datasets_dir / "tiles_samples"))
    assert np.all(cpp_ds.tensors[0][0].numpy() == ds.masks[0].numpy())
    assert cpp_ds.tensors[1].name == "some"
    assert cpp_ds.tensors[2].name == "small"
    assert np.all(cpp_ds.tensors[1][0].numpy() == ds.some[0].numpy())
    assert np.all(cpp_ds.tensors[2][0].numpy() == ds.small[0].numpy())


def test_non_string_commit_message(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "non_string_commit", overwrite=True)
    with ds:
        ds.create_tensor("labels", htype="class_label")
        ds.labels.append(1)
        ds.commit(["array commit message"])
        ds.labels.append(2)
    ids = api.dataset(str(tmp_datasets_dir / "non_string_commit"))
    assert len(ids) == 2
    ids.checkout("firstdbf9474d461a19e9333c2fd19b46115348f")
    assert len(ids) == 1
    ids.checkout(ds.pending_commit_id)
    assert len(ids) == 2


"""
TODO Disabling this test because deeplake temporarily disabled apng support.
def test_apng_mask(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "apng_mask", overwrite=True)
    with ds:
        ds.create_tensor("masks", htype="binary_mask", dtype=np.uint8, sample_compression="apng")
        ds.masks.append(np.random.randint(0, 255, (20, 20, 50), np.uint8))
    ids = api.dataset(str(tmp_datasets_dir / "apng_mask"))
    assert np.all(ids.tensors[0][0].numpy() == ds.masks[0].numpy())
"""


def test_sequences_with_same_length(tmp_datasets_dir):
    ds = deeplake.dataset(
        tmp_datasets_dir / "sequence_with_same_length", overwrite=True
    )
    with ds:
        ds.create_tensor(
            "images", htype="sequence[image]", dtype=np.uint8, sample_compression=None
        )
        ds.images.append(np.random.randint(0, 255, (3, 20, 20, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (4, 20, 20, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (5, 20, 20, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (5, 20, 20, 3), np.uint8))
        ds.images.append(np.random.randint(0, 255, (6, 20, 20, 3), np.uint8))
    ids = api.dataset(str(tmp_datasets_dir / "sequence_with_same_length"))
    for i in range(0, 4):
        f1 = ids.tensors[0][i].numpy()
        f2 = ds.images[i].numpy()
        assert len(f1) == len(f2)
        for j in range(0, len(f1)):
            assert np.all(f1[j] == f2[j])


def test_non_standard_jpegs(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "non_standard_jpegs", overwrite=True)
    with ds:
        ds.create_tensor(
            "images", htype="link[image]", dtype=np.uint8, sample_compression=None
        )
        image = Image.fromarray(np.random.randint(0, 255, (40, 20, 3), np.uint8))
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        stream.seek(3)
        stream.write(b"\xe0")
        f = open(tmp_datasets_dir / "non_standard_jpeg_0.jpg", "wb")
        f.write(stream.getbuffer())
        f.close()
        ds.images.append(
            deeplake.link(str(tmp_datasets_dir / "non_standard_jpeg_0.jpg"))
        )
        stream.seek(3)
        stream.write(b"\xe1")
        f = open(tmp_datasets_dir / "non_standard_jpeg_1.jpg", "wb")
        f.write(stream.getbuffer())
        f.close()
        ds.images.append(
            deeplake.link(str(tmp_datasets_dir / "non_standard_jpeg_1.jpg"))
        )
        stream.seek(3)
        stream.write(b"\xe2")
        f = open(tmp_datasets_dir / "non_standard_jpeg_2.jpg", "wb")
        f.write(stream.getbuffer())
        f.close()
        ds.images.append(
            deeplake.link(str(tmp_datasets_dir / "non_standard_jpeg_2.jpg"))
        )
        stream.seek(3)
        stream.write(b"\xfe")
        f = open(tmp_datasets_dir / "non_standard_jpeg_3.jpg", "wb")
        f.write(stream.getbuffer())
        f.close()
        ds.images.append(
            deeplake.link(str(tmp_datasets_dir / "non_standard_jpeg_3.jpg"))
        )

    ids = api.dataset(str(tmp_datasets_dir / "non_standard_jpegs"))
    assert ids.images[0].numpy().shape == (40, 20, 3)
    assert ids.images[1].numpy().shape == (40, 20, 3)
    assert ids.images[2].numpy().shape == (40, 20, 3)
    assert ids.images[3].numpy().shape == (40, 20, 3)


def test_multidimensional_tiles_bug(tmp_datasets_dir):
    ds = deeplake.dataset(tmp_datasets_dir / "md_array_dataset", overwrite=True)
    with ds:
        images = ds.create_tensor("data")
        random_tensor = np.random.randint(
            0, 65535, size=(5, 3679, 5502, 3), dtype=np.uint16
        )
        entry = {"data": random_tensor}
        ds.append(entry)

    ids = api.dataset(str(tmp_datasets_dir / "md_array_dataset"))
    assert np.all(ids.data[0].numpy() == ds.data[0].numpy())
