from indra import api
from .constants import (
    COCO_DS_NAME,
    MNIST_DS_NAME,
    IMAGENET_DS_NAME,
    ENV_ACTIVELOOP_TOKEN,
)
from .utils import tmp_datasets_dir, agree
import deeplake
import numpy as np
import math

from time import time
import math
import pytest
import random
import os
from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake

root = os.environ["VIZ_TEST_DATASETS_PATH"]


@pytest.mark.parametrize(
    "ds_name,queries",
    [
        (
            COCO_DS_NAME,
            [
                (
                    "select * where images_meta['file_name'] == '000000291712.jpg'",
                    1,
                    [59157],
                ),
                (
                    "select * where contains(images_meta['file_name'], '00000011')",
                    2240,
                    None,
                ),
                ("select * where images_meta['license'] == categories[0]", 2770, None),
                ("select * where categories[0] == images_meta['license']", 2770, None),
                (
                    "select * sample by max_weight(any(categories == 'person'): 10, True: 0.1)",
                    118287,
                    None,
                ),
                (
                    "select * sample by max_weight(contains(categories, 'person'): 10, True: 0.1)",
                    118287,
                    None,
                ),
                ("select * where not contains(categories, 'truck')", 112160, None),
                (
                    "select * where contains(\"stuff/super_categories\", 'sports')",
                    15534,
                    None,
                ),
                (
                    "(select * where contains(categories, 'car') limit 1000)\nunion (select * where contains(categories, 'motorcycle')\n limit 1000)",
                    1871,
                    None,
                ),
                ("select * where shape(images)[0] > 400", 100744, None),
                ("select * where shape(boxes)[0] > 10", 28004, None),
                (
                    "select * where any(logical_and(boxes[:,3]>500,\n categories == 'truck'))",
                    40,
                    None,
                ),
                ("SELECT * WHERE categories[0] == 'person'", 22497, None),
                ("SELECT * WHERE shape(categories)[0] == 0", 1021, None),
                ("SELECT * WHERE shape(boxes)[0] == 5", 7712, None),
                ("SELECT * WHERE shape(images)[0] > 200", 118078, None),
                ("SELECT * WHERE shape(images)[1] > 200", 118219, None),
                ("SELECT * WHERE shape(images)[2] == 3", 118060, None),
                ("SELECT * WHERE shape(images)[2] == 2", 0, None),
                (
                    "SELECT * WHERE ALL_STRICT(categories == 'person') or shape(categories)[0] == 0",
                    1375,
                    None,
                ),
                ("SELECT * WHERE ALL(categories == 'person')", 1375, None),
                ("select * where ALL_STRICT(categories == 'person')", 354, None),
                (
                    "select * where ALL_STRICT(\"pose/categories\" == 'person')",
                    64115,
                    None,
                ),
                ("select * where any(categories == 'banana')", 2243, None),
                ("select * where any(categories == 'person')", 64115, None),
                (
                    "select * where all_strict(logical_and(categories == 2, boxes[:,3] > 200))",
                    5,
                    None,
                ),
                (
                    "select * where all_strict(logical_and(categories == 'car', boxes[:,3] > 200))",
                    5,
                    None,
                ),
                (
                    "select * where any(logical_and(categories == 'car', boxes[:,3] > 200))",
                    1242,
                    None,
                ),
                (
                    "select * where all_strict(logical_or(categories == 'person', categories == 'banana'))",
                    873,
                    None,
                ),
                (
                    "select * where all(logical_and(categories == 'person', categories == 'person'))",
                    1375,
                    None,
                ),
                ("SELECT * where images_meta['license'] == 3", 32184, None),
                ("SELECT * where images_meta['id'] > 285529", 60365, None),
                ("SELECT * where 285529 <= images_meta['id']", 60365, None),
                ("SELECT * where contains(categories, '*ers*')", 64115, None),
                ("SELECT * where contains(categories, '?ers*')", 64115, None),
                (
                    "SELECT * ORDER BY shape(boxes)[0] limit 100",
                    100,
                    [
                        42,
                        92,
                        224,
                        302,
                        415,
                        693,
                        699,
                        722,
                        750,
                        817,
                        852,
                        861,
                        1068,
                        1252,
                        1473,
                        1483,
                        1502,
                        1527,
                        1638,
                        1921,
                        1928,
                        1935,
                        1999,
                        2034,
                        2063,
                        2069,
                        2110,
                        2191,
                        2488,
                        2511,
                        2602,
                        2687,
                        2740,
                        3074,
                        3093,
                        3177,
                        3314,
                        3367,
                        3417,
                        4137,
                        4270,
                        4324,
                        4465,
                        4549,
                        4569,
                        4670,
                        4796,
                        4906,
                        4974,
                        5152,
                        5240,
                        5253,
                        5298,
                        5336,
                        5464,
                        5763,
                        5777,
                        5868,
                        5875,
                        5901,
                        5955,
                        6061,
                        6071,
                        6182,
                        6454,
                        6537,
                        6779,
                        6835,
                        6841,
                        6858,
                        6986,
                        7177,
                        7262,
                        7277,
                        7342,
                        7486,
                        7676,
                        7752,
                        7858,
                        7866,
                        7915,
                        8003,
                        8082,
                        8165,
                        8283,
                        8377,
                        8652,
                        8992,
                        9233,
                        9288,
                        9372,
                        9395,
                        9560,
                        9701,
                        9940,
                        10088,
                        10171,
                        10176,
                        10201,
                        10358,
                    ],
                ),
                ("SELECT * order by images_meta['license']", 118287, None),
                (
                    "SELECT * order by images_meta['license'] limit 100",
                    100,
                    [
                        1,
                        7,
                        20,
                        24,
                        25,
                        31,
                        37,
                        41,
                        42,
                        46,
                        48,
                        52,
                        53,
                        56,
                        59,
                        60,
                        61,
                        65,
                        70,
                        71,
                        72,
                        73,
                        75,
                        79,
                        80,
                        86,
                        89,
                        98,
                        99,
                        107,
                        111,
                        117,
                        119,
                        120,
                        127,
                        138,
                        140,
                        148,
                        150,
                        156,
                        158,
                        162,
                        164,
                        167,
                        171,
                        172,
                        180,
                        191,
                        200,
                        205,
                        207,
                        210,
                        211,
                        214,
                        217,
                        228,
                        229,
                        233,
                        234,
                        244,
                        245,
                        248,
                        252,
                        253,
                        262,
                        273,
                        276,
                        280,
                        283,
                        302,
                        313,
                        315,
                        320,
                        326,
                        331,
                        336,
                        337,
                        339,
                        349,
                        351,
                        352,
                        354,
                        355,
                        359,
                        370,
                        383,
                        388,
                        389,
                        392,
                        393,
                        399,
                        400,
                        406,
                        408,
                        409,
                        416,
                        419,
                        430,
                        431,
                        432,
                    ],
                ),
            ],
        ),
        (
            MNIST_DS_NAME,
            [
                ("select * where contains(labels, '1')", 6742, None),
                ("select * where labels == 8", 5851, None),
                ("select * where labels % 3 == 1", 18849, None),
                ("select * where labels / 3 == 0", 18623, None),
                ("select * where labels * 3 == 6", 5958, None),
                ("select * where labels + 3 == 6", 6131, None),
                ("select * where labels - 3 == 6", 5949, None),
                (
                    "SELECT * WHERE labels == 0 expand by 2 2 as exp group by exp",
                    5923,
                    None,
                ),
                ("SELECT * WHERE labels == 0 expand by 2 2 as exp", 29614, None),
                (
                    "SELECT * WHERE labels == 0 expand by 2 2 OVERLAP true as exp",
                    29614,
                    None,
                ),
                (
                    "SELECT * WHERE labels == 0 expand by 2 2 OVERLAP false as exp",
                    24476,
                    None,
                ),
                (
                    "(select * where labels == 4 limit 10)\nunion (select * where labels == 5 limit 10)",
                    20,
                    None,
                ),
                ("SELECT * WHERE labels == 0", 5923, None),
                ("SELECT * WHERE SHAPE(images)[0] == 28", 60000, None),
                ("SELECT * WHERE SHAPE(images)[0] == 29", 0, None),
                (
                    "select * sample by max_weight(any(labels == 1): 10, True: 1)",
                    60000,
                    None,
                ),
                (
                    "(select * where labels == 4 limit 10) union (select * where labels == 5 limit 10) order by labels",
                    20,
                    [
                        2,
                        9,
                        20,
                        26,
                        53,
                        58,
                        60,
                        61,
                        64,
                        89,
                        0,
                        11,
                        35,
                        47,
                        65,
                        100,
                        132,
                        138,
                        145,
                        173,
                    ],
                ),
                (
                    "SELECT * SAMPLE BY sum_weight(labels == 0: 10, labels == 1: 5) LIMIT 100",
                    100,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY max_weight(labels == 0: 10, labels == 1: 5) LIMIT 100",
                    100,
                    None,
                ),
                ("SELECT * SAMPLE BY labels", 60000, None),
                (
                    "SELECT * SAMPLE BY sum_weight(labels > 5: 1, labels == 0: 3) REPLACE LIMIT 10000",
                    10000,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY max_weight(labels > 5: 1, labels == 0: 3) REPLACE LIMIT 10000",
                    10000,
                    None,
                ),
                ("SELECT * WHERE labels == 0 SAMPLE BY labels", 5923, None),
                ("SELECT * SAMPLE BY 1", 60000, None),
                ("SELECT * SAMPLE BY 0.1", 60000, None),
                ("SELECT * SAMPLE BY 0.1 LIMIT 30000", 30000, None),
                ("SELECT * SAMPLE BY 0.1 REPLACE FALSE LIMIT 190000", 60000, None),
                ("SELECT * SAMPLE BY 0.1 LIMIT 190000", 190000, None),
                ("SELECT * SAMPLE BY 0.1 REPLACE LIMIT 190000", 190000, None),
                (
                    "select * from(select * from (select *[0:10] from (select * group by labels) where shape(labels)[0] > 6000) ungroup by split) order by random()",
                    30,
                    None,
                ),
                ("SELECT * WHERE labels between 0 and 3", 24754, None),
                ("SELECT * WHERE labels IN (0, 2, 4, 6, 8)", 29492, None),
            ],
        ),
        (
            IMAGENET_DS_NAME,
            [
                ("select * where not contains(labels, 'llama')", 1279866, None),
                (
                    "(select * where contains(labels, 'chimpanzee') limit 10)\nunion (select * where contains(labels, 'patas') limit 10)",
                    20,
                    None,
                ),
                ("select * where shape(images)[0] > 400", 436859, None),
                ("select * where shape(boxes)[0] > 2", 12683, None),
                ("select * where all_strict(boxes[:,0]>500)", 674, None),
                ("SELECT * WHERE labels == 'bikini'", 1300, None),
                ("SELECT * WHERE contains(labels, 'bikini')", 1300, None),
                ("SELECT * WHERE contains(labels, '*iki*')", 2600, None),
                ("SELECT * WHERE contains(labels, '?iki*')", 1300, None),
                ("SELECT * WHERE SHAPE(boxes)[0] > 15", 2, None),
                (
                    "(SELECT * WHERE labels == 'bikini' LIMIT 10) UNION (SELECT * WHERE labels == 1 LIMIT 10) UNION (SELECT * WHERE labels == 43 LIMIT 10)",
                    30,
                    None,
                ),
                (
                    "select * sample by max_weight(contains(labels, 'banana'): 100, True: 0.1)",
                    1281166,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY sum_weight(labels == 'fur coat': 10, labels == 'bikini': 3) limit 0.1 PERCENT",
                    1281,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY max_weight(labels == 'fur coat': 10, labels == 'bikini': 3) limit 0.1 PERCENT",
                    1281,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY sum_weight(labels == 'fur coat': 10, labels == 'bikini': 3) replace",
                    1281166,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY sum_weight(labels == 'fur coat': 10, labels == 'bikini': 3) replace",
                    1281166,
                    None,
                ),
                (
                    "SELECT * SAMPLE BY max_weight(labels == 'fur coat': 10, True: 0.1)",
                    1281166,
                    None,
                ),
            ],
        ),
        (
            "hub://davitbun/places365-train-challenge",
            [
                ("SELECT * WHERE labels == 'hotel_room'", 32947, None),
                ("SELECT * ORDER by random()", 8026628, None),
            ],
        ),
    ],
)
def test_dataset_query_results(ds_name, queries):
    ds = dataset_to_libdeeplake(
        agree(IMAGENET_DS_NAME, token=os.getenv(ENV_ACTIVELOOP_TOKEN))
        if ds_name == IMAGENET_DS_NAME
        else deeplake.load(
            ds_name, token=os.getenv(ENV_ACTIVELOOP_TOKEN), read_only=True
        )
    )

    for query, result_size, result_indices in queries:
        print(f"\tRunning query: {query}")
        start = time()
        result = ds.query(query)
        print("\tQuery time: ", time() - start)
        assert len(result) == result_size
        assert result_indices is None or list(result.indexes) == result_indices
        start = time()
        result = ds.query(query)
        print("\tSecond Query time: ", time() - start)


def test_nested_queries():
    ds = api.dataset(MNIST_DS_NAME)
    dsv = ds.query("SELECT * WHERE labels < 4")
    dsv2 = dsv.query("SELECT * WHERE labels > 2")
    assert len(dsv2) == 6131


def test_query_selection_expressions(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "query_selection_slice"))
    ds.create_tensor("image", dtype="uint8", htype="image")
    with ds:
        ds.image.extend(np.random.randint(0, 255, (1, 200, 200, 3), dtype=np.uint8))

    ids = api.dataset(str(tmp_datasets_dir / "query_selection_slice"))
    view = ids.query("SELECT *[10:100,10:100]")
    assert np.all(view.image[0].numpy() == ids.image[0].numpy()[10:100, 10:100])
    view = ids.query("SELECT image[10:100,10:100]")
    assert np.all(view.image[0].numpy() == ids.image[0].numpy()[10:100, 10:100])
    view = ids.query("SELECT image + 1")
    assert np.all(view.image[0].numpy() == ids.image[0].numpy() + 1)
    with pytest.raises(RuntimeError):
        view = ids.query("SELECT image, l2_norm(image)")
    view = ids.query("SELECT image, l2_norm(image) as score")
    assert np.all(view.image[0].numpy() == ids.image[0].numpy())
    assert np.all(
        (
            view.score[0].numpy()
            - np.linalg.norm(ids.image[0].numpy().reshape(120000), 2)
        )
        / view.score[0].numpy()
        < 0.0001
    )


def test_query_order_by_array_error(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "query_order_by_array"))
    with ds:
        ds.create_tensor("boxes", dtype="uint8", htype="bbox")
        ds.boxes.extend(
            np.array(
                [
                    [0, 1, 2, 3],
                    [1, 2, 10, 2],
                    [10, 2, 10, 2],
                    [5, 2, 10, 2],
                ]
            )
        )
    ids = api.dataset(str(tmp_datasets_dir / "query_order_by_array"))
    with pytest.raises(RuntimeError):
        view = ids.query("SELECT * order by boxes")


def test_query_group_by_result_access(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "query_group_by_access"))
    with ds:
        ds.create_tensor(
            "image",
            dtype="uint8",
            max_chunk_size=10 * 1024,
            htype="image",
        )
        ds.create_tensor("label", dtype="int32", htype="class_label")
        ds.image.extend(np.random.randint(0, 255, (1000, 28, 28, 1), np.uint8))
        labels = [i % 4 for i in range(1000)]
        ds.label.extend(np.array(labels))

    ids = api.dataset(str(tmp_datasets_dir / "query_group_by_access"))
    view = ids.query("SELECT * group by label")
    assert len(view) == 4
    for i in range(4):
        assert len(view.tensors[0][i].numpy()) == 250
        for j in range(250):
            assert np.all(view.tensors[0][i].numpy()[j] == ids.image[4 * j + i].numpy())


def test_polygon_query(tmp_datasets_dir):
    ids = api.dataset(root + "/polygon_dataset")
    dsv = ids.query("SELECT * WHERE any(polygon[:,:,0] > 300)")
    assert len(dsv) == 8
    dsv = ids.query("SELECT * WHERE all(polygon[:,:,0] < 300)")
    assert len(dsv) == 2
    dsv = ids.query("SELECT * WHERE all(polygon[:,:,1] < 300)")
    assert len(dsv) == 4
    dsv = ids.query("SELECT * WHERE all(polygon[:,:,1] > 0)")
    assert len(dsv) == 10


def test_group_ungroup_queries(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "frames_dataset_for_query"))
    ds.create_tensor("frame", htype="generic", dtype="int32")
    ds.create_tensor("video_id", htype="generic", dtype="int32")
    ds.create_tensor("label", htype="class_label", dtype="int32")
    with ds:
        ds.frame.extend(np.array([int(1000 * random.uniform(0.0, 1.0))] * 10000))
        ds.video_id.extend(np.array([int(i / 20) for i in range(10000)]))
        ds.label.extend(np.array([i % 2 for i in range(10000)]))

    ids = api.dataset(str(tmp_datasets_dir / "frames_dataset_for_query"))

    view = ids.query("SELECT * GROUP BY video_id, label")
    assert len(view) == 1000
    for i in range(len(view)):
        for j in range(len(view.frame[i])):
            l = i % 2
            k = int(i / 2)
            assert np.all(
                view.frame[i].numpy()[j] == ids.frame[20 * k + l + 2 * j].numpy()
            )

    view2 = view.query("SELECT * UNGROUP BY SPLIT")
    assert len(view2) == 10000
    for i in range(len(view2)):
        l = int(i / 10) % 2
        k = int(i / 20)
        j = i % 10
        assert np.all(view2.frame[i].numpy() == ids.frame[20 * k + l + 2 * j].numpy())


def test_uneven_tensors_query(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "uneven_tensors_dataset"))
    ds.create_tensor("tensor_1")
    ds.create_tensor("tensor_2")

    with ds:
        ds.tensor_1.extend(np.arange(100))
        ds.tensor_2.extend(np.arange(50))

    ids = api.dataset(str(tmp_datasets_dir / "uneven_tensors_dataset"))
    view = ids.query("select * where tensor_1==1 and tensor_2==1")
    assert len(view) == 1

    view_2 = ids.query("select * where tensor_1==80")
    assert len(view_2) == 1
    assert np.all(view_2.tensor_1[0].numpy() == 80)
    assert len(view_2.tensor_2[0].numpy()) == 0


def test_order_group_by_string(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "string_ds_for_query"))
    with ds:
        ds.create_tensor("filename", htype="text")
        ds.filename.extend(
            [
                "v3.jpg",
                "v3.jpg",
                "v3.jpg",
                "v1.jpg",
                "v2.jpg",
                "v2.jpg",
                "v1.jpg",
                "v2.jpg",
                "v1.jpg",
            ]
        )
    ids = api.dataset(str(tmp_datasets_dir / "string_ds_for_query"))
    view = ids.query("SELECT * ORDER BY filename")
    assert len(view) == 9
    assert view.filename[0].numpy() == "v1.jpg"
    assert view.filename[1].numpy() == "v1.jpg"
    assert view.filename[2].numpy() == "v1.jpg"
    assert view.filename[3].numpy() == "v2.jpg"
    assert view.filename[4].numpy() == "v2.jpg"
    assert view.filename[5].numpy() == "v2.jpg"
    assert view.filename[6].numpy() == "v3.jpg"
    assert view.filename[7].numpy() == "v3.jpg"
    assert view.filename[8].numpy() == "v3.jpg"
    view = ids.query("SELECT * GROUP BY filename")
    assert len(view) == 3
    assert view.filename[0].numpy() == ["v3.jpg", "v3.jpg", "v3.jpg"]
    assert view.filename[1].numpy() == ["v1.jpg", "v1.jpg", "v1.jpg"]
    assert view.filename[2].numpy() == ["v2.jpg", "v2.jpg", "v2.jpg"]


def test_invalid_class_name_query(tmp_datasets_dir):
    ds = deeplake.dataset(
        tmp_datasets_dir / "invalid_class_names_dataset", overwrite=True
    )
    with ds:
        ds.create_tensor(
            "label",
            dtype=np.int32,
            htype="class_label",
            sample_compression=None,
        )
        for i in range(5):
            ds.label.append("bike")
        for i in range(5):
            ds.label.append("car")

    ids = api.dataset(str(tmp_datasets_dir / "invalid_class_names_dataset"))
    v = ids.query("SELECT * WHERE label == 'bike'")
    assert len(v) == 5
    v = ids.query("SELECT * WHERE label == 'car'")
    assert len(v) == 5
    with pytest.raises(RuntimeError):
        v = ids.query("SELECT * WHERE label == 'plane'")

    v = ids.query("SELECT * WHERE contains(label, 'bike')")
    assert len(v) == 5
    v = ids.query("SELECT * WHERE contains(label, 'car')")
    assert len(v) == 5
    v = ids.query("SELECT * WHERE contains(label, '*ike')")
    assert len(v) == 5
    with pytest.raises(RuntimeError):
        v = ids.query("SELECT * WHERE contains(label, 'plane')")
    with pytest.raises(RuntimeError):
        v = ids.query("SELECT * WHERE contains(label, '*lan*')")


def test_order_by_asc_desc_query(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "order_by_query_dataset"))
    with ds:
        ds.create_tensor(
            "label",
            dtype="int32",
            htype="class_label",
        )
        arr = np.arange(10)
        random.shuffle(arr)
        ds.label.extend(arr.reshape(10, 1))

    ids = api.dataset(str(tmp_datasets_dir / "order_by_query_dataset"))
    v = ids.query("SELECT * WHERE label > 3 ORDER BY label")
    assert np.all(v.label.numpy() == [[4], [5], [6], [7], [8], [9]])
    v = ids.query("SELECT * WHERE label > 3 ORDER BY label ASC")
    assert np.all(v.label.numpy() == [[4], [5], [6], [7], [8], [9]])
    v = ids.query("SELECT * WHERE label > 3 ORDER BY label DESC")
    assert np.all(v.label.numpy() == [[9], [8], [7], [6], [5], [4]])


def test_embeddings_positive_and_negative(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "embeddings_dataset"))
    with ds:
        ds.create_tensor(
            "embeddings",
            dtype="int32",
            htype="generic",
        )
        arr = np.arange(3, dtype=np.int32)
        ds.embeddings.extend(arr.reshape(1, 3))

    ids = api.dataset(str(tmp_datasets_dir / "embeddings_dataset"))
    v = ids.query("SELECT *, l1_norm(embeddings - ARRAY[1, 1, 1]) as score")
    assert v.score[0].numpy() == 2
    v = ids.query("SELECT *, l1_norm(embeddings - ARRAY[0, 1, 2]) as score")
    assert v.score[0].numpy() == 0
    v = ids.query("SELECT *, l1_norm(embeddings - ARRAY[-1, 0, 1]) as score")
    assert v.score[0].numpy() == 3


def test_row_number(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "row_number_dataset"))
    with ds:
        ds.create_tensor(
            "dummy",
            dtype="int32",
            htype="generic",
        )
        ds.dummy.extend([9, 8, 7, 6, 5, 4, 3, 2, 1])

    ids = api.dataset(str(tmp_datasets_dir / "row_number_dataset"))
    v = ids.query(
        "SELECT * FROM (SELECT *, ROW_NUMBER() as sample_index) WHERE dummy < 7"
    )
    assert len(v) == 6
    for i in range(len(v)):
        assert v.sample_index[i].numpy() == i + 3

    v = ids.query("SELECT * WHERE ROW_NUMBER() % 3 == 1")
    assert len(v) == 3
    assert v.dummy[0].numpy() == 8
    assert v.dummy[1].numpy() == 5
    assert v.dummy[2].numpy() == 2


def test_bytes_crash(tmp_datasets_dir):
    ds = api.dataset_writer(str(tmp_datasets_dir / "embedding_dataset_text"))
    with ds:
        ds.create_tensor("embeddings", htype="generic")
        ds.create_tensor("text", htype="text")
        ds.text.extend(["hello world"] * 10)
        ds.embeddings.extend(np.random.randint(0, 1, size=(10, 100)))

    ids = api.dataset(str(tmp_datasets_dir / "embedding_dataset_text"))
    v = ids.query("SELECT *, l2_norm(embeddings) as score LIMIT 10")
    for i in v.text:
        assert i.bytes().decode("utf-8") == "hello world"


def test_empty_tensors_ds_crash(tmp_datasets_dir):
    ds = deeplake.empty(tmp_datasets_dir / "empty_tensors_ds", overwrite=True)
    ids = api.dataset(str(tmp_datasets_dir / "empty_tensors_ds"))
    with pytest.raises(RuntimeError):
        v = ids.query("SELECT * WHERE labels == 1")


def test_tag_htype_query():
    ds = api.dataset(root + "/tag_dataset")
    view = ds.query("select * where contains(abc2, 'tag 41')")
    assert len(view) == 1
    view = ds.query("select * where contains(abc2, 'tag 4')")
    assert len(view) == 3
    view = ds.query("select * where abc2[1] == 'tag 1'")
    assert len(view) == 4


def test_json_keys_query():
    ds = api.dataset(root + "/twitter-algorithm-metadata")
    view = ds.query("select keys(metadata) as json_keys")
    assert view.json_keys[0].numpy() == ["source"]


def test_vectorstore_crash():
    ds = api.dataset(root + "/empty_vectorstore/")
    view = ds.query(
        "select embedding, metadata, text, id, score from (select *, COSINE_SIMILARITY(embedding, ARRAY[1]) as score order by COSINE_SIMILARITY(embedding, ARRAY[1]) DESC limit 4) LIMIT 4"
    )
    assert len(view.embedding.numpy()) == 0
    assert len(view.score.numpy()) == 0

def test_join_failure():
    try:
        api.tql.query('SELECT * from "hub://missing_org/missing_ds_1" as first INNER JOIN "hub://missing_org/missing_ds_2" as second USING(missing_label)', token='')
    except RuntimeError as e:
        assert e.args == ('{"description":"The dataset you\'re trying to read doesn\'t exist."}', )

    try: 
        api.tql.query(f'SELECT * from "{MNIST_DS_NAME}" as first INNER JOIN "hub://missing_org/missing_ds_2" as second USING(missing_label)', token='')
    except RuntimeError as e:
        assert e.args == ('{"description":"The dataset you\'re trying to read doesn\'t exist."}', )

    try:
        api.tql.query(f'SELECT * from "hub://missing_org/missing_ds_2" as first INNER JOIN  "{MNIST_DS_NAME}" as second USING(missing_label)', token='')
    except RuntimeError as e:
        assert e.args == ('{"description":"The dataset you\'re trying to read doesn\'t exist."}', )
