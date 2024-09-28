from indra import api
import os

root = os.environ["VIZ_TEST_DATASETS_PATH"]

def test_nifti_gz():
    ds = api.dataset(os.path.join(root, 'nifti/'))
    d = ds.t2[0].numpy()
    assert(d.shape == (240, 240, 155))

    ds = api.dataset(os.path.join(root, 'nifti_resampled_gz/'))
    d = ds.scan[0].numpy()
    assert(d.shape == (49, 49, 32))

    d = ds.segmentation[0].numpy()
    assert(d.shape == (49, 49, 32))

    d = ds.scan[1].numpy()
    assert(d.shape == (87, 56, 45))

def test_nifti():
    ds = api.dataset(os.path.join(root, 'nifti_resampled/'))
    d = ds.scan[0].numpy()
    assert(d.shape == (87, 56, 45))
