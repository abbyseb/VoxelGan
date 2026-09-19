import numpy as np
import pytest

from tre_viewer.warp_dvf import lung_mae, pack_arrows_zyx, upsample_dvf_to_pack, upsample_to_pack, warp_pull


def test_identity_and_translation_pull_warp():
    source = np.indices((4, 5, 6))[2].astype(float)
    dvf = np.zeros((*source.shape, 3))
    np.testing.assert_allclose(warp_pull(source, dvf), source)
    dvf[..., 0] = 1
    np.testing.assert_allclose(warp_pull(source, dvf), np.minimum(source + 1, 5))


def test_r3_scalar_restores_pack_anatomy():
    # R3 (z,y,x) corresponds to flipped pack (y,z,x).
    pack = np.arange(4 * 6 * 8, dtype=float).reshape(4, 6, 8)
    r3 = pack[::-1, ::-1, :].transpose(1, 0, 2)
    np.testing.assert_allclose(upsample_to_pack(r3, pack.shape, frame="r3"), pack)


def test_sub_grid_resampling_uses_same_index_scale_as_landmarks():
    sub = np.indices((4, 4, 4))[2].astype(float)
    out = upsample_to_pack(sub, (8, 8, 8))
    assert out[0, 0, 4] == 2  # pack x=4 -> sub x=4*4/8


@pytest.mark.parametrize("frame,expected", [("native", [2., 6., 12.]), ("r3", [2., -9., -8.])])
def test_vector_lengths_and_directions_are_pack_voxels(frame, expected):
    dvf = np.ones((4, 4, 4, 3)) * [1., 2., 3.]
    pack_shape = (16, 12, 8)
    out = upsample_dvf_to_pack(dvf, pack_shape, frame=frame)
    np.testing.assert_allclose(out[3, 4, 5], expected)
    arrows = pack_arrows_zyx(dvf, pack_shape, frame=frame, slice_index=5, step=3)
    np.testing.assert_allclose(arrows[:, 0, 0], 5)
    np.testing.assert_allclose(arrows[:, 1], np.tile(expected[::-1], (len(arrows), 1)))


def test_bad_lung_mask_never_silently_reports_whole_volume_mae():
    a = np.ones((3, 3, 3))
    with pytest.raises(ValueError, match="shape"):
        lung_mae(a, a, np.ones((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        lung_mae(a, a, np.zeros_like(a))
