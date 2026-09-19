from pathlib import Path

import numpy as np
import pytest

from tre_viewer.data import RunRef
from tre_viewer.drr import parse_geometry_xml, r3_ct_physical_landmarks


def test_bad_geometry_does_not_silently_shift_projection_indices(tmp_path):
    path = tmp_path / "Geometry.xml"
    path.write_text("""<Geometry>
      <Projection><Matrix>1 0 0 0 0 1 0 0 0 0 1 0</Matrix></Projection>
      <Projection><Matrix>1 2</Matrix></Projection>
      <Projection><Matrix>1 0 0 0 0 1 0 0 0 0 1 0</Matrix></Projection>
    </Geometry>""")
    with pytest.raises(ValueError, match="Projection 2"):
        parse_geometry_xml(path)


def test_projection_coordinates_respect_native_frame_and_itk_direction(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    train = tmp_path / "train"
    train.mkdir()
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    image.SetSpacing((2., 3., 4.))
    image.SetOrigin((10., 20., 30.))
    image.SetDirection((-1., 0., 0., 0., 1., 0., 0., 0., -1.))
    sitk.WriteImage(image, str(train / "CT_06.mha"))
    run = RunRef("test", 1, "DIR_C01", tmp_path, "native")
    result, _ = r3_ct_physical_landmarks(run, np.array([[1., 2., 3.]]))
    np.testing.assert_allclose(result, [[8., 26., 18.]])
